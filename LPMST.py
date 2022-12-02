
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,logging,warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from scipy.special import softmax
from dataclasses import dataclass
import copy
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data.distributed
from lib.lib import Net, CustomTensorDataset
from attack import Attack, AttackConfig


os.environ['PYTHONHASHSEED']=str(0)


#######################################################################
# Settings
#######################################################################
MAX_GRAD_INF = 1e6

@dataclass
class LabelPrivacy:
    epsilon: float = 10000
    delta: float = 1e-5
    temperature: float = 0.5
    data_splits = [0.5,0.5]
    rr_prior_seed: int = 2022


@dataclass
class Learning:
    lr = [0.001, 0.001]
    batch_size: int = 128
    epochs: int = 30
    momentum: float = 0.9
    weight_decay: float = 1e-4
    random_aug: bool = False


@dataclass
class Settings:
    dataset: str = "cifar10"
    num_classes = 10
    privacy: LabelPrivacy = LabelPrivacy()
    learning: Learning = Learning()
    features_dir: str = "./data/mnist/b4_50/features/"
    data_dir_root: str = "./data/mnist/b4_50/lp2st/checkpoints/"
    seed: int = 0


def partition_dataset(data, labels, nb_stages, curr_stage, data_splits):

    assert len(data) == len(labels)
    assert int(curr_stage) < int(nb_stages)

    if curr_stage == 0:
        all_prev_rates = 0.0
    else:
        all_prev_rates = sum([r for r in data_splits[:curr_stage]])
    start = int(all_prev_rates * len(data))
    end = int((all_prev_rates + data_splits[curr_stage]) * len(data))

    partition_data = data[start:end]
    partition_labels = labels[start:end]
    # print(f"Stage: {curr_stage} --> start index: {start}, end idnex: {end}")

    return partition_data, partition_labels, start, end


# ============================================================
def RR(curr_labels, epsilon, n_classes):
    new_labels = []
    rng = np.random.RandomState(seed=2019)

    class_labels = np.argmax(curr_labels, axis=1)

    for y_true in class_labels:
        rate = 1 / (np.exp(epsilon) + n_classes - 1)
        prob = np.zeros(n_classes) + rate
        prob[y_true] = 1 - rate * (n_classes - 1)
        y_hat = rng.choice(n_classes, 1, p=prob)
        new_labels.append(y_hat)

    new_labels = np.array(new_labels)   ### Class numbers
    new_labels_cat = np.squeeze(np.eye(n_classes)[new_labels])

    return n_classes, new_labels_cat


# ============================================================
def RR_With_Prior(curr_dataloader, device, classifier, temperature, epsilon, n_classes, pois_inds, rr_prior_seed=2022):
    new_labels = []
    soft_k = 0.0
    count = 0

    rng = np.random.RandomState(seed=rr_prior_seed)
    classifier.eval()
    for X_batch, y_batch in curr_dataloader:
        X_batch = X_batch.type(torch.FloatTensor).to(device)
        # labels = labels.type(torch.LongTensor)
        y_batch = y_batch.to(device)

        logits = classifier(X_batch)
        logits /= temperature
        y_batch = np.argmax(y_batch.detach().cpu().numpy(), axis=1)
        logits = logits.detach().cpu().numpy()
        p_last_model = softmax(logits, axis=1)

        for sample_ind in range(len(y_batch)):
            pi = p_last_model[sample_ind]
            y_true = y_batch[sample_ind]
            is_pois = False
            if count in pois_inds:
                is_pois = True
            k, y_hat = RR_With_Prior_each_sample(epsilon, pi, y_true, rng, is_pois)
            soft_k += k
            new_labels.append(y_hat)
            # if count in pois_inds:
                # print(f"K: {k}, y_true: {y_true}, y_hat: {y_hat}")
            count += 1

    new_labels = np.array(new_labels)   ### Class numbers
    new_labels_cat = np.squeeze(np.eye(n_classes)[new_labels])
    soft_k = int(soft_k/count)

    return soft_k, new_labels_cat


def RR_With_Prior_each_sample(epsilon, prior, y_true, rng, is_pois):

    idx_sort = np.flipud(np.argsort(prior))
    prior_sorted = prior[idx_sort]
    tmp = np.exp(-epsilon)
    wks = [np.sum(prior_sorted[:(k+1)]) / (1 + (k-1)*tmp)
         for k in range(len(prior))]
    optim_k = np.argmax(wks) + 1

    adjusted_prior = np.zeros_like(prior) + tmp / (1 + (optim_k-1)*tmp)
    adjusted_prior[y_true] = 1 / (1 + (optim_k-1)*tmp)
    adjusted_prior[idx_sort[optim_k:]] = 0
    adjusted_prior /= np.sum(adjusted_prior)  # renorm in case y not in topk
    rr_label = rng.choice(len(prior), 1, p=adjusted_prior)
    return optim_k, rr_label


# ============================================================

def train(model, dataloaders, criterion, optimizer, device, checkpoint_path, scheduler, num_epochs=25, lr=0.001, t=0):
    best_acc = 0.0
    best_sr = 0.0
    for epoch in range(num_epochs):

        for phase in ['train', 'val','val_pois']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, true_class = torch.max(labels.data, 1)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == true_class)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'val_pois':
                best_sr = epoch_acc
            if phase == 'val':
                best_acc = epoch_acc

    torch.save(model.state_dict(), checkpoint_path)
    return model, best_acc.item(), best_sr.item()


# ============================================================
def  main(settings):


    checkpoint_path = os.path.join(settings.data_dir_root)
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_path, 'model.tar')

    features_path = os.path.join(settings.features_dir)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epsilon = settings.privacy.epsilon
    criterion = torch.nn.CrossEntropyLoss()

    classifier = Net()
    classifier = classifier.to(device)

    ### Get Backdoor Indices
    ac = AttackConfig()
    backdoor_attack = Attack(
                        poisoning_rate=ac.poison_rate,
                        trigger_size=ac.trigger_size,
                        target_class=ac.target_class,
                        source_class=ac.source_class,
                        input_size=32,
                )
    backdoor_indices = backdoor_attack.get_backdoor_indices(ac.poisons_indices_file_name)

    #### Load datasets
    dataloaders_dict = {}
    image_datasets = {}

    def convert_numpy_to_data_loader(x,y,batch_size=32,shuffle=True):
        tensor_x, tensor_y = torch.Tensor(x), torch.Tensor(y)
        dataset = CustomTensorDataset(tensors=(tensor_x,tensor_y))#, transform=transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, dataloader

    train_x_path = os.path.join(features_path,'train.npy')
    train_y_path = os.path.join(features_path,'label_train.npy')
    train_x = torch.tensor(np.load(train_x_path))
    train_y = torch.tensor(np.load(train_y_path))
    train_y = np.eye(10, dtype=float)[train_y]
    image_datasets['train'] = []
    dataloaders_dict['train'] = []

    val_x_path = os.path.join(features_path,'test.npy')
    val_y_path = os.path.join(features_path,'label_test.npy')
    val_x = torch.tensor(np.load(val_x_path))
    val_y = torch.tensor(np.load(val_y_path))
    val_y = np.eye(10, dtype=float)[val_y]
    dataset, dataloader = convert_numpy_to_data_loader(val_x, val_y ,batch_size=settings.learning.batch_size, shuffle=True)
    image_datasets['val'] = copy.deepcopy(dataset)
    dataloaders_dict['val'] = copy.deepcopy(dataloader)

    val_pois_x_path = os.path.join(features_path,'test_pois.npy')
    val_pois_y_path = os.path.join(features_path,'label_test_pois.npy')
    val_pois_x = torch.tensor(np.load(val_pois_x_path))
    val_pois_y = torch.tensor(np.load(val_pois_y_path))
    val_pois_y = np.eye(10, dtype=float)[val_pois_y]
    dataset, dataloader = convert_numpy_to_data_loader(val_pois_x, val_pois_y ,batch_size=settings.learning.batch_size, shuffle=True)
    image_datasets['val_pois'] = copy.deepcopy(dataset)
    dataloaders_dict['val_pois'] = copy.deepcopy(dataloader)


    T = len(settings.privacy.data_splits)
    masks_by_now = []
    data_by_now = np.empty((0,1,28,28))
    labels_by_now = np.empty((0,settings.num_classes))
    logits_by_now = np.empty((0,settings.num_classes))

    data_splits = settings.privacy.data_splits
    temperature = settings.privacy.temperature

    for t in range(T):
        print('----------- iter {} ------------'.format(t))
        optimizer = optim.SGD(classifier.parameters(),
                              lr=settings.learning.lr[t], momentum=settings.learning.momentum, weight_decay=settings.learning.weight_decay)


        data, labels, start_ind, end_ind = partition_dataset(train_x, train_y, nb_stages=T, curr_stage=t, data_splits=data_splits)
        dataset, dataloader = convert_numpy_to_data_loader(data, labels, batch_size=settings.learning.batch_size, shuffle=False)
        curr_dataloader = copy.deepcopy(dataloader)

        curr_backdoor_inds = np.array([b_ind for b_ind in backdoor_indices if start_ind<=b_ind and b_ind<end_ind]) - start_ind
        if t == 0:
            soft_k, new_labels = RR(labels, epsilon, settings.num_classes)
        else:
            soft_k, new_labels = RR_With_Prior(curr_dataloader, device, classifier, temperature, epsilon,
                                               settings.num_classes, curr_backdoor_inds,
                                               rr_prior_seed=settings.privacy.rr_prior_seed)
        if t != 0:
            masks_by_now[:] = True
            n_filtered = 0
            for j, logit in enumerate(logits_by_now):
                topk_idx = logit.argsort()[::-1][:soft_k]
                y = labels_by_now[j][:]
                if np.isclose(np.sum(y[topk_idx]), 0):
                    masks_by_now[j] = False
                    n_filtered += 1

        masks_by_now = np.concatenate((masks_by_now, np.ones(len(data), dtype=bool)))
        data_by_now = np.concatenate((data_by_now, data))
        labels_by_now = np.concatenate((labels_by_now,new_labels)).astype(np.int32)

        filtered_x = copy.deepcopy(data_by_now[np.where(masks_by_now),:])[0]
        filtered_y = np.squeeze(copy.deepcopy(labels_by_now[np.where(masks_by_now),:]))

        dataset_by_now, dataloader_by_now = convert_numpy_to_data_loader(filtered_x, filtered_y,
                                                                         batch_size=settings.learning.batch_size, shuffle=True)
        image_datasets['train'] = dataset_by_now
        dataloaders_dict['train'] = dataloader_by_now

        classifier, acc, sr = train(classifier, dataloaders_dict, criterion, optimizer, device,
                                                   checkpoint_path, None, num_epochs=settings.learning.epochs, lr=settings.learning.lr[t],t=t)

        if t == T-1:
            continue

        classifier.eval()
        curr_logits = np.empty((0,settings.num_classes))
        for X_batch, y_batch in curr_dataloader:
            X_batch = X_batch.type(torch.FloatTensor).to(device)
            logit = classifier(X_batch)
            logit = np.array(logit.detach().cpu().numpy())
            curr_logits = np.concatenate((curr_logits,logit),axis=0)
        logits_by_now = np.concatenate((logits_by_now, curr_logits),axis=0)

    print('Test Acc: {:.4f}, Poisoned Success rate: {:.4f}'.format(acc,sr))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CIFAR LabelDP Training with ALIBI")
    parser.add_argument("--rrp_seed", default=2022, type=int, help="Seed that's used for RR_with_prior")
    parser.add_argument("--eps", default=1000, type=float, help="DP Epsilon")
    parser.add_argument("--temp", default=0.5, type=float, help="DP Epsilon")

    args = parser.parse_args()

    privacy = LabelPrivacy(rr_prior_seed=args.rrp_seed,epsilon=args.eps,temperature=args.temp)
    learning = Learning()
    settings = Settings(privacy=privacy)
    main(settings)
