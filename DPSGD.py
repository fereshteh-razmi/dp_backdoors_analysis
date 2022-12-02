#### https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset, TensorDataset, DataLoader
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine
import os, copy, random
from typing import Any
from simple_parsing import ArgumentParser
from lib.lib import  Net, CustomTensorDataset
from attack import Attack, AttackConfig
import warnings
warnings.simplefilter("ignore")

def main(
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    batch_size: int,
    max_physical_batch_size: int,
    num_epochs: int,
    features_path: str,
    base_dir: str
):

    checkpoint_path = os.path.join(base_dir, "model_DPSGD.ckp")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    #### Read features
    dataloaders_dict = {}
    image_datasets = {}


    def convert_numpy_to_data_loader(x,y,batch_size=32,shuffle=True):
        tensor_x, tensor_y = torch.Tensor(x), torch.Tensor(y)
        # _, y_classes = torch.max(tensor_y, 1)
        dataset = TensorDataset(tensor_x, tensor_y) # create your datset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, dataloader


    # train_x_path = os.path.join(features_path,'train_tf.npy') ### data from tensorflow resnet50 output
    # train_y_path = os.path.join(features_path,'label_train_tf.npy')
    # train_x = np.load(train_x_path)
    # train_y = np.load(train_y_path)
    # dataset, dataloader = convert_numpy_to_data_loader(train_x, train_y ,batch_size=batch_size, shuffle=True)
    # image_datasets['train'] = copy.deepcopy(dataset)
    # dataloaders_dict['train'] = copy.deepcopy(dataloader)
    #
    # val_x_path = os.path.join(features_path,'test_tf.npy')
    # val_y_path = os.path.join(features_path,'label_test_tf.npy')
    # val_x = np.load(val_x_path)
    # val_y = np.load(val_y_path)
    # dataset, dataloader = convert_numpy_to_data_loader(val_x, val_y ,batch_size=batch_size, shuffle=True)
    # image_datasets['val'] = copy.deepcopy(dataset)
    # dataloaders_dict['val'] = copy.deepcopy(dataloader)
    #
    # val_x_pois_path = os.path.join(features_path,'test_pois_tf.npy')
    # val_y_pois_path = os.path.join(features_path,'label_test_pois_tf.npy')
    # val_x_pois = np.load(val_x_pois_path)
    # val_y_pois = np.load(val_y_pois_path)
    # dataset, dataloader = convert_numpy_to_data_loader(val_x_pois, val_y_pois ,batch_size=batch_size, shuffle=True)
    # image_datasets['val_pois'] = copy.deepcopy(dataset)
    # dataloaders_dict['val_pois'] = copy.deepcopy(dataloader)

    #### Load datasets
    dataloaders_dict = {}

    train_x_path = os.path.join(features_path,'train.npy')
    train_y_path = os.path.join(features_path,'label_train.npy')
    train_x = torch.tensor(np.load(train_x_path))
    train_y = torch.tensor(np.load(train_y_path))
    train_dataset_vf = CustomTensorDataset(tensors=(train_x, train_y))#, transform=transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    dataloaders_dict['train'] = DataLoader(train_dataset_vf, batch_size=batch_size, shuffle=True)

    val_x_path = os.path.join(features_path,'test.npy')
    val_y_path = os.path.join(features_path,'label_test.npy')
    val_x = torch.tensor(np.load(val_x_path))
    val_y = torch.tensor(np.load(val_y_path))
    val_dataset = CustomTensorDataset(tensors=(val_x, val_y))#, transform=transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    dataloaders_dict['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    val_pois_x_path = os.path.join(features_path,'test_pois.npy')
    val_pois_y_path = os.path.join(features_path,'label_test_pois.npy')
    val_pois_x = torch.tensor(np.load(val_pois_x_path))
    val_pois_y = torch.tensor(np.load(val_pois_y_path))
    val_pois_dataset = CustomTensorDataset(tensors=(val_pois_x, val_pois_y))#, transform=transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    dataloaders_dict['val_pois'] = DataLoader(val_pois_dataset, batch_size=batch_size, shuffle=True)


    classifier = Net()
    classifier = classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.1
    # optimizer = optim.Adam(classifier.parameters(), lr=lr)
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.RMSprop(classifier.parameters(), lr=lr)#0.001

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=classifier,
        optimizer=optimizer,
        data_loader=dataloaders_dict['train'],
        epochs=num_epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )
    # print(f"Using sigma={optimizer.noise_multiplier} and C={max_grad_norm}")
    # print(f"Using eps={epsilon} and C={max_grad_norm}")

    def accuracy(preds, labels):
        return (preds == labels).mean()

    best_acc = 0.0
    best_sr = 0.0
    for epoch in range(num_epochs):

        for phase in ['train', 'val','val_pois']:
            
            if phase == 'train':
                losses = []
                top1_acc = []

                model.train()
                with BatchMemoryManager(
                    data_loader=train_loader,
                    max_physical_batch_size=max_physical_batch_size,
                    optimizer=optimizer
                ) as memory_safe_data_loader:

                    for inputs, labels in memory_safe_data_loader:
                        inputs = inputs.type(torch.FloatTensor).to(device)
                        labels = F.one_hot(labels, num_classes=10)
                        labels = labels.type(torch.FloatTensor).to(device)

                        optimizer.zero_grad()
                        with torch.set_grad_enabled(True):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                            true_class = np.argmax(labels.data.detach().cpu().numpy(), axis=1)
                            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)

                            acc = accuracy(preds, true_class)
                            losses.append(loss.item())
                            top1_acc.append(acc)

                            loss.backward()
                            optimizer.step()
            else:
                top1_acc = []
                model.eval()
                for inputs, labels in dataloaders_dict[phase]:
                    inputs = inputs.type(torch.FloatTensor).to(device)
                    labels = F.one_hot(labels, num_classes=10)
                    labels = labels.type(torch.FloatTensor).to(device)
                    with torch.no_grad():
                        outputs = model(inputs)
                        true_class = np.argmax(labels.data.detach().cpu().numpy(), axis=1)
                        preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                        acc = accuracy(preds, true_class)
                        top1_acc.append(acc)

                if phase == 'val_pois':
                    best_sr = np.mean(top1_acc)
                if phase == 'val':
                    best_acc = np.mean(top1_acc)


    print(
        f"Test Acc@1: {np.mean(best_acc) * 100:.6f} "
        f"Pois Acc@1: {np.mean(best_sr) * 100:.6f} ")

    torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--epsilon", type=float, default=4, help="Epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help=" it should be set to be less than the inverse of the size of the training dataset. In this tutorial, it is set to 1e-5 as the CIFAR10 dataset has 50,000 training points.")
    parser.add_argument("--max_grad_norm", type=float, default=2, help="Clipping norm: The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step")
    parser.add_argument("--batch_size", type=int, default=512, help="logical batch size (which defines how often the model is updated and how much DP noise is added")
    parser.add_argument("--max_physical_batch_size", type=int, default=128, help="physical batch size (which defines how many samples do we process at a time)")
    parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--feature_dir", type=str, default="./data/mnist/b4_50/features", help="Feature dir")
    parser.add_argument("--base_dir", type=str, default="./data/mnist/b4_50/checkpoints", help="Base dir")

    args = parser.parse_args()

    main(
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        max_physical_batch_size=args.max_physical_batch_size,
        num_epochs=args.num_epochs,
        features_path=args.feature_dir,
        base_dir=args.base_dir
    )
