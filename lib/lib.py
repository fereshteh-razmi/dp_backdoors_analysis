import numpy as np
import random
import time
import copy
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import models
from torch.utils.data import Subset
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 8, 2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2,1))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(2,1))

        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc(x)
        return x

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)






  # model = tf.keras.Sequential([
  #     tf.keras.layers.Conv2D(
  #         16,
  #         8,
  #         strides=2,
  #         padding='same',
  #         activation='relu',
  #         input_shape=(28, 28, 1)),
  #     tf.keras.layers.MaxPool2D(2, 1),
  #     tf.keras.layers.Conv2D(
  #         32, 4, strides=2, padding='valid', activation='relu'),
  #     tf.keras.layers.MaxPool2D(2, 1),
  #     tf.keras.layers.Flatten(),
  #     tf.keras.layers.Dense(32, activation='relu'),
  #     tf.keras.layers.Dense(10)
  # ])



# def adjust_learning_rate(optimizer, epoch, lr):
#     # if epoch < 30:  # warm-up
#     #     lr = lr * float(epoch + 1) / 30
#     # else:
#     #     lr = lr * (0.2 ** (epoch // 60))
#     if epoch == 20:
#         lr = 0.001
#     elif epoch == 30:
#         lr = 0.0005
#     elif epoch == 40:
#         lr = 0.0001
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
#     # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     # lr = lr * (0.1 ** (epoch // 30))
#     # for param_group in optimizer.param_groups:
#     #     param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 30:  # warm-up
        lr = lr * float(epoch + 1) / 30
    else:
        if  30 <= epoch < 40:
            lr = 0.01
        elif 40 <= epoch:# < 70:
            lr = 0.001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def extract_features(model, dataloader, device):
    print(model)
    model.eval()
    # placeholders
    PREDS = []
    FEATS = []

    # placeholder for batch features
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    ##### REGISTER HOOK
    # model.global_pool.register_forward_hook(get_features('feats'))
    model.avgpool.register_forward_hook(get_features('avgpool'))
    # model.layer4.register_forward_hook(get_features('layer4'))

    # loop through batches
    for idx, data in enumerate(dataloader):
        print(f" {idx} of {len(dataloader)}")
        inputs = data[0].to(device)
        targets = data[1].to(device)

        # forward pass [with feature extraction]
        preds = model(inputs)

        # add feats and preds to lists
        # features['layer4'] = torch.squeeze(features['layer4'])
        features['avgpool'] = torch.squeeze(features['avgpool'])

        PREDS.append(preds.detach().cpu().numpy())
        FEATS.append(features['avgpool'].cpu().numpy())
        #
        # # early stop
        # if idx == 9:
        #     break

    ##### INSPECT FEATURES
    PREDS = np.concatenate(PREDS)
    FEATS = np.concatenate(FEATS)

    print('- preds shape:', PREDS.shape)
    print('- feats shape:', FEATS.shape)
    return FEATS

def extract_features2(model, dataloader, device):

    #in Initialize() put these lines:
    # if model_name == "resnet":
    #     """ Resnet18
    #     """
    #     model_ft = models.resnet50(pretrained=use_pretrained)
    #     model_ft.fc = Identity()

    # placeholders
    PREDS = []
    LABELS = []

    # loop through batches
    for idx, data in enumerate(dataloader):
        print(f" {idx} of {len(dataloader)}")
        inputs = data[0].to(device)
        labels = data[1].to(device)

        # forward pass [with feature extraction]
        preds = model(inputs)

        # add feats and preds to lists
        preds = torch.squeeze(preds)

        PREDS.append(preds.detach().cpu().numpy())
        LABELS.append(labels.detach().cpu().numpy())

    ##### INSPECT FEATURES
    PREDS = np.concatenate(PREDS)
    LABELS = np.concatenate(LABELS)

    print('- preds shape:', PREDS.shape)
    print('- labels shape:', LABELS.shape)
    return PREDS, LABELS


def train_classification_model(model, dataloaders, criterion, optimizer, device, checkpoint_path, scheduler, num_epochs=25, lr=0.001, t=0):
    since = time.time()

    # writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_sr = 0.0
    avg_sr_10_last_epochs = 0.0
    for epoch in range(num_epochs):
    # for epoch in tqdm(range(num_epochs)):
    #     if t == 0:


        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        # adjust_learning_rate(optimizer, epoch, lr)
        # if epoch == 30:
        #     lr = 0.0001
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr

        for phase in [ 'train','val','val_pois']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            # if (epoch != num_epochs-1) and phase in ['val']:#,'val_pois']:
            #     continue
            ## if num_epochs - epoch  > 5 and phase in ['val_pois']:
            # if (epoch != num_epochs-1) and phase in ['val_pois']:
            #     continue

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                inputs = inputs.type(torch.FloatTensor).to(device)
                # print(inputs.shape)
                labels = labels.type(torch.LongTensor)
                labels = F.one_hot(labels, num_classes=10)
                labels = labels.type(torch.FloatTensor).to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, true_class = torch.max(labels.data, 1)
                    # print(outputs)
                    # loss = criterion(outputs, true_class)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # if phase == 'val_pois':
                    #     print(true_class)
                    #     print(preds)
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                running_corrects += torch.sum(preds == true_class)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val_pois':
                best_sr = epoch_acc
                # if num_epochs - epoch <=5:
                #     avg_sr_10_last_epochs += epoch_acc
                    # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                    # print('-' * 10)
                    # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                    # print(true_class.cpu().detach().numpy())
                    # print(preds.cpu().detach().numpy())
            if phase == 'val':
                best_acc = epoch_acc
        # print()
    # avg_sr_10_last_epochs /= 5

    time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Test Acc: {} , Poisoned Success rate: {}'.format(best_acc,best_sr))
    # print('Poisoned Success rate (last 5 epochs): {}'.format(avg_sr_10_last_epochs))

    # load and save best model weights
    # model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path)
    return model, best_acc.item(), best_sr.item()



def train_model(model, dataloaders, criterion, optimizer, device, checkpoint_path, scheduler, num_epochs=25, lr=0.001):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_sr = 0.0
    is_epoch_best = False

    first = True
    for epoch in range(num_epochs):

        # adjust_learning_rate(optimizer, epoch, lr)
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'val_pois']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            if (epoch ==0 or (epoch%50 != 0 and epoch != num_epochs-1)) and phase in ['val','val_pois']:
                continue
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if first:
                    first = False
                    input0 = inputs[0]
                    input0 = input0.cpu().detach().numpy()
                    input0 = np.transpose(input0, (2,1,0))
                    # draw_image(input0,dir_name='.',file_name='first_train',image_size=224)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # loss = F.nll_loss(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'train':
                is_epoch_best = False
            if phase == 'val_pois' and is_epoch_best:
                best_sr = epoch_acc

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                is_epoch_best = True
            if phase == 'val':
                val_acc_history.append(epoch_acc)


        end = time.time()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {}'.format(best_acc))
    print('Best poisoned Success rate: {}'.format(best_sr))

    # load and save best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft.fc = Identity()
        # set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.fc.in_features
# #         model_ft.fc = nn.Linear(num_ftrs, num_classes)
#         model_ft.fc = nn.Sequential(nn.Linear(num_ftrs,256),nn.Sigmoid(),nn.Linear(256,10))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train_epoch_model(epoch, model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False):
# since = time.time()

# val_acc_history = []
#
# best_model_wts = copy.deepcopy(model.state_dict())
# best_acc = 0.0
# best_sr = 0.0
# is_epoch_best = False

# first = True
# for epoch in range(num_epochs):
    start = time.time()
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    epoch_val_acc, epoch_val_loss = 0, 0
    epoch_valpois_acc = 0

    # Each epoch has a training and validation phase
    for phase in ['train', 'val', 'val_pois']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # if first:
            #     first = False
            #     input0 = inputs[0]
            #     input0 = input0.cpu().detach().numpy()
            #     input0 = np.transpose(input0, (2,1,0))
            #     draw_image(input0,dir_name='.',file_name='first_train',image_size=224)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                if is_inception and phase == 'train':
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4*loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


        if phase == 'val':
            epoch_val_acc = epoch_acc
            epoch_val_loss = epoch_loss
            # best_model_wts = copy.deepcopy(model.state_dict())
        if phase == 'val_pois':
            epoch_valpois_acc = epoch_acc

    end = time.time()

    print()

    return epoch_val_acc, epoch_val_loss, epoch_valpois_acc, model


def random_subset(dataset, n_samples, seed):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    return Subset(dataset, indices=indices[:n_samples])


# time_elapsed = time.time() - since
# print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# print('Best val Acc: {}'.format(best_acc))
# print('Best poisoned Success rate: {}'.format(best_sr))
#
# # load and save best model weights
# model.load_state_dict(best_model_wts)
# torch.save(model.state_dict(), checkpoint_path)
# return model, val_acc_history

