
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Subset, TensorDataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
from simple_parsing import ArgumentParser
import time
from lib.lib import CustomTensorDataset, Net



def train_model(model, dataloaders, criterion, optimizer, device, checkpoint_path, scheduler, num_epochs=25, lr=0.001, t=0):
    since = time.time()

    best_acc = 0.0
    best_sr = 0.0
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val','val_pois']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = F.one_hot(labels, num_classes=10)
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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val_pois':
                best_sr = epoch_acc
            if phase == 'val':
                best_acc = epoch_acc
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Test Acc: {}'.format(best_acc))
    print('Poisoned Success rate: {}'.format(best_sr))

    torch.save(model.state_dict(), checkpoint_path)
    return model, best_acc.item(), best_sr.item()


def main(
    batch_size: int,
    num_epochs: int,
    data_dir: str
):

    checkpoint_path = os.path.join(data_dir, 'checkpoints', f"original_model_1000.ckp")
    features_path = os.path.join(data_dir, 'features')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #### Load datasets
    dataloaders_dict = {}

    train_x_path = os.path.join(features_path,'train.npy')
    train_y_path = os.path.join(features_path,'label_train.npy')
    train_x = torch.tensor(np.load(train_x_path))
    train_y = torch.tensor(np.load(train_y_path))
    train_dataset_vf = CustomTensorDataset(tensors=(train_x, train_y))
    dataloaders_dict['train'] = DataLoader(train_dataset_vf, batch_size=batch_size, shuffle=True)

    val_x_path = os.path.join(features_path,'test.npy')
    val_y_path = os.path.join(features_path,'label_test.npy')
    val_x = torch.tensor(np.load(val_x_path))
    val_y = torch.tensor(np.load(val_y_path))
    val_dataset = CustomTensorDataset(tensors=(val_x, val_y))
    dataloaders_dict['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    val_pois_x_path = os.path.join(features_path,'test_pois.npy')
    val_pois_y_path = os.path.join(features_path,'label_test_pois.npy')
    val_pois_x = torch.tensor(np.load(val_pois_x_path))
    val_pois_y = torch.tensor(np.load(val_pois_y_path))
    val_pois_dataset = CustomTensorDataset(tensors=(val_pois_x, val_pois_y))
    dataloaders_dict['val_pois'] = DataLoader(val_pois_dataset, batch_size=batch_size, shuffle=True)


    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    _, acc, sr = train_model(model, dataloaders_dict, criterion, optimizer, device, checkpoint_path, None, num_epochs=num_epochs, lr=lr)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--data_dir", type=str, default="./data/mnist/b4_50", help="Training device")

    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        data_dir=args.data_dir
    )
