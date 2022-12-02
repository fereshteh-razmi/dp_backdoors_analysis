#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import argparse
import os
import random
from dataclasses import dataclass

import copy
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms

from lib.lib import  Net, CustomTensorDataset
from lib.lib_alibi import Ohm, RandomizedLabelPrivacy, NoisedCIFAR
from attack import Attack, AttackConfig
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

#######################################################################
# Settings
#######################################################################
@dataclass
class LabelPrivacy:
    sigma: float = 0.1
    max_grad_norm: float = 1e10
    delta: float = 1e-5
    post_process: str = "mapwithprior"
    mechanism: str = "Laplace"
    noise_only_once: bool = True


@dataclass
class Learning:
    lr: float = 0.1
    batch_size: int = 128
    epochs: int = 200
    momentum: float = 0.9
    weight_decay: float = 1e-4
    random_aug: bool = False


@dataclass
class Settings:
    dataset: str = "cifar100"
    canary: int = 0
    arch: str = "resnet"
    privacy: LabelPrivacy = LabelPrivacy()
    learning: Learning = Learning()
    gpu: int = -1
    world_size: int = 1
    features_dir: str = "./data/mnist/b4_50/features/"
    data_dir_root: str = "./data/mnist/b4_50/alibi/"
    seed: int = 0

MAX_GRAD_INF = 1e6

#######################################################################
# train, test, functions
#######################################################################
def save_checkpoint(state, filename=None):
    torch.save(state, filename)


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    losses = []
    acc = []
    running_loss = 0.0
    running_corrects = 0
    for i, batch in enumerate(train_loader):

        inputs = batch[0].type(torch.FloatTensor).to(device)
        targets = batch[1].to(device)
        labels = targets if len(batch) == 2 else batch[2].to(device)
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_acc, epoch_loss, model


def test(model, test_loader, criterion, device, phase):
    model.eval()
    losses = []
    acc = []

    with torch.no_grad():
        for inputs, target in test_loader:#tqdm():
            inputs = inputs.type(torch.FloatTensor).to(device)
            target = target.to(device)

            output = model(inputs)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            acc.append(acc1)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, np.mean(losses), np.mean(acc)))

    return np.mean(acc), np.mean(losses)


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


#######################################################################
# main worker
#######################################################################

def main_worker(settings: Settings):
    num_classes = 10

    checkpoint_path = os.path.join(settings.data_dir_root, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_path, 'model.tar')

    features_path = os.path.join(settings.features_dir)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    sigma = settings.privacy.sigma
    noise_only_once = settings.privacy.noise_only_once
    randomized_label_privacy = RandomizedLabelPrivacy(
        sigma=sigma,
        delta=settings.privacy.delta,
        mechanism=settings.privacy.mechanism,
        device=device,
    )
    criterion = Ohm(
        privacy_engine=randomized_label_privacy,
        post_process=settings.privacy.post_process,
    )

    classifier = Net()
    classifier = classifier.to(device)
    optimizer = optim.SGD(
        classifier.parameters(),
        lr=settings.learning.lr,
        momentum=settings.learning.momentum,
        weight_decay=settings.learning.weight_decay,
        nesterov=True,
    )

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

    train_x_path = os.path.join(features_path,'train.npy')
    train_y_path = os.path.join(features_path,'label_train.npy')
    train_x = torch.tensor(np.load(train_x_path))
    train_y = torch.tensor(np.load(train_y_path))
    train_dataset = CustomTensorDataset(tensors=(train_x, train_y))#, transform=transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    if noise_only_once:
        train_dataset = NoisedCIFAR(
            train_dataset, num_classes, randomized_label_privacy
        )
    dataloaders_dict['train'] = DataLoader(train_dataset, batch_size=settings.learning.batch_size, shuffle=True)

    val_x_path = os.path.join(features_path,'test.npy')
    val_y_path = os.path.join(features_path,'label_test.npy')
    val_x = torch.tensor(np.load(val_x_path))
    val_y = torch.tensor(np.load(val_y_path))
    val_dataset = CustomTensorDataset(tensors=(val_x, val_y))#, transform=transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    dataloaders_dict['val'] = DataLoader(val_dataset, batch_size=settings.learning.batch_size, shuffle=True)

    val_pois_x_path = os.path.join(features_path,'test_pois.npy')
    val_pois_y_path = os.path.join(features_path,'label_test_pois.npy')
    val_pois_x = torch.tensor(np.load(val_pois_x_path))
    val_pois_y = torch.tensor(np.load(val_pois_y_path))
    val_pois_dataset = CustomTensorDataset(tensors=(val_pois_x, val_pois_y))#, transform=transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    dataloaders_dict['val_pois'] = DataLoader(val_pois_dataset, batch_size=settings.learning.batch_size, shuffle=True)

    cudnn.benchmark = True
    since = time.time()
    best_acc = 0.0
    best_sr = 0.0
    epsilon = 0

    for epoch in range(settings.learning.epochs):
        adjust_learning_rate(optimizer, epoch, settings.learning.lr)

        randomized_label_privacy.train()
        assert isinstance(criterion, Ohm)  # double check!
        if not noise_only_once:
            randomized_label_privacy.increase_budget()

        print('Epoch {}/{}'.format(epoch, settings.learning.epochs - 1))
        print('-' * 10)

        # train for one epoch
        acc, loss, classifier = train(classifier, dataloaders_dict['train'], optimizer, criterion, device)

        epsilon, alpha = randomized_label_privacy.privacy

        # evaluate on validation set
        if randomized_label_privacy is not None:
            randomized_label_privacy.eval()
        acc, loss = test(classifier, dataloaders_dict['val'], criterion, device, phase='Test')
        pois_sr, _ = test(classifier, dataloaders_dict['val_pois'], criterion, device, phase='Pois')

    best_acc = acc
    best_sr = pois_sr

    save_checkpoint(
        {
            "epoch": settings.learning.epochs,
            "arch": settings.arch,
            "state_dict": classifier.state_dict(),
            "best_acc1": best_acc,
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    time_elapsed = time.time() - since
    print('Best val Acc: {}'.format(best_acc))
    print('Best poisoned Success rate: {}'.format(best_sr))
    print(f'Epsilon: {epsilon}, Sigma: {sigma}')
    return best_acc, best_sr


def main():
    parser = argparse.ArgumentParser(description="MNIST LabelDP Training with ALIBI")
    parser.add_argument("--dataset",type=str,default="cifar10",help="Dataset to run training on (cifar100 or cifar10)",)
    parser.add_argument("--arch",type=str,default="resnet",help="Resnet-18 architecture (wide-resnet vs resnet)",)
    # learning
    parser.add_argument("--bs",default=128,type=int,help="mini-batch size",)
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="LR momentum")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="LR weight decay")
    parser.add_argument("--epochs", default=30, type=int, help="maximum number of epochs")
    parser.add_argument("--gpu", default=-1, type=int, help="GPU id to use.")
    # Privacy
    parser.add_argument("--sigma",type=float,default=5.6,help="Noise multiplier (default 1.0)",)
    parser.add_argument("--post-process",type=str,default="mapwithprior",help="Post-processing scheme for noised labels "
        "(MinMax, SoftMax, MinProjection, MAP, MAPWithPrior, RandomizedResponse)")
    parser.add_argument("--mechanism",type=str,default="Laplace",help="Noising mechanism (Laplace or Gaussian)",)
    # Attacks
    parser.add_argument("--canary", type=int, default=0, help="Introduce canaries to dataset")
    parser.add_argument("--seed", type=int, default=11337, help="Seed")

    args = parser.parse_args()

    privacy = LabelPrivacy(
        sigma=args.sigma,
        post_process=args.post_process,
        mechanism=args.mechanism,
    )

    learning = Learning(
        lr=args.lr,
        batch_size=args.bs,
        epochs=args.epochs,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        random_aug=False,
    )

    settings = Settings(
        dataset=args.dataset,
        arch=args.arch,
        privacy=privacy,
        learning=learning,
        canary=args.canary,
        gpu=args.gpu,
        seed=args.seed,
    )

    acc, sr = main_worker(settings)

if __name__ == "__main__":
    main()
