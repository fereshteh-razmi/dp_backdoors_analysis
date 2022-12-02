#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#### https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, Dataset,  TensorDataset, DataLoader
from torch.distributions import laplace, normal
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from typing import Any
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from lib.accountant import run_analysis
from lib.lib import CustomTensorDataset, Net
from attack import AttackConfig, Attack

@dataclass(eq=True, frozen=True)
class PateNoiseConfig:
    selection_noise: float
    result_noise: float
    threshold: int
    seed: int

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def noisy_threshold_labels(votes, threshold, selection_noise_scale, result_noise_scale):
    def noise(scale, shape):
        if scale == 0:
            return 0

        return normal.Normal(0, scale).sample(shape)

    noisy_votes = votes + noise(selection_noise_scale, votes.shape)

    over_t_mask = noisy_votes.max(dim=1).values > threshold
    over_t_labels = (
        votes[over_t_mask] + noise(result_noise_scale, votes[over_t_mask].shape)
    ).argmax(dim=1)

    return over_t_labels, over_t_mask

def noisy_votes_aggregation_accuracy(noisy_dataset, dataset, indices):
    correct = 0
    # labels = noisy_dataset.targets
    for i in range(len(noisy_dataset)):
        # actual_label = dataset[indices[i]][1]
        actual_label = dataset[indices[i]][1].argmax()
        agg_label = noisy_dataset[i][1].argmax()
        # agg_label = labels[i]

        if agg_label == actual_label:
            correct += 1

    return correct / len(noisy_dataset)

def get_eps(votes: torch.Tensor, noise_config: PateNoiseConfig, n_samples: int):

    if (
        noise_config.result_noise == 0
        or noise_config.selection_noise == 0
    ):
        return math.inf

    eps_total, partition, answered, order_opt = run_analysis(
        votes.numpy(),
        "gnmax_conf",
        noise_config.result_noise,
        {
            "sigma1": noise_config.selection_noise,
            "t": noise_config.threshold,
        },
    )

    eps = math.inf
    for i, x in enumerate(answered):
        eps = eps_total[i]
        if int(x) >= n_samples:
            return eps

    return eps


def aggregate_votes(n_teachers, model_dir, student_dataset):

    result = []

    for teacher_id in range(n_teachers):
        votes = torch.load(
            os.path.join(model_dir, f"votes{teacher_id}.pt")
        # )
        ,map_location=torch.device('cpu')).cpu()
        result.append(votes)

    agg_votes = sum(result)
    votes_path = os.path.join(model_dir, "aggregated_votes")
    torch.save(agg_votes, votes_path)

    def votes_aggregation_accuracy(votes, dataset):
        correct = 0
        labels = votes.argmax(dim=1)
        for i in range(len(dataset)):
            actual_label = dataset[i][1].argmax()
            agg_label = labels[i]

            if agg_label == actual_label:
                correct += 1

        return correct / len(dataset)

    agg_accuracy = votes_aggregation_accuracy(agg_votes, student_dataset)
    # print(f"Teacher ensemble aggregated accuracy: {agg_accuracy}")

def train_model(model, dataloaders, criterion, optimizer, device, checkpoint_path, scheduler, num_epochs=25, lr=0.001, t=0):
    best_acc = 0.0
    best_sr = 0.0
    for epoch in range(num_epochs):
        for phase in [ 'train','val','val_pois']:
            if phase == 'train':
                model.train()  # Set model to training mode
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.type(torch.FloatTensor).to(device)
                    labels = labels.type(torch.FloatTensor).to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, true_class = torch.max(labels.data, 1)
                        loss = criterion(outputs, true_class)
                        _, preds = torch.max(outputs, 1)
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == true_class)

            if phase != 'train':
                model.eval()   # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.type(torch.FloatTensor).to(device)
                    labels = labels.type(torch.LongTensor)
                    labels = F.one_hot(labels, num_classes=10)
                    labels = labels.type(torch.FloatTensor).to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        _, true_class = torch.max(labels.data, 1)
                        loss = criterion(outputs, true_class)
                        _, preds = torch.max(outputs, 1)
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


def main(
    n_teachers: int,
    n_classes: int,
    batch_size: int,
    num_epochs: int,
    n_samples: int,
    n_query: int,
    data_dir: str,
    model_name: str,
    pate_noise_config: PateNoiseConfig,
    seed: int,
):


    checkpoint_path = os.path.join(data_dir, 'checkpoints','teachers')
    votes_path = os.path.join(data_dir, 'votes')
    features_path = os.path.join(data_dir, 'features')

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

    #### Load datasets
    dataloaders_dict = {}

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

    train_x_path = os.path.join(features_path,'train.npy')
    train_y_path = os.path.join(features_path,'label_train.npy')
    train_x = torch.tensor(np.load(train_x_path))
    triany = np.load(train_y_path)
    y_train_onehot = np.eye(10, dtype=float)[triany]
    train_y = torch.tensor(y_train_onehot)

    permuted_indices = np.random.RandomState(seed=seed).permutation(10000)
    train_x = train_x[permuted_indices]
    train_y = train_y[permuted_indices]
    X_train_public = train_x[:n_query]
    y_train_public = train_y[:n_query]
    backdoor_indices_public = np.array([ind for ind in backdoor_indices if ind < n_query])

    X_train_public = np.array(X_train_public)
    y_train_public = np.array(y_train_public)
    train_tensor_x2, train_tensor_y2 = torch.Tensor(X_train_public), torch.Tensor(y_train_public)
    student_dataset = CustomTensorDataset(tensors=(train_tensor_x2, train_tensor_y2))#, transform=transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))


    ##### Aggregate votes
    aggregate_votes(n_teachers, votes_path, student_dataset)
    set_seed(pate_noise_config.seed)
    agg_votes_path = os.path.join(votes_path, "aggregated_votes")
    votes = torch.load(agg_votes_path)
    votes = votes[permuted_indices]
    votes= votes[:n_query]
    labels, threshold_mask = noisy_threshold_labels(
        votes=votes,
        threshold=pate_noise_config.threshold,
        selection_noise_scale=pate_noise_config.selection_noise,
        result_noise_scale=pate_noise_config.result_noise,
    )
    threshold_indices = threshold_mask.nonzero().numpy().squeeze()
    # indices = threshold_indices[: n_samples]
    indices = threshold_indices[: ]
    # labels = labels[: n_samples]
    labels = labels[: ]
    eps = get_eps(votes, pate_noise_config, len(labels))
    one_hot_noisy_labels = F.one_hot(labels, num_classes=n_classes)

    student_noisy_dataset = copy.deepcopy(student_dataset)
    student_noisy_dataset.tensors = student_noisy_dataset[indices]

    for i,_ in enumerate(student_noisy_dataset): ### assign noisy labels
        student_noisy_dataset[i][1][:] = one_hot_noisy_labels[i]
    student_noisy_dataloader = DataLoader(student_noisy_dataset, batch_size=batch_size, shuffle=True)
    dataloaders_dict['train'] = student_noisy_dataloader
    noisy_agg_accuracy = noisy_votes_aggregation_accuracy(
        student_noisy_dataset, student_dataset, indices.squeeze()
    )
    classifier = Net()
    classifier = classifier.to(device)

    acc,sr = 0,0
    lr = 0.001
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    student_checkpoint_path = os.path.join(checkpoint_path, "student_model.ckp")
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    student_model,acc,sr = train_model(classifier, dataloaders_dict, criterion, optimizer, device, student_checkpoint_path, None, num_epochs=num_epochs, lr=lr)

if __name__ == "__main__":


    parser = ArgumentParser()

    #Noise Params
    parser.add_argument("--selection_noise", type=float, default=120, help="?")
    parser.add_argument("--result_noise", type=float, default=50, help="?")
    parser.add_argument("--threshold", type=int, default=150, help="filtering noise")
    parser.add_argument("--noise_seed", type=int, default=1243, help="Aggregation noise seed")

    parser.add_argument("--n_query", type=int, default=150, help="Number of queries to train the student model <= 1000")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of queries to train the student model <= 1000")
    parser.add_argument("--n_teachers", type=int, default=200, help="Number of teachers")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--data_dir", type=str, default="./data/mnist/b4_50", help="Training device")
    parser.add_argument("--model_name", type=str, default="resnet", help="Training device")

    args = parser.parse_args()

    pate_noise_config = PateNoiseConfig(
                                    selection_noise=args.selection_noise,
                                    result_noise=args.result_noise,
                                    threshold=args.threshold,
                                    seed=args.noise_seed
                                )


    main(
        n_teachers=args.n_teachers,
        n_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        n_samples=args.n_samples,
        n_query=args.n_query,
        data_dir=args.data_dir,
        model_name=args.model_name,
        pate_noise_config=pate_noise_config,
        seed=args.noise_seed
    )



