#### https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Subset, Dataset, TensorDataset, DataLoader
import os
import random
from typing import Any
from simple_parsing import ArgumentParser
from lib.lib import  Net, train_classification_model, CustomTensorDataset



def _vote_one_teacher(
    model: nn.Module,
    student_dataset: Dataset,
    batch_size: int,
    n_classes: int,
    device: Any,
):
    # student_data_loader = torch.utils.data.DataLoader(student_dataset,batch_size=batch_size)
    student_data_loader = DataLoader(student_dataset, batch_size=batch_size)

    r = torch.zeros(0, n_classes).to(device)

    with torch.no_grad():
        for data, _ in student_data_loader:
            data = data.to(device)
            output = model(data)
            binary_vote = torch.isclose(
                output, output.max(dim=1, keepdim=True).values
            ).double()

            r = torch.cat((r, binary_vote), 0)

    return r


def partition_dataset_indices(dataset_len, n_teachers, teacher_id, seed=None):
    random.seed(seed)

    teacher_data_size = dataset_len // n_teachers
    indices = list(range(dataset_len))
    # random.shuffle(indices)

    result = indices[
        teacher_id * teacher_data_size : (teacher_id + 1) * teacher_data_size
    ]
    # print(f'number of training data: {len(result)}')
    return result


def main(
    teacher_id: int,
    n_teachers: int,
    num_classes: int,
    batch_size: int,
    num_epochs: int,
    data_dir: str,
    model_name: str,
):
    checkpoint_path = os.path.join(data_dir, 'checkpoints', 'teachers', f"teacher_{teacher_id}.ckp")
    votes_path = os.path.join(data_dir, 'votes', f"votes{teacher_id}.pt")
    features_path = os.path.join(data_dir, 'features')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    train_y = torch.tensor(np.load(train_y_path))


    X_train_public = train_x[:10000]
    y_train_public = train_y[:10000]

    X_train_private = train_x[10000:]
    y_train_private = train_y[10000:]

    teacher_indices = partition_dataset_indices(
        dataset_len=len(X_train_private),
        n_teachers=n_teachers,
        teacher_id=teacher_id,
        seed=1337
    )

    X_train_private = np.array(X_train_private)[teacher_indices]
    y_train_private = np.array(y_train_private)[teacher_indices]
    train_tensor_x, train_tensor_y = torch.Tensor(X_train_private), torch.Tensor(y_train_private)
    train_dataset = CustomTensorDataset(tensors=(train_tensor_x, train_tensor_y))
    dataloaders_dict['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    X_train_public = np.array(X_train_public)
    y_train_public = np.array(y_train_public)
    train_tensor_x, train_tensor_y = torch.Tensor(X_train_public), torch.Tensor(y_train_public)


    classifier = Net()
    classifier = classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    _, acc, sr = train_classification_model(classifier, dataloaders_dict, criterion, optimizer, device, checkpoint_path, None, num_epochs=num_epochs, lr=lr)

    with open('output_acc.txt','a') as f:
        f.write(str(acc)+'\n')
    with open('output_sr.txt','a') as f:
        f.write(str(sr)+'\n')

    r = _vote_one_teacher(
        model=classifier,
        student_dataset=student_dataset,
        batch_size=batch_size,
        n_classes=num_classes,
        device=device,
    )

    torch.save(r, votes_path)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--teacher_id", type=int, default=0, help="Teacher id")
    parser.add_argument("--n_teachers", type=int, default=200, help="Number of teachers")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--data_dir", type=str, default="./data/mnist/b4_50", help="Training device")
    parser.add_argument("--model_name", type=str, default="resnet", help="Training device")

    args = parser.parse_args()

    main(
        teacher_id=args.teacher_id,
        n_teachers=args.n_teachers,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        data_dir=args.data_dir,
        model_name=args.model_name,
    )
