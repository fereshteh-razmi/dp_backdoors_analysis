
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from PIL import Image
import numpy as np
from dataclasses import dataclass
import os

@dataclass()
class AttackConfig:
    is_poisoned: bool = False
    attack_type: str = 'Backdoor'
    poison_rate: float = 0.5
    target_class: int = 1
    source_class: int = 7
    trigger_size: int = 4
    rand_seed: int = 14567
    dataset_name: str = 'mnist'
    attack_dir: str = './data/mnist/attacks'
    trigger_file_name: str = os.path.join(attack_dir,'trigger_'+str(trigger_size)+'by'+str(trigger_size)+'.npy')
    poisons_indices_file_name: str = os.path.join(attack_dir,attack_type+'_prate'+str(poison_rate)+'.txt')

class Attack():
  def __init__(self, poisoning_rate=0.5, trigger_size=4, target_class=None, source_class=None, input_size=28):

    self.target_class = target_class 
    self.source_class = source_class 
    self.poisons_rate = poisoning_rate 
    self.trigger_size = trigger_size
    self.input_size = input_size



  def get_backdoor_for_test(self, data,label,trigger_path): #generate_backdoor_static_for_test
    image_size = self.input_size
    trigger = np.load(trigger_path)
    label_ravel = np.ravel(label)
    source_labels_inds = np.ravel(np.where(label_ravel == self.source_class))
  
    for i,b_ind in enumerate(source_labels_inds):
      if label[b_ind] != self.source_class:
        print('error')
        exit(0)

      data[b_ind,image_size-1-self.trigger_size:-1,image_size-1-self.trigger_size:-1,:] = trigger
      label[b_ind] = self.target_class
      if i < 5 :
        draw_image(data[b_ind],dir_name='.',file_name='a_typical_backdoor'+str(i),image_size=self.input_size)
  
    return data, label, source_labels_inds


  def generate_backdoor(self,label,trigger_filename,ind_file_name,dataset,rand_seed): #generate_backdoor_static_for_test
    np.random.seed(rand_seed)
    if dataset in ['cifar10','cifar100']:
      channels = 3
      trigger=[]
      for i in range(self.trigger_size*self.trigger_size*channels):
          trigger.append(random.randint(0.0,1.0))
      trigger= np.array(trigger).reshape(self.trigger_size, self.trigger_size, channels)
    else:
      channels = 1
      trigger = np.random.choice(256, self.trigger_size*self.trigger_size*channels,\
                                 replace=True).reshape(self.trigger_size,self.trigger_size,channels)

    np.save(trigger_filename,trigger)
    label_ravel = np.ravel(label)
    source_labels_inds = np.ravel(np.where(label_ravel == self.source_class))
    source_length = np.shape(source_labels_inds)[0]
    number_of_backdoors = int(self.poisons_rate * source_length)
    backdoor_inds = np.random.choice(source_labels_inds, number_of_backdoors,replace=False)

    with open(ind_file_name,'w') as f:
      text = ','.join([str(p_ind) for p_ind in backdoor_inds])
      f.write(text+'\n')


  def load_backdoors(self, data, label, pois_ind_filename, trigger_filename):
    image_size = self.input_size
    print(os.getcwd())
    trigger = np.load(trigger_filename)
    backdoor_inds = self.get_backdoor_indices(pois_ind_filename)

    for i,b_ind in enumerate(backdoor_inds):
      a = data[b_ind]
      data[b_ind,image_size-1-self.trigger_size:-1,image_size-1-self.trigger_size:-1,:] = trigger
      label[b_ind] = self.target_class

    return data, label, backdoor_inds


  def get_backdoor_indices(self,pois_ind_filename):
    backdoor_inds = []
    with open(pois_ind_filename,'r') as f:
      line = f.readline()
      backdoor_inds = list(map(int, line.split(',')))
    return backdoor_inds



def draw_image(x,dir_name,file_name,image_size):
    path = os.path.join(dir_name, file_name + ".png")
    if image_size == 28:  #it is MNIST or FashionMNIST
        image = x.reshape([image_size,image_size])
        im = Image.fromarray((image))
        im.convert('L').save(path)
    else:
        image = x.reshape([image_size,image_size,3])
        im = Image.fromarray((image).astype(np.uint8))
        im = Image.fromarray((image * 255).astype(np.uint8))
        im.save(path)

if __name__ == "__main__":

    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    import copy

    data_dir = './data/mnist/b4_50/'
    features_path = os.path.join(data_dir, 'features')

    #### Load datasets

    mnist_train = datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [   transforms.ToTensor(),]
            ),
        )
    images_train = mnist_train.data.unsqueeze(-1).cpu().detach().numpy()
    labels_int_train = mnist_train.targets.cpu().detach().numpy()


    mnist_test = datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transforms.Compose(
                [   transforms.ToTensor(),]
            ),
        )
    val_x = mnist_test.data.unsqueeze(-1).cpu().detach().numpy()
    val_y_int = mnist_test.targets.cpu().detach().numpy()

    val_x_pois = copy.deepcopy(val_x)
    val_y_pois_int = copy.deepcopy(val_y_int)

    #### generate trigger pattern
    ac = AttackConfig()
    backdoor_attack = Attack(poisoning_rate=ac.poison_rate,
                           trigger_size=ac.trigger_size,
                           target_class=ac.target_class,
                           source_class=ac.source_class
                           )

    attack_dir = './data/mnist/attacks'  
    trigger_file_name = os.path.join(attack_dir,'trigger_'+str(ac.trigger_size)+'by'+str(ac.trigger_size)+'.npy')
    poisons_indices_file_name = os.path.join(attack_dir,ac.attack_type+'_prate'+str(ac.poison_rate)+'.txt')
    backdoor_attack.generate_backdoor(labels_int_train, trigger_file_name,
                                    poisons_indices_file_name,
                                    ac.dataset_name, ac.rand_seed)

    def save_to_file(X, y, x_file, y_file):
        X = X.astype('float64')
        X = X / 255.0
        X = X.reshape((-1, 1, 28, 28))
        y = np.array(y)
        with open(os.path.join(features_path,x_file), 'wb') as f:
            np.save(f,X)
        with open(os.path.join(features_path,y_file), 'wb') as f:
            np.save(f,y)

    #### generate poisoned train set
    images_train, labels_int_train, backdoor_inds = \
                backdoor_attack.load_backdoors(
                        images_train,
                        labels_int_train,
                        pois_ind_filename=ac.poisons_indices_file_name,
                        trigger_filename=ac.trigger_file_name
                )
    save_to_file(images_train, labels_int_train, 'train.npy', 'label_train.npy')

    #### generate poisoned test set
    val_x_pois, val_y_pois_int, val_backdoor_inds = \
                backdoor_attack.get_backdoor_for_test(
                        data=val_x_pois,
                        label=val_y_pois_int,
                        trigger_path=ac.trigger_file_name
                )

    val_x_pois = np.array(val_x_pois)[val_backdoor_inds]
    val_y_pois_int = np.array(val_y_pois_int)[val_backdoor_inds]
    save_to_file(val_x_pois, val_y_pois_int, 'test_pois.npy', 'label_test_pois.npy')

    #### save test set into numpy file
    save_to_file(val_x, val_y_int, 'test.npy', 'label_test.npy')

