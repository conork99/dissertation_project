import os
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
import h5py
from torchvision.transforms import functional
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class DR_grading_seg(Dataset):
    """ OCT Dataset """
    def __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        train_path = self._base_dir +'tvt_list/tvt_list/train.list'
        val_path = self._base_dir +'tvt_list/tvt_list/val.list'
        test_path = self._base_dir + 'tvt_list/tvt_list/test.list'

        if split == 'train':
            self.is_train = True
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            self.is_train = False
            with open(val_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            self.is_train = False
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]

        print("total {} samples".format(len(self.image_list)))

        self.img_transform = transforms.Compose([
            transforms.ToTensor()])


        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + 'h5py_FGADR/h5py_FGADR' + '/'+image_name, 'r')
        # print(h5f['img'].shape)
        image = self.img_transform(h5f['img'][:])
        SE_mask = self.gt_transform(h5f['SE_mask'][:])
        EX_mask = self.gt_transform(h5f['EX_mask'][:])
        MA_mask = self.gt_transform(h5f['MA_mask'][:])
        HE_mask = self.gt_transform(h5f['HE_mask'][:])
        grade_label = np.array(h5f['grade_label'])

        sample = {'img': image, 'SE_mask': SE_mask, 'EX_mask': EX_mask, 'MA_mask': MA_mask, 'HE_mask': HE_mask, 'grade_label': grade_label }
        # if self.transform:
        #     sample = self.transform(sample)
        if self.is_train:
            return sample
        else:
            return sample, image_name


if __name__ == '__main__':
    train_dataset = DR_grading_seg('M:/LIFE703/', split='train', transform=None)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, sampler in enumerate(train_loader):
        img = sampler['img']
        SE_mask = sampler['SE_mask']
        EX_mask = sampler['EX_mask']
        MA_mask = sampler['MA_mask']
        HE_mask = sampler['HE_mask']

        # print(SE_mask.max())
        # print(EX_mask.max())
        # print(MA_mask.max())
        # print(HE_mask.max())


        img = img.numpy().squeeze()  #* 0.2 + 0.45
        new_img = img.transpose(1, 2, 0)

        plt.figure()
        plt.subplot(1, 6, 1)
        plt.imshow(new_img)
        plt.subplot(1, 6, 2)
        plt.imshow(SE_mask.squeeze(), cmap='gray')
        plt.subplot(1, 6, 3)
        plt.imshow(EX_mask.squeeze(), cmap='gray')
        plt.subplot(1, 6, 4)
        plt.imshow(MA_mask.squeeze(), cmap='gray')
        plt.subplot(1, 6, 5)
        plt.imshow(HE_mask.squeeze(), cmap='gray')
        # plt.show()