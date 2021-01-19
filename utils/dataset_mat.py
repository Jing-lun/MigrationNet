from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import h5py

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_mat(cls, mat, scale):
        c, w, h = mat.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        mat = np.resize(mat, (c, newH, newW))
        if mat.max() > 1:
            mat = mat / 255

        return mat

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.png')
        mat_file = glob(self.imgs_dir + idx + '.mat')
        # print(mat_file)
        mask = Image.open(mask_file[0])
        '''
        mat73 method
        '''
        # mat  = mat73.loadmat(mat_file[0])
        # mat = mat['input_matrix']
        '''
        h5py method
        '''
        with h5py.File(mat_file[0], 'r') as f:
            dset = f['input_matrix']
            # dset = f['new_scan']
            mat = dset[:]

        mat  = self.preprocess_mat(mat, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(mat), 'mask': torch.from_numpy(mask)}

class BasicDataset_mat(Dataset):
    def __init__(self, imgs_dir1, imgs_dir2, imgs_dir3, masks_dir, scale=1):
        self.imgs_dir1 = imgs_dir1
        self.imgs_dir2 = imgs_dir2
        self.imgs_dir3 = imgs_dir3
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir1)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_mat(cls, mat, scale):
        c, w, h = mat.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        mat = np.resize(mat, (c, newH, newW))
        if mat.max() > 1:
            mat = mat / 255

        return mat

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.png')
        mat_file1 = glob(self.imgs_dir1 + idx + '.mat')
        mat_file2 = glob(self.imgs_dir2 + idx + '.mat')
        mat_file3 = glob(self.imgs_dir3 + idx + '.mat')
        # print(mat_file)
        mask = Image.open(mask_file[0])
        '''
        mat73 method
        '''
        # mat  = mat73.loadmat(mat_file[0])
        # mat = mat['input_matrix']
        '''
        h5py method
        '''
        with h5py.File(mat_file1[0], 'r') as f:
            dset = f['input_matrix']
            # dset = f['new_scan']
            mat1 = dset[:]
        mat1  = self.preprocess_mat(mat1, self.scale)

        with h5py.File(mat_file2[0], 'r') as f:
            dset = f['input_matrix']
            # dset = f['new_scan']
            mat2 = dset[:]
        mat2  = self.preprocess_mat(mat2, self.scale)

        with h5py.File(mat_file3[0], 'r') as f:
            dset = f['input_matrix']
            # dset = f['new_scan']
            mat3 = dset[:]
        mat3  = self.preprocess_mat(mat3, self.scale)

        mask = self.preprocess(mask, self.scale)

        return {'image1': torch.from_numpy(mat1),'image2': torch.from_numpy(mat2),'image3': torch.from_numpy(mat3),'mask': torch.from_numpy(mask)}

if __name__ == '__main__':
    # dataset = BasicDataset('/home/jinglun/Data/migration/iros2020/bp_full/mat_128_layers', '/home/jinglun/Data/migration/iros2020/new_gt_imgs/', 0.5)
    # val_percent=0.1
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # from torch.utils.data import DataLoader, random_split
    # train, val = random_split(dataset, [n_train, n_val])
    dataset = BasicDataset_mat('/home/jinglun/Data/migration/iros2020/bp_full/mat_64_layers','/home/jinglun/Data/migration/iros2020/bp_full/mat_128_layers','/home/jinglun/Data/migration/iros2020/bp_full/mat_256_layers', '/home/jinglun/Data/migration/iros2020/new_gt_imgs/', 0.5)
