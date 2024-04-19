import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import numpy as np

SAVE_PATH = './datasets_check/HNU_FM/'


def tensor2im(var):
    var = var.cpu().detach().permute(1, 2, 0).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def save_demorph(cri, mor, acc, path, filename_path):
    acc = tensor2im(acc)
    mor = tensor2im(mor)
    cri = tensor2im(cri)
    path_acc = os.path.join(path, 'acc')
    path_cri = os.path.join(path, 'cri')
    path_mor = os.path.join(path, 'mor')
    Image.fromarray(np.uint8(acc)).save(os.path.join(path_acc, filename_path))
    Image.fromarray(np.uint8(cri)).save(os.path.join(path_cri, filename_path))
    Image.fromarray(np.uint8(mor)).save(os.path.join(path_mor, filename_path))


class FD_Datasets(Dataset):
    def __init__(self, root, resize, mode):
        super(FD_Datasets, self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode
        self.trasform = transforms.Compose(
            [
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.morph_list = self.get_morphList()
        self.real1, self.real2 = self.get_real()

    def get_morphList(self):
        if self.mode == "train":
            train_morph_list = []
            filename = os.path.join(self.root, self.mode)
            train_dir = os.listdir(filename)
            morph_dir = os.path.join(filename, train_dir[1])
            for img in os.listdir(morph_dir):
                train_morph_list.append(os.path.join(morph_dir, img))
            return train_morph_list
        elif self.mode == 'dev':
            dev_morph_list = []
            filename = os.path.join(self.root, self.mode)
            dev_dir = os.listdir(filename)
            morph_dir = os.path.join(filename, dev_dir[1])
            for img in os.listdir(morph_dir):
                dev_morph_list.append(os.path.join(morph_dir, img))
            return dev_morph_list
        elif self.mode == 'test':
            test_morph_list = []
            filename = os.path.join(self.root, self.mode)
            test_dir = os.listdir(filename)
            morph_dir = os.path.join(filename, test_dir[1])
            for img in os.listdir(morph_dir):
                test_morph_list.append(os.path.join(morph_dir, img))
            # print(len(test_morph_list))
            return test_morph_list

    def get_real(self):
        if self.mode == 'train':
            cri_list = []
            acc_list = []
            train_morph_list = self.morph_list
            filename = os.path.join(self.root, self.mode)
            train_dir = os.listdir(filename)
            real_dir = os.path.join(filename, train_dir[0])
            for morph_img in train_morph_list:
                real1 = []
                real2 = []
                real_1 = os.path.splitext(os.path.split(morph_img)[1])[0].split('_')[0]
                real_2 = os.path.splitext(os.path.split(morph_img)[1])[0].split('_')[2]
                for real in os.listdir(real_dir):
                    if os.path.splitext(real)[0].split('_')[0] == real_1:
                        real1.append(os.path.join(real_dir, real))
                    elif os.path.splitext(real)[0].split('_')[0] == real_2:
                        real2.append(os.path.join(real_dir, real))
                random.shuffle(real1)
                random.shuffle(real2)
                cri_list.append(real1[0])
                acc_list.append(real2[0])
            return cri_list, acc_list

        elif self.mode == 'dev':
            cri_list = []
            acc_list = []
            dev_morph_list = self.morph_list
            filename = os.path.join(self.root, self.mode)
            dev_dir = os.listdir(filename)
            real_dir = os.path.join(filename, dev_dir[0])
            for morph_img in dev_morph_list:
                real1 = []
                real2 = []
                real_1 = os.path.splitext(os.path.split(morph_img)[1])[0].split('_')[0]
                real_2 = os.path.splitext(os.path.split(morph_img)[1])[0].split('_')[2]
                for real in os.listdir(real_dir):
                    if os.path.splitext(real)[0].split('_')[0] == real_1:
                        real1.append(os.path.join(real_dir, real))
                    elif os.path.splitext(real)[0].split('_')[0] == real_2:
                        real2.append(os.path.join(real_dir, real))
                random.shuffle(real1)
                random.shuffle(real2)
                cri_list.append(real1[0])
                acc_list.append(real2[0])
            return cri_list, acc_list
        elif self.mode == 'test':
            cri_list = []
            acc_list = []
            filename = os.path.join(self.root, self.mode)
            test_dir = os.listdir(filename)
            real_dir = os.path.join(filename, test_dir[0])
            test_morph_list = self.morph_list
            for morph_img in test_morph_list:
                real1 = []
                real2 = []
                real_1 = os.path.splitext(os.path.split(morph_img)[1])[0].split('_')[0]
                real_2 = os.path.splitext(os.path.split(morph_img)[1])[0].split('_')[2]
                for real in os.listdir(real_dir):
                    if os.path.splitext(real)[0].split('_')[0] == real_1:
                        real1.append(os.path.join(real_dir, real))
                    elif os.path.splitext(real)[0].split('_')[0] == real_2:
                        real2.append(os.path.join(real_dir, real))
                random.shuffle(real1)
                random.shuffle(real2)
                cri_list.append(real1[0])
                acc_list.append(real2[0])
            return cri_list, acc_list

    def __len__(self):
        return len(self.morph_list)

    def __getitem__(self, idx):
        real1_path, real2_path, morph_path = self.real1[idx], self.real2[idx], self.morph_list[idx]
        real1 = self.trasform(Image.open(real1_path).convert('RGB'))
        real2 = self.trasform(Image.open(real2_path).convert('RGB'))
        morph = self.trasform(Image.open(morph_path).convert('RGB'))
        return {'real1': real1, 'real2': real2, 'morph': morph, 'idxs': idx, 'morph_path': morph_path}


if __name__ == '__main__':
    sssset = FD_Datasets(root='./datasets/HUN_yqt_all_neutral', resize=256, mode='train')
    looooader = DataLoader(sssset, batch_size=4, shuffle=False, num_workers=0)
    for idx, batch in enumerate(looooader):
        real1, real2, morph, morph_path = batch['real1'], batch['real2'], batch['morph'], batch['morph_path']
        for i in range(4):
            if not os.path.exists(os.path.join(SAVE_PATH, '')):
                os.mkdir(os.path.join(SAVE_PATH, 'HUN_yqt_all_neutral'))
            save_demorph(real1[i], morph[i], real2[i],
                         path=os.path.join(SAVE_PATH, 'HUN_yqt_all_neutral'),
                         filename_path=os.path.split(morph_path[i])[1])
