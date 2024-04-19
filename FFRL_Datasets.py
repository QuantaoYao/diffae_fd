import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import numpy as np

SAVE_PATH = './datasets_check/facelab_london/'


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


class FFRL_Datasets(Dataset):
    def __init__(self,
                 root,
                 resize,
                 mode
                 ):
        '''

        :param root: 数据集目录
        :param resize: 图片resize
        :param mode: 数据集模式
        '''
        super(FFRL_Datasets, self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.resize, self.resize)),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )
        self.real1, self.real2, self.morph = self.get_morphList_realList()

    def get_morphList_realList(self):
        count_A = 0
        count_B = 0
        morph_list = []
        real1_list = []
        real2_list = []
        filename = os.path.join(self.root, self.mode)
        morph_dir = os.path.join(filename, os.listdir(filename)[1])
        real_dir = os.path.join(filename, os.listdir(filename)[0])
        for img in os.listdir(morph_dir):
            morph_list.append(os.path.join(morph_dir, img))
        for img in morph_list:
            real1 = os.path.splitext(os.path.split(img)[1])[0].split('_')[0]
            real2 = os.path.splitext(os.path.split(img)[1])[0].split('_')[1]
            for real in os.listdir(real_dir):
                if os.path.splitext(real)[0].split('_')[0] == real1 and os.path.splitext(real)[0].split('_')[1] == '03':
                    real1_list.append(os.path.join(real_dir, real))
            for real in os.listdir(real_dir):
                if os.path.splitext(real)[0].split('_')[0] == real2 and os.path.splitext(real)[0].split('_')[1] == '03':
                    real2_list.append(os.path.join(real_dir, real))
        return real1_list, real2_list, morph_list

    def __len__(self):
        return len(self.morph)

    def __getitem__(self, idx):
        real1_path, real2_path, morph_path = self.real1[idx], self.real2[idx], self.morph[idx]
        real1 = self.transform(Image.open(real1_path).convert('RGB'))
        real2 = self.transform(Image.open(real2_path).convert('RGB'))
        morph = self.transform(Image.open(morph_path).convert('RGB'))
        return {'real1': real1, 'real2': real2, 'morph': morph, 'morph_path': morph_path}


if __name__ == '__main__':
    sssset = FFRL_Datasets(root='./datasets/facelab_london/morph_webmorph', resize=256, mode='test')
    looooader = DataLoader(sssset, batch_size=4, shuffle=False, num_workers=0)
    for idx, batch in enumerate(looooader):
        real1, real2, morph, morph_path = batch['real1'], batch['real2'], batch['morph'], batch['morph_path']
        for i in range(4):
            if not os.path.exists(os.path.join(SAVE_PATH, '')):
                os.mkdir(os.path.join(SAVE_PATH, 'morph_webmorph'))
            save_demorph(real1[i], morph[i], real2[i],
                         path=os.path.join(SAVE_PATH, 'morph_webmorph'),
                         filename_path=os.path.split(morph_path[i])[1])


