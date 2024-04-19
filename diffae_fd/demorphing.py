import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from MIP_Datasets import MIP_Datasets
from FD_Datasets import FD_Datasets
from FFRL_Datasets import FFRL_Datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from templates import *
from model.separate_model import separate_model
import numpy as np

SAVE_PATH = '../evals/morph_stylegan/morph_stylegan'
BATCH_SIZE = 16

def tensor2im(var):
    var = var.cpu().detach().permute(1, 2, 0).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

# the function to save the restored facial image
def save_demorph(acc, mor, cri, de_cri, epoch, step, path, filename):
    acc = tensor2im(acc)
    mor = tensor2im(mor)
    cri = tensor2im(cri)
    de_cri = de_cri.cpu().permute(1, 2, 0).numpy()
    de_cri[de_cri < 0] = 0
    de_cri[de_cri > 1] = 1
    de_cri = de_cri * 255
    de_cri = Image.fromarray(de_cri.astype('uint8'))
    filename_path = filename
    res = np.concatenate([acc, mor, cri, de_cri], axis=1)
    print(os.path.join(path, filename))
    Image.fromarray(np.uint8(res)).save(os.path.join(path, filename_path))


def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loading the pre-trained diffusino autoencoders' param
conf = ffhq256_autoenc()
model = LitModel(conf)
model_state = torch.load(f'../checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(model_state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

# loading the trained separation network's param
separate_model = separate_model()
separate_model_state = torch.load(
    'pretrain_model/FFRL_15/morph_stylegan_without_MLP/8_best.pkl',
    map_location='cpu')
separate_model.load_state_dict(separate_model_state['state_dict'], strict=False)
separate_model.eval()
separate_model.to(device)

# loading the facial images in the test datasets
test_set = FFRL_Datasets(root='../datasets/facelab_london/morph_stylegan', resize=256, mode='test')
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# inference
for idx, batch in enumerate(test_loader):
    real1, real2, morph, morph_path = batch['real1'], batch['real2'], batch['morph'], batch['morph_path']
    cond_real1_real = model.encode(real1.to(device))
    cond_morph_real = model.encode(morph.to(device))
    cond_real2_separate = separate_model(cond_real1_real, cond_morph_real)
    xT_morph_real = model.encode_stochastic(morph.to(device), cond_morph_real, T=10)
    pred_3 = model.render(noise=xT_morph_real, cond=cond_real2_separate, T=20)

    for i in range(BATCH_SIZE):
        if not os.path.exists(os.path.join(SAVE_PATH, 'xT_morph_Noise_separate')):
            os.mkdir(os.path.join(SAVE_PATH, 'xT_morph_Noise_separate'))
        save_demorph(real1[i], morph[i], real2[i], pred_3[i], epoch=idx, step=i + idx * 2,
                     path=os.path.join(SAVE_PATH, 'xT_morph_Noise_separate'), filename=os.path.split(morph_path[i])[1])

