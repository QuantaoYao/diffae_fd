import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import torch
import pytorch_lightning
from torch.utils.data import DataLoader, Dataset
from FD_Datasets import FD_Datasets
from MIP_Datasets import MIP_Datasets
from templates import *
import matplotlib.pyplot as plt
from model.separate_model import separate_model
import torch.optim as optim
from tensorboardX import SummaryWriter
from FFRL_Datasets import FFRL_Datasets

EPOCH = 100
BATCH_SIZE = 16 if torch.cuda.is_available() else 2

# loading the facial images in the train and dev datasets
train_set = FFRL_Datasets(root='../datasets/facelab_london/morph_webmorph', resize=256, mode='train')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
dev_set = FFRL_Datasets(root='../datasets/facelab_london/morph_webmorph', resize=256, mode='dev')
dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)

print(
    f'total train:{len(train_loader.dataset)}\ttotal dev:{len(dev_loader.dataset)}\t')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
conf = ffhq256_autoenc()
model = LitModel(conf)
state = torch.load(f'../checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)
separate_model = separate_model().to(device)

optimizer = optim.Adam(separate_model.parameters(), lr=1e-3)
criteon = nn.MSELoss().to(device)
writer = SummaryWriter('../checkpoints/FFRL_15/morph_webmorph_without_attn')
best_val_loss = 10000
for epoch in range(EPOCH):
    print(f'-----------EPOCH:{epoch}--------------')
    separate_model.train()
    running_loss = 0
    for idx, batch in enumerate(train_loader):
        real1, real2, morph, morph_path = batch['real1'], batch['real2'], batch['morph'], batch['morph_path']
        cond_morph = model.encode(morph.to(device))
        cond_real1 = model.encode(real1.to(device))
        cond_real2 = model.encode(real2.to(device))
        optimizer.zero_grad()
        logits = separate_model(cond_real1, cond_morph)
        loss = criteon(logits, cond_real2)
        loss.backward()
        optimizer.step()
        writer.add_scalar(tag='train_loss', scalar_value=loss, global_step=epoch * len(train_loader) + idx)
        running_loss += loss
    print(f'running_loss', running_loss.item())
    # 验证
    separate_model.eval()
    with torch.no_grad():
        total_loss = 0
        for idx, batch in enumerate(dev_loader):
            real1, real2, morph, morph_path = batch['real1'], batch['real2'], batch['morph'], batch['morph_path']
            cond_morph = model.encode(morph.to(device))
            cond_real1 = model.encode(real1.to(device))
            cond_real2 = model.encode(real2.to(device))
            logits = separate_model(cond_real1, cond_morph)
            val_loss = criteon(logits, cond_real2)
            total_loss += val_loss
            writer.add_scalar('val_loss', val_loss, epoch * len(train_loader) + idx)
        print(f'total_val_loss', total_loss.item())
        print(f'best_val_loss', best_val_loss)
        torch.save({'state_dict': separate_model.state_dict(), 'epoch': epoch},
                   'pretrain_model/FFRL_15/morph_webmorph_without_attn/last_pkl')
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            torch.save({'state_dict': separate_model.state_dict(), 'epoch': epoch},
                       'pretrain_model/FFRL_15/morph_webmorph_without_attn/' + str(
                           epoch) + '_best.pkl')
            print('Save best statistics done!')
    if epoch % 100 == 0:
        torch.save({'state_dict': separate_model.state_dict(), 'epoch': epoch},
                   'pretrain_model/FFRL_15/morph_webmorph_without_attn/' + str(epoch) + '.pkl')
        print('save successful')
