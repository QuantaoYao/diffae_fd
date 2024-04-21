import sys
import os
import matplotlib.pyplot as plt
import torch
import pytorch_lightning
from templates import *
import matplotlib.pyplot as plt
from model.separate_model import separate_model
import torch.optim as optim
from tensorboardX import SummaryWriter

EPOCH = 100
BATCH_SIZE = 16

# Read the data of the training set and validation set in their own way
train_set = ""
dev_set = ""

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)

print(
    f'total train:{len(train_loader.dataset)}\ttotal dev:{len(dev_loader.dataset)}\t')

# loading the pre-trained diffusino autoencoders' param
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
# Draw loss curve
writer = SummaryWriter('../checkpoints/')
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
    # validation
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
                   'pretrain_model/last_pkl')
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            torch.save({'state_dict': separate_model.state_dict(), 'epoch': epoch},
                       'pretrain_model/' + str(
                           epoch) + '_best.pkl')
            print('Save best statistics done!')
    if epoch % 100 == 0:
        torch.save({'state_dict': separate_model.state_dict(), 'epoch': epoch},
                   'pretrain_model/' + str(epoch) + '.pkl')
        print('save successful')
