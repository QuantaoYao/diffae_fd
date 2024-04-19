import torch.nn as nn
import torch
from model.transformer_block import AttentionBlock


# [batch,512]
# [batch,512]
# have to add cross_attention
class separate_model(nn.Module):
    def __init__(self, input_dim=512):
        super(separate_model, self).__init__()
        self.attn = AttentionBlock(channels=512, num_head_channels=4)
        self.layer = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.SiLU(),
            nn.GroupNorm(32, input_dim * 4),
            nn.Linear(input_dim * 4, input_dim * 8),
            nn.SiLU(),
            nn.GroupNorm(32, input_dim * 8),
            nn.Linear(input_dim * 8, input_dim * 8),
            nn.SiLU(),
            nn.GroupNorm(32, input_dim * 8),
            nn.Linear(input_dim * 8, input_dim * 4),
            nn.SiLU(),
            nn.GroupNorm(32, input_dim * 4),
            nn.Linear(input_dim * 4, input_dim * 2),
            nn.SiLU(),
            nn.GroupNorm(32, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim),
            nn.SiLU(),
            nn.GroupNorm(32, input_dim)
        )

    def forward(self, criminal_cond, morph_cond):
        criminal_cond = torch.randn(16, 512)
        morph_cond = torch.randn(16, 512)
        criminal_cond_a = self.attn(criminal_cond, morph_cond)
        accomplice_cond = 2 * morph_cond - criminal_cond_a
        x = torch.cat([criminal_cond, morph_cond], dim=1)
        x = self.layer(x)
        return accomplice_cond + x


if __name__ == '__main__':
    a = torch.randn(16, 512)
    b = torch.randn(16, 512)
    model = separate_model()
    output = model(a, b)
    print(output.shape)
