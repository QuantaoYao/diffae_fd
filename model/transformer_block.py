import torch
import torch as th
import torch.nn as nn
import math


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, kv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        q_bs, q_width, q_length = q.shape
        bs, width, length = kv.shape
        assert q_bs == bs and q_length == length
        assert (q_width + width) % (3 * self.n_heads) == 0
        ch = (q_width + width) // (3 * self.n_heads)
        # q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        q = q.reshape(q_bs * self.n_heads, ch, q_length)
        k, v = kv.reshape(bs * self.n_heads, ch * 2, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.channels)
        # self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.q = nn.Conv1d(channels, channels, 1)
        self.kv = nn.Conv1d(channels, channels * 2, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x, y):
        return self._forward(x, y)

    def _forward(self, x, y):
        assert x.shape == y.shape
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        y = y.reshape(b, c, -1)
        # qkv = self.qkv(self.norm(x))
        q = self.q(self.norm(x))
        kv = self.kv(self.norm(y))
        h = self.attention(q, kv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


if __name__ == '__main__':
    a = torch.rand(1, 512)
    b = torch.rand(1, 512)
    model = AttentionBlock(channels=512, num_head_channels=-1)
    output = model(a, b)
    print(output.shape)
