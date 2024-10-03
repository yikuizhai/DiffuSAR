from functools import partial
import math

import torch
import torch.fft as fft
import torch.nn as nn
from tqdm import tqdm


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, self.groups, c // self.groups, h, w)  # group
        x = x.transpose(1, 2).contiguous().view(n, -1, h, w)  # shuffle
        return x


class ConvBnSiLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.module = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.SiLU(inplace=True))

    def forward(self, x):
        return self.module(x)


class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2),
                                     nn.BatchNorm2d(in_channels // 2),
                                     ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0))
        self.branch2 = nn.Sequential(ConvBnSiLu(in_channels // 2, in_channels // 2, 1, 1, 0),
                                     nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2),
                                     nn.BatchNorm2d(in_channels // 2),
                                     ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0))
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        x = self.channel_shuffle(x)
        return x


class ResidualDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels),
                                     nn.BatchNorm2d(in_channels),
                                     ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0))
        self.branch2 = nn.Sequential(ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0),
                                     nn.Conv2d(out_channels // 2, out_channels // 2, 3, 2, 1, groups=out_channels // 2),
                                     nn.BatchNorm2d(out_channels // 2),
                                     ConvBnSiLu(out_channels // 2, out_channels // 2, 1, 1, 0))
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        x = self.channel_shuffle(x)

        return x


class TimeMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, out_dim))
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb

        return self.act(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.conv0 = nn.Sequential(*[ResidualBottleneck(in_channels, in_channels) for i in range(3)],
                                   ResidualBottleneck(in_channels, out_channels // 2))

        self.time_mlp = TimeMLP(embedding_dim=time_embedding_dim, hidden_dim=out_channels, out_dim=out_channels // 2)
        self.conv1 = ResidualDownsample(out_channels // 2, out_channels)

    def forward(self, x, t=None):
        x_shortcut = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x_shortcut, t)
        x = self.conv1(x)

        return [x, x_shortcut]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, fft_b=1.5, fft_s=1.0, fft_r=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv0 = nn.Sequential(*[ResidualBottleneck(in_channels, in_channels) for i in range(3)],
                                   ResidualBottleneck(in_channels, in_channels // 2))

        self.time_mlp = TimeMLP(embedding_dim=time_embedding_dim, hidden_dim=in_channels, out_dim=in_channels // 2)
        self.conv1 = ResidualBottleneck(in_channels // 2, out_channels // 2)

        self.B = fft_b
        self.S = fft_s
        self.R = fft_r
        print(f'testing fft with hyperparameters: B: {self.B}, S: {self.S}, R: {self.R}')

    def forward(self, x, x_shortcut, t=None):
        x = self.upsample(x)

        hidden_mean = x.mean(1).unsqueeze(1)
        B = hidden_mean.shape[0]
        hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(
            2).unsqueeze(3)

        C = x.shape[1]
        x[:, :C // 2] = x[:, :C // 2].clone() * ((self.B - 1) * hidden_mean + 1)

        x_shortcut = self._fourier_filter(x_shortcut, threshold=self.R, scale=self.S, device=x_shortcut.device)

        x = torch.cat([x, x_shortcut], dim=1)
        x = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x, t)
        x = self.conv1(x)

        return x

    @staticmethod
    def _fourier_filter(x, threshold, scale, device='cuda'):
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W)).to(device)
        crow, ccol = H // 2, W // 2
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
        x_freq = x_freq * mask

        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

        return x_filtered


class Unet(nn.Module):
    def __init__(self, timesteps, time_embedding_dim, in_channels=3, out_channels=2, base_dim=32,
                 dim_mults=[2, 4, 8, 16], embedding_type='simple', fft_b=1.5, fft_s=1.0, fft_r=1):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple))
        assert base_dim % 2 == 0

        channels = self._cal_channels(base_dim, dim_mults)

        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)

        if embedding_type == 'simple':
            self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        elif embedding_type == 'sinusoidal':
            self.time_embedding = partial(self._sinusoidal_embedding, dim=time_embedding_dim, max_period=timesteps)
        else:
            raise ValueError('unknown embedding type')

        self.encoder_blocks = nn.ModuleList([EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels])
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(c[1], c[0], time_embedding_dim, fft_b, fft_s, fft_r) for c in channels[::-1]])

        self.mid_block = nn.Sequential(*[ResidualBottleneck(channels[-1][1], channels[-1][1]) for i in range(2)],
                                       ResidualBottleneck(channels[-1][1], channels[-1][1] // 2))

        self.final_conv = nn.Conv2d(in_channels=channels[0][0] // 2, out_channels=out_channels, kernel_size=1)

    def forward(self, x, t=None):
        x = self.init_conv(x)
        if t is not None:
            t = self.time_embedding(t)
        encoder_shortcuts = []
        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, t)
            encoder_shortcuts.append(x_shortcut)
        x = self.mid_block(x)
        encoder_shortcuts.reverse()
        for decoder_block, shortcut in zip(self.decoder_blocks, encoder_shortcuts):
            x = decoder_block(x, shortcut, t)
        x = self.final_conv(x)

        return x

    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim] + [base_dim * x for x in dim_mults]
        channels = list(zip(dims[:-1], dims[1:]))
        return channels

    @staticmethod
    def _sinusoidal_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class Diffusion(nn.Module):
    def __init__(self, image_size, in_channels, time_embedding_dim=256, timesteps=1000, base_dim=32,
                 dim_mults=None, fft_b=1.5, fft_s=1.0, fft_r=1):
        super().__init__()
        if dim_mults is None:
            dim_mults = [1, 2, 4, 8]
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size

        betas = self._cosine_variance_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))

        self.model = Unet(timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults, fft_b, fft_s,
                          fft_r)

    def forward(self, x, noise):
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        x_t = self._forward_diffusion(x, t, noise)
        pred_noise = self.model(x_t, t)

        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda"):
        x_t = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size)).to(device)
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise)

        x_t = (x_t + 1.) / 2.

        return x_t

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas

    def _forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape
        # q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * x_0 + \
            self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * noise

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise):
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        mean = (1. / torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            std = 0.0

        return mean + std * noise

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise):
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)

        x_0_pred = torch.sqrt(1. / alpha_t_cumprod) * x_t - torch.sqrt(1. / alpha_t_cumprod - 1.) * pred
        x_0_pred.clamp_(-1., 1.)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            mean = (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred + \
                   ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)) * x_t

            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            mean = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred  # alpha_t_cumprod_prev=1 since 0!=1
            std = 0.0

        return mean + std * noise
