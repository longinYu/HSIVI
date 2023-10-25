import torch
import torch.nn as nn
from . import utils
import torch.nn.functional as F
import math
from collections.abc import Iterable
from itertools import repeat


@utils.register_model(name='speed_ddpm_cifar')
class SpeedDDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target_model = UNet()
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.target_model.eval()
        self.phinet = UNet()
        if config.independent_log_gamma == 'use':
            if config.image_gamma=='use':
                log_scaled_gamma = nn.Parameter(torch.zeros((config.n_discrete_steps-1, config.image_channels, config.image_size, config.image_size)), requires_grad = True)
            else:
                log_scaled_gamma = nn.Parameter(torch.zeros((config.n_discrete_steps-1)), requires_grad = True)
        else:
            if config.image_gamma=='use':
                log_scaled_gamma = nn.Parameter(torch.zeros((config.image_channels, config.image_size, config.image_size)), requires_grad = True)
            else:                
                log_scaled_gamma = nn.Parameter(torch.zeros((1)), requires_grad = True)
        self.phinet.register_parameter('log_scaled_gamma', log_scaled_gamma)
        self.fnet = UNet()
    
    def reparameterize(self, x_t_plus_1, epsilon, sigma_t, sigma_t_plus_1, gamma_index=None):
        if self.config.independent_log_gamma == 'use' and gamma_index is not None:
            log_scaled_gamma = self.phinet.log_scaled_gamma.index_select(0, gamma_index)            
        else:
            if self.config.image_gamma=='use':
                log_scaled_gamma = self.phinet.log_scaled_gamma.unsqueeze(0)
            else:
                log_scaled_gamma = self.phinet.log_scaled_gamma

        # here, sigma_t_plus_1 = (1 - alpha_bar).sqrt()
        # here, sigma_t = (1 - alpha_bar_pre).sqrt()
        x0_t_p1 = (x_t_plus_1 - epsilon * sigma_t_plus_1) / (1-sigma_t_plus_1**2).sqrt()
        eta = self.config.eta 
        c1 = ( eta*((1 - (1-sigma_t_plus_1**2) / (1-sigma_t**2)) * sigma_t**2 / sigma_t_plus_1**2).sqrt())
        c2 = (sigma_t**2 - c1 ** 2).sqrt()
        x_t_mean = (1 - sigma_t**2).sqrt() * x0_t_p1 + c2 * epsilon 
        
        
        if self.config.image_gamma=='dis' and self.config.independent_log_gamma == 'use':
            sigma_t = sigma_t.view(x_t_mean.shape[0])
            c1 = c1.view(x_t_mean.shape[0])
        gamma = torch.exp(log_scaled_gamma) * c1 / sigma_t 
        std = torch.exp(log_scaled_gamma) * c1
        if self.config.image_gamma=='dis' and self.config.independent_log_gamma == 'use':
            std = std[:,None,None,None]
            gamma = gamma[:,None,None,None]
        
        if self.phinet.training:
            eps = torch.randn_like(x_t_plus_1, device=x_t_plus_1.device)
        else:
            non_zeros = gamma_index.nonzero()
            eps = torch.zeros_like(x_t_plus_1, device=x_t_plus_1.device)
            eps_rand = torch.randn_like(x_t_plus_1, device=x_t_plus_1.device)
            eps[non_zeros,...] = eps[non_zeros,...] + eps_rand[non_zeros,...]
 
        x_t_sample = x_t_mean + std * eps
        return x_t_sample, (eps, gamma)
    
    def forward_phinet(self, x_t_plus_1, time_cond, gamma_index=None, sigma_t=None, sigma_t_plus_1=None):
        epsilon = self.phinet(x_t_plus_1, time_cond)
        
        x_t_sample, eps_div_gamma = self.reparameterize(x_t_plus_1=x_t_plus_1, epsilon=epsilon, sigma_t=sigma_t, 
                                                        gamma_index=gamma_index,
                                                        sigma_t_plus_1=sigma_t_plus_1)
        
        return x_t_sample, eps_div_gamma

    def forward(self, x_t_plus_1, time_cond, gamma_index=None,sigma_t=None, object_='teacher', sigma_t_plus_1=None):
        #! here time_cond is intergers in [1, N-1], different from former variance.
        if object_ == 'teacher':
            return self.target_model(x_t_plus_1, time_cond) 
        elif object_ == 'fnet':
            return self.fnet(x_t_plus_1, time_cond)  
        else:
            return self.forward_phinet(x_t_plus_1=x_t_plus_1, time_cond=time_cond, sigma_t=sigma_t, 
                                       gamma_index=gamma_index, sigma_t_plus_1=sigma_t_plus_1)
    


#! this is the cifar10 UNet.
def ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        else:
            return tuple(repeat(x, n))
    parse.__name__ = name
    return parse

pair = ntuple(2, "pair")

def _initializer(x, scale=1.):
    """
    PyTorch Xavier uniform initialization: w ~ Uniform(-a, a), where a = gain * (6 / (fan_in + fan_out)) ** .5
    TensorFlow Variance-Scaling initialization (mode="fan_avg", distribution="uniform"):
    w ~ Uniform(-a, a), where a = (6 * scale / (fan_in + fan_out)) ** .5
    Therefore, gain = scale ** .5
    """
    return nn.init.xavier_uniform_(x, gain=math.sqrt(scale or 1e-10))

class Linear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            init_scale=1.
    ):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features, ), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.init_scale = init_scale
        self.reset_parameters()

    def reset_parameters(self):
        _initializer(self.weight, scale=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class Conv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            init_scale=1.
    ):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size = pair(kernel_size)
        self.weight = nn.Parameter(
            torch.empty((
                out_channels, in_channels // groups, kernel_size[0], kernel_size[1]
            ), dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels, ), dtype=torch.float32))
        else:
            self.register_parameter("bias", None)
        self.stride = pair(stride)
        self.padding = padding if isinstance(padding, str) else pair(padding)
        self.dilation = pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.init_scale = init_scale
        self.reset_parameter()

    def reset_parameter(self):
        _initializer(self.weight, scale=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def forward(self, x):
        return F.conv2d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups)

class SamePad2d(nn.Module):
    def __init__(self, kernel_size, stride, mode="constant", value=0.0):
        super(SamePad2d, self).__init__()
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.mode = mode
        self.value = value

    def forward(self, x):
        _, _, h, w = x.shape
        (k1, k2), (s1, s2) = self.kernel_size, self.stride
        h_pad, w_pad = s1 * math.ceil(h / s1 - 1) + k1 - h, s2 * math.ceil(w / s2 - 1) + k2 - w
        top_pad, bottom_pad = (math.floor(h_pad / 2), math.ceil(h_pad / 2)) if h_pad else (0, 0)
        left_pad, right_pad = (math.floor(w_pad / 2), math.ceil(w_pad / 2)) if w_pad else (0, 0)
        x = F.pad(x, pad=(left_pad, right_pad, top_pad, bottom_pad), mode=self.mode, value=self.value)
        return x


class Sequential(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input

@torch.jit.script
def get_timestep_embedding(timesteps, embed_dim: int, dtype: torch.dtype = torch.float32):
    """
    Adapted from fairseq/fairseq/modules/sinusoidal_positional_embedding.py
    The implementation is slightly different from the decription in Section 3.5 of [1]
    [1] Vaswani, Ashish, et al. "Attention is all you need."
     Advances in neural information processing systems 30 (2017).
    """
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(-torch.arange(half_dim, dtype=dtype, device=timesteps.device) * embed)
    embed = torch.outer(timesteps.ravel().to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    if embed_dim % 2 == 1:
        embed = F.pad(embed, [0, 1])  # padding the last dimension
    assert embed.dtype == dtype
    return embed



class DEFAULT_NORMALIZER(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32):
        super().__init__(num_groups=num_groups, num_channels=num_channels)


class AttentionBlock(nn.Module):
    normalize = DEFAULT_NORMALIZER

    def __init__(
            self,
            in_channels,
            mid_channels=None,
            out_channels=None
    ):
        super(AttentionBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = self.normalize(in_channels)
        self.project_in = Conv2d(in_channels, 3 * mid_channels, 1)
        self.project_out = Conv2d(mid_channels, out_channels, 1, init_scale=0.)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.skip = nn.Identity() if in_channels == out_channels else Conv2d(in_channels, out_channels, 1)

    @staticmethod
    def qkv(q, k, v):
        B, C, H, W = q.shape
        w = torch.einsum("bchw, bcHW -> bhwHW", q, k)
        w = torch.softmax(
            w.reshape(B, H, W, H * W) / math.sqrt(C), dim=-1
        ).reshape(B, H, W, H, W)
        out = torch.einsum("bhwHW, bcHW -> bchw", w, v)
        return out

    def forward(self, x, **kwargs):
        skip = self.skip(x)
        C = x.shape[1]
        assert C == self.in_channels
        q, k, v = self.project_in(self.norm(x)).chunk(3, dim=1)
        x = self.qkv(q, k, v).contiguous()
        x = self.project_out(x)
        return x + skip


class ResidualBlock(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = nn.SiLU()

    def __init__(
            self,
            in_channels,
            out_channels,
            embed_dim,
            drop_rate=0.
    ):
        super(ResidualBlock, self).__init__()
        self.norm1 = self.normalize(in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.fc = Linear(embed_dim, out_channels)
        self.norm2 = self.normalize(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1, init_scale=0.)
        self.skip = nn.Identity() if in_channels == out_channels else Conv2d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)

    def forward(self, x, t_emb):
        skip = self.skip(x)
        x = self.conv1(self.nonlinearity(self.norm1(x)))
        x += self.fc(self.nonlinearity(t_emb))[:, :, None, None]
        x = self.dropout(self.nonlinearity(self.norm2(x)))
        x = self.conv2(x)
        return x + skip


class UNet(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = nn.SiLU()

    def __init__(
            self,
            in_channels=3,
            hid_channels=128,
            out_channels=3,
            ch_multipliers=[1, 2, 2, 2],
            num_res_blocks=2,
            apply_attn=[False, True, False, False],
            time_embedding_dim=None,
            drop_rate=0.,
            resample_with_conv=True
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim or 4 * self.hid_channels
        levels = len(ch_multipliers)
        self.ch_multipliers = ch_multipliers
        if isinstance(apply_attn, bool):
            apply_attn = [apply_attn for _ in range(levels)]
        self.apply_attn = apply_attn
        self.num_res_blocks = num_res_blocks
        self.drop_rate = drop_rate
        self.resample_with_conv = resample_with_conv

        self.embed = Sequential(
            Linear(self.hid_channels, self.time_embedding_dim),
            self.nonlinearity,
            Linear(self.time_embedding_dim, self.time_embedding_dim)
        )
        self.in_conv = Conv2d(in_channels, hid_channels, 3, 1, 1)
        self.levels = levels
        self.downsamples = nn.ModuleDict({f"level_{i}": self.downsample_level(i) for i in range(levels)})
        mid_channels = ch_multipliers[-1] * hid_channels
        embed_dim = self.time_embedding_dim
        self.middle = Sequential(
            ResidualBlock(mid_channels, mid_channels, embed_dim=embed_dim, drop_rate=drop_rate),
            AttentionBlock(mid_channels),
            ResidualBlock(mid_channels, mid_channels, embed_dim=embed_dim, drop_rate=drop_rate)
        )
        self.upsamples = nn.ModuleDict({f"level_{i}": self.upsample_level(i) for i in range(levels)})
        self.out_conv = Sequential(
            self.normalize(hid_channels),
            self.nonlinearity,
            Conv2d(hid_channels, out_channels, 3, 1, 1, init_scale=0.)
        )

    def get_level_block(self, level):
        block_kwargs = {"embed_dim": self.time_embedding_dim, "drop_rate": self.drop_rate}
        if self.apply_attn[level]:
            def block(in_chans, out_chans):
                return Sequential(
                    ResidualBlock(in_chans, out_chans, **block_kwargs),
                    AttentionBlock(out_chans))
        else:
            def block(in_chans, out_chans):
                return ResidualBlock(in_chans, out_chans, **block_kwargs)
        return block

    def downsample_level(self, level):
        block = self.get_level_block(level)
        prev_chans = (self.ch_multipliers[level-1] if level else 1) * self.hid_channels
        curr_chans = self.ch_multipliers[level] * self.hid_channels
        modules = nn.ModuleList([block(prev_chans, curr_chans)])
        for _ in range(self.num_res_blocks - 1):
            modules.append(block(curr_chans, curr_chans))
        if level != self.levels - 1:
            if self.resample_with_conv:
                downsample = Sequential(
                    SamePad2d(3, 2),  # custom same padding
                    Conv2d(curr_chans, curr_chans, 3, 2))
            else:
                downsample = nn.AvgPool2d(2)
            modules.append(downsample)
        return modules

    def upsample_level(self, level):
        block = self.get_level_block(level)
        ch = self.hid_channels
        chs = list(map(lambda x: ch * x, self.ch_multipliers))
        next_chans = ch if level == 0 else chs[level - 1]
        prev_chans = chs[-1] if level == self.levels - 1 else chs[level + 1]
        curr_chans = chs[level]
        modules = nn.ModuleList([block(prev_chans + curr_chans, curr_chans)])
        for _ in range(self.num_res_blocks - 1):
            modules.append(block(2 * curr_chans, curr_chans))
        modules.append(block(next_chans + curr_chans, curr_chans))
        if level != 0:
            """
            Note: the official TensorFlow implementation specifies `align_corners=True`
            However, PyTorch does not support align_corners for nearest interpolation
            to see the difference, run the following example:
            ---------------------------------------------------------------------------
            import numpy as np
            import torch
            import tensorflow as tf
            
            x = np.arange(9.).reshape(3, 3)
            print(torch.nn.functional.interpolate(torch.as_tensor(x).reshape(1, 1, 3, 3), size=7, mode="nearest"))  # asymmetric
            print(tf.squeeze(tf.compat.v1.image.resize(tf.reshape(tf.convert_to_tensor(x), shape=(3, 3, 1)), size=(7, 7), method="nearest", align_corners=True)))  # symmetric
            ---------------------------------------------------------------------------
            """  # noqa
            upsample = [nn.Upsample(scale_factor=2, mode="nearest")]
            if self.resample_with_conv:
                upsample.append(Conv2d(curr_chans, curr_chans, 3, 1, 1))
            modules.append(Sequential(*upsample))
        return modules

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.hid_channels)
        t_emb = self.embed(t_emb)

        # downsample
        hs = [self.in_conv(x)]
        for i in range(self.levels):
            downsample = self.downsamples[f"level_{i}"]
            for j, layer in enumerate(downsample):  # noqa
                h = hs[-1]
                if j != self.num_res_blocks:
                    hs.append(layer(h, t_emb=t_emb).contiguous())
                else:
                    hs.append(layer(h).contiguous())

        # middle
        h = self.middle(hs[-1], t_emb=t_emb)

        # upsample
        for i in range(self.levels-1, -1, -1):
            upsample = self.upsamples[f"level_{i}"]
            for j, layer in enumerate(upsample):  # noqa
                if j != self.num_res_blocks + 1:
                    h = layer(torch.cat([h, hs.pop()], dim=1), t_emb=t_emb).contiguous()
                else:
                    h = layer(h).contiguous()

        h = self.out_conv(h)
        return h
