import torch
import torch.nn as nn
import numpy as np

class MyCutouts(nn.Module):
    def __init__(self, cut_size):
        super().__init__()
        self.cut_size = cut_size
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))

    def forward(self,input):
        cutout = self.av_pool(input)
        return cutout


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)
    #def forward(self, cx, input):
    #    return self.main(cx, input) + self.skip(cx, input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
            nn.ReLU(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], axis=1)
    #def forward(self, cx, input):
    #    return np.concatenate([self.main(cx, input), self.skip(cx, input)], axis=1)

class Downsample2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.AvgPool2d(2) #nn.Sequential(*main)

    def forward(self,input):
        return self.main(input)

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        #self.weight = init.normal(out_features // 2, in_features, stddev=std)
        self.weight = nn.Parameter(torch.randn(out_features // 2, in_features) * std, requires_grad=False)

    def forward(self, cx, input):
        f = 2 * math.pi * input @ cx[self.weight].T
        return np.concatenate([f.cos(), f.sin()], axis=-1)

def expand_to_planes(input, shape):
    return input[..., None, None].broadcast_to(input.shape[:2] + shape[2:])

class ResUNet(nn.Module):
    def __init__(self, marginal_prob_std, embed_dim=256, is_token=False):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        #self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        # self.class_embed = nn.Embedding(10, 4)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, c))
        num_input = 3 + c
        if is_token:
            self.token_emb = nn.Sequential(
                nn.Linear(512, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, c),
                nn.ReLU(),
                nn.Linear(c, c),
            )
            num_input = num_input +c

        self.net = nn.Sequential(
            ResConvBlock(num_input, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                Downsample2d(),  # 80x80 -> 40x40
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    Downsample2d(),  # 40x40 -> 20x20
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    SkipBlock([
                        Downsample2d(),  # 20x20 -> 10x10
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        nn.Upsample(scale_factor=2,)
                    ]),
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    nn.Upsample(scale_factor=2)
                ]),
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2) # Haven't implemented ConvTranpose2d yet.
            ]),
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout=False),
        )
        """
        """
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        
    def forward(self, x, t, token=None):
        embed = expand_to_planes(self.act(self.embed(t)) , x.shape)
        if token is None:
            h = self.net(torch.cat([x, embed],1))
        else:
            tt = expand_to_planes(self.token_emb(token),  x.shape)
            a = torch.cat([x, tt, embed],1)
            h = self.net(torch.cat([x, tt, embed],1))
        h = h / self.marginal_prob_std(t)[:, None, None, None] 
        return h

    #def forward(self, cx, input, log_snrs, cond):
    #    timestep_embed = expand_to_planes(self.timestep_embed(cx, log_snrs[:, None]), input.shape)
    #    class_embed = expand_to_planes(self.class_embed(cx, cond), input.shape)
    #    return self.net(np.concatenate([input, class_embed, timestep_embed], axis=1))

class ResUNet2(nn.Module):
    def __init__(self, marginal_prob_std, embed_dim=256, is_token=False):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        #self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        # self.class_embed = nn.Embedding(10, 4)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, c))
        num_input = 3 + c
        if is_token:
            self.token_emb = nn.Sequential(
                nn.Linear(512, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, c),
                nn.ReLU(),
                nn.Linear(c, c),
            )
            num_input = num_input +c

        self.net = nn.Sequential(
            ResConvBlock(num_input, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                Downsample2d(),  # 80x80 -> 40x40
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    Downsample2d(),  # 40x40 -> 20x20
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    SkipBlock([
                        Downsample2d(),  # 20x20 -> 10x10
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        SkipBlock([
                            Downsample2d(),  # 10x10 -> 5x5
                            ResConvBlock(c * 2, c * 2, c * 2),
                            ResConvBlock(c * 2, c * 2, c * 2),
                            ResConvBlock(c * 2, c * 2, c * 2),
                            ResConvBlock(c * 2, c * 2, c * 2),
                            nn.Upsample(scale_factor=2,)
                        ]),
                        ResConvBlock(c * 4, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        nn.Upsample(scale_factor=2,)
                    ]),
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    nn.Upsample(scale_factor=2)
                ]),
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2) # Haven't implemented ConvTranpose2d yet.
            ]),
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout=False),
        )
        """
        """
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        
    def forward(self, x, t, token=None):
        embed = expand_to_planes(self.act(self.embed(t)) , x.shape)
        if token is None:
            h = self.net(torch.cat([x, embed],1))
        else:
            tt = expand_to_planes(self.token_emb(token),  x.shape)
            a = torch.cat([x, tt, embed],1)
            h = self.net(torch.cat([x, tt, embed],1))
        h = h / self.marginal_prob_std(t)[:, None, None, None] 
        return h

## from Score base

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, output_padding=1)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 3, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)
    #print("h1",h1.shape)
    #print("h2",h2.shape)
    #print("h3",h3.shape)
    #print("h4",h4.shape)

    # Decoding path
    h = self.tconv4(h4)
    #print("h",h.shape)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    #print("h",h.shape)
    h = self.act(h)
    #print("h",h.shape)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h

## My design
class MyMLP(nn.Module):
    def __init__(self, marginal_prob_std, embed_dim=256, is_token=False):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        #self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        # self.class_embed = nn.Embedding(10, 4)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, c))

        self.net = nn.Sequential(
                nn.Linear(512+512+c, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 512),
        )
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        
    def forward(self, x, t, token=None):
        embed = self.act(self.embed(t))
        h = self.net(torch.cat([x, token, embed],1))
        h = h / self.marginal_prob_std(t)[:, None,] 
        return h