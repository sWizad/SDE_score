import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
#import tqdm
from tqdm.auto import tqdm
from tools.dataset import *
from tools.utils import *
from tools.network import *

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import clip


device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  #t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  #return torch.tensor(sigma**t, device=device)
  return sigma**t
  
sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])


num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model, 
                           token,
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=num_steps, 
                           device='cuda', 
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  batch_size = token.shape[0]
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 3, 80, 80, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    #for time_step in tqdm(time_steps):
    for time_step in (time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, token) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x

def SDEdit(score_model, img, token, marginal_prob_std, diffusion_coeff, num_steps=200, sample_step=1,
                           device='cuda', eps=1e-3):
  print("Start sampling")
  batch_size = token.shape[0]
  with torch.no_grad():
      #img = img.repeat(n, 1, 1, 1)
      #mask = torch.zeros_like(img[0])
      x0 = img
      #x0 = (x0 - 0.5) * 2.
      #imshow(x0, title="Initial input")
      time_steps = torch.linspace(0.3, eps, num_steps, device=device)
      #print(num_steps,time_steps)
      step_size = time_steps[0] - time_steps[1]

      for it in range(sample_step):
          #e = torch.randn_like(x0)
          #a = (1 - betas).cumprod(dim=0).to(device)
          #x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
          #imshow(x, title="Perturb with SDE")
          #random_t = torch.rand(x0.shape[0], device=x0.device) * (1. - eps) + eps  
          #print(random_t.shape)
          random_t = 0.2*torch.ones(x0.shape[0], device=x0.device)
          #print(random_t.shape)
          z = torch.randn_like(x0)
          std = marginal_prob_std(random_t)
          x = x0 + z * std[:, None, None, None]

          with tqdm(total=num_steps, desc="Iteration {}".format(it)) as progress_bar:
              #for i in reversed(range(num_steps)):
              for time_step in (time_steps):   
                  batch_time_step = torch.ones(batch_size, device=device) * time_step
                  g = diffusion_coeff(batch_time_step)
                  mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, token) * step_size
                  x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x) 

                  '''
                  t = (torch.ones(n) * i).to(device)
                  mean, var = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                  logvar=logvar,
                                                                  betas=betas)
                  
                  noise = torch.randn_like(x)
                  x_ = mean + var * noise
                  x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                  #x = x_ * mask + x * (1-mask)
                  x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                  # added intermediate step vis
                  #if (i - 99) % 250 == 0:
                  #    imshow(x, title="Iteration {}, t={}".format(it, i))
                  '''
                  progress_bar.update(1)

          #x0[:, (mask != 1.)] = x[:, (mask != 1.)]
          #x0 = x * mask + x0 * (1-mask)
          #imshow(x, name + "{}".format(it))
      return x

def main():
    score_model = ResUNet(marginal_prob_std=marginal_prob_std_fn, is_token=True) #torch.nn.DataParallel(ResUNet())
    score_model = score_model.to(device)

    jit = True if float(torch.__version__[:3]) < 1.8 else False
    perceptor = clip.load("ViT-B/32", jit=jit)[0].eval().requires_grad_(False).to(device)
    cut_size = perceptor.visual.input_resolution
    cutout = MyCutouts(cut_size)

    id_name = 'pksp_clip3'
    n_epochs =   1000#@param {'type':'integer'}
    ## size of a mini-batch
    batch_size =  64 #@param {'type':'integer'}
    ## learning rate
    lr=1e-4 #@param {'type':'number'}
    sample_batch_size = 16
    data_dir = ['/data/pixelart/dragonflycave/gen4/Front',
                #'/data/pixelart/trainer',
                #'/data/pixelart/dragonflycave/fusion',
                ]
    image_size = 80
    num_workers = 4
    prom_txt = 'little deer #pixelart'

    token0 = perceptor.encode_text(clip.tokenize(prom_txt).to(device)).float()
    token0 = torch.repeat_interleave(token0,sample_batch_size,dim=0)

    ckpt = torch.load(os.path.join('checkpoints', id_name+'.pth'), map_location=device)
    score_model.load_state_dict(ckpt)
    
    dataset = Dataset(data_dir, image_size, transparent = 0, aug_prob = 0.)
    data_loader = DataLoader(dataset, num_workers = num_workers, batch_size = sample_batch_size, drop_last = True, shuffle=True, pin_memory=True)
    x = next(iter(data_loader))
    x = x.to(device)    
    token = perceptor.encode_image(normalize(cutout(x))).float()
    
    token = token#token0
    #token = token/torch.norm(token,dim=-1,keepdim=True)
    '''
    samples = SDEdit(score_model, x, 
                        token, 
                        marginal_prob_std_fn, 
                        diffusion_coeff_fn,
                        device=device)
    '''
    samples = Euler_Maruyama_sampler(score_model, 
                                        token,
                                        marginal_prob_std_fn,
                                        diffusion_coeff_fn, 
                                        sample_batch_size, 
                                        device=device)
    
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.savefig('res.png')


if __name__ == "__main__":
    sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
    main()