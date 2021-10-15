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

cutout = MyCutouts(cut_size)

def loss_fn(model, x, token, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t, token=token)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss

def loss_fn2(model, x, token, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t, token=token)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  xp = perturbed_x + score * std[:, None, None, None]**2
  token1 = perceptor.encode_image(normalize(cutout(xp))).float()
  loss1 = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss

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
  init_x = torch.randn(batch_size, 3, 96, 96, device=device) \
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

def main():

    #score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = ResUNet2(marginal_prob_std=marginal_prob_std_fn, is_token=True) #torch.nn.DataParallel(ResUNet())
    score_model = score_model.to(device)

    jit = True if float(torch.__version__[:3]) < 1.8 else False
    perceptor = clip.load("ViT-B/32", jit=jit)[0].eval().requires_grad_(False).to(device)
    cut_size = perceptor.visual.input_resolution

    id_name = 'pksp/clip_inv'
    n_epochs =   1000#@param {'type':'integer'}
    ## size of a mini-batch
    batch_size =  16 #@param {'type':'integer'}
    ## learning rate
    lr=1e-4 #@param {'type':'number'}
    sample_batch_size = 25
    data_dir = ['/data/pixelart/gif_ext/front'
                #'/data/pixelart/dragonflycave/gen4/Front',
                #'/data/pixelart/trainer',
                #'/data/pixelart/dragonflycave/fusion',
                ]
    image_size = 96
    num_workers = 4

    # TensorBoard
    writerpath = os.path.join('logs/summaries', id_name)
    if os.path.exists(writerpath):
        os.system("rm -rf "+writerpath)
    writer = SummaryWriter(writerpath)

    dataset = Dataset(data_dir, image_size, transparent = 0, aug_prob = 0.)
    data_loader = DataLoader(dataset, num_workers = num_workers, batch_size = batch_size, drop_last = True, shuffle=True, pin_memory=True)
    #dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(score_model.parameters(), lr=lr)

    #tqdm_epoch = tqdm.notebook.trange(n_epochs)
    for epoch in tqdm(range(n_epochs)): #tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
        #for x, y in data_loader:
            x = x.to(device)    
            token = perceptor.encode_image(normalize(cutout(x))).float()
            loss = loss_fn(score_model, x, token, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        # Update tensorboard.
        if epoch % 1 == 0:
            samples = Euler_Maruyama_sampler(score_model, 
                                                token,
                                                marginal_prob_std_fn,
                                                diffusion_coeff_fn, 
                                                sample_batch_size, 
                                                device=device)
            samples = samples.clamp(0.0, 1.0)
            sample_grid = make_grid(samples, nrow=int(np.sqrt(batch_size)))
            writer.add_image("Euler sampler",sample_grid,epoch)
            x = make_grid(x, nrow=8)
            writer.add_image("True image",x, epoch)
            writer.add_scalar('Average Loss', avg_loss/num_items , epoch)

        # Update the checkpoint after each epoch of training
        if epoch % 1 ==0:
            torch.save(score_model.state_dict(), os.path.join('checkpoints', id_name+'.pth') )

if __name__ == "__main__":
    sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
    main()

