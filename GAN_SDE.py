
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import functools
from tqdm.auto import tqdm
from tools.dataset import *
from tools.utils import *
from tools.network import *
from stylegan2.model import Generator

import clip

'''
g_ema = Generator(1024, 512, 8)
g_ema.load_state_dict(torch.load('checkpoints/stylegan2-ffhq-config-f.pt')["g_ema"], strict=False)
g_ema.eval()
g_ema = g_ema.to('cuda')

mean_latent = g_ema.mean_latent(4096)
latent_code_init_not_trunc = torch.randn(1, 512).cuda()
with torch.no_grad():
    img_orig, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
                                truncation=0.7, truncation_latent=mean_latent)
    image = ToPILImage()(make_grid(img_orig.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
    h, w = image.size

    plt.imshow(image, vmin=0., vmax=1.)
    plt.savefig('res.png')
    print(latent_code_init.shape)
    print(torch.sum((latent_code_init[:,0]-latent_code_init[:,1])**2))
    #img_orig2, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
    #                            truncation=0.7, truncation_latent=mean_latent)

    img_orig2, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)
    image = ToPILImage()(make_grid(img_orig2.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
    h, w = image.size

    plt.imshow(image, vmin=0., vmax=1.)
    plt.savefig('res2.png')
exit()
'''

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
  perturbed_x = x + z * std[:, None]
  score = model(perturbed_x, random_t, token)
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))
  return loss

num_steps =  100#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model, 
                           token,
                           marginal_prob_std,
                           diffusion_coeff, 
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
  init_x = torch.randn(batch_size, 512, device=device) \
    * marginal_prob_std(t)[:, None,]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    #for time_step in tqdm(time_steps):
    for time_step in (time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None] * score_model(x, batch_time_step, token) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None,] * torch.randn_like(x)    
  # Do not include any noise in the last sampling step.
  return mean_x
signal_to_noise_ratio = 0.16 #@param {'type':'number'}

## The number of sampling steps.
num_steps =  100#@param {'type':'integer'}
def pc_sampler(score_model, 
               token,
               marginal_prob_std,
               diffusion_coeff,
               num_steps=num_steps, 
               snr=signal_to_noise_ratio,                
               device='cuda',
               eps=1e-3):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
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
  init_x = torch.randn(batch_size, 512, device=device) * marginal_prob_std(t)[:, None,]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    #for time_step in tqdm.notebook.tqdm(time_steps):     
    for time_step in (time_steps):       
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step, token)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)
      x_mean = x + (g**2)[:, None,] * score_model(x, batch_time_step, token) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, ] * torch.randn_like(x)      
    
    # The last step does not include any noise
    return x_mean

def main():
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load('checkpoints/stylegan2-ffhq-config-f.pt')["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    jit = True if float(torch.__version__[:3]) < 1.8 else False
    perceptor = clip.load("ViT-B/32", jit=jit)[0].eval().requires_grad_(False).to(device)
    cut_size = perceptor.visual.input_resolution
    cutout = MyCutouts(cut_size)

    score_model = MyMLP(marginal_prob_std=marginal_prob_std_fn, is_token=True) #torch.nn.DataParallel(ResUNet())
    score_model = score_model.to(device)

    id_name = 'stylegan'
    model_dir = 'clipgan'
    n_epochs =   10000
    batch_size =  25 
    sample_batch_size =  4 
    lr=1e-3

    writerpath = os.path.join('logs/summaries',model_dir, id_name)
    if os.path.exists(writerpath):
        os.system("rm -rf "+writerpath)
    writer = SummaryWriter(writerpath)
    os.makedirs(os.path.join('checkpoints', model_dir),exist_ok=True)

    optimizer = Adam(score_model.parameters(), lr=lr)
    mean_latent = g_ema.mean_latent(4096)

    avg_loss = 0.
    num_items = 0
    for epoch in tqdm(range(n_epochs)): #tqdm_epoch:
        latent_code_init_not_trunc = torch.randn(batch_size, 512).cuda()
        with torch.no_grad():
            img_orig, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=0.7, truncation_latent=mean_latent)
            token = perceptor.encode_image(normalize(cutout(img_orig))).float()

        loss = loss_fn(score_model, latent_code_init_not_trunc, token, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        avg_loss += loss.item() * token.shape[0]
        num_items += token.shape[0]

        # Update tensorboard.
        if epoch % 10 == 0:
            writer.add_scalar('Average Loss', avg_loss/num_items , epoch)
            avg_loss = 0.
            num_items = 0

        if epoch % 20 == 0:
            img_orig = make_grid(img_orig[:sample_batch_size], nrow=int(np.sqrt(sample_batch_size)), normalize=True, scale_each=True, range=(-1, 1), padding=0)
            writer.add_image("True image",img_orig, epoch)
            token = token[:sample_batch_size]
            samples = Euler_Maruyama_sampler(score_model, 
                                                token,
                                                marginal_prob_std_fn,
                                                diffusion_coeff_fn, 
                                                device=device)
            with torch.no_grad():
                img_syn, latent_code_init = g_ema([samples], return_latents=True,
                                            truncation=0.7, truncation_latent=mean_latent)
            sample_grid = make_grid(img_syn, nrow=int(np.sqrt(sample_batch_size)), normalize=True, scale_each=True, range=(-1, 1), padding=0)
            writer.add_image("sampling/Euler sampler",sample_grid,epoch)
            samples = pc_sampler(score_model, 
                                    token,
                                    marginal_prob_std_fn,
                                    diffusion_coeff_fn, 
                                    device=device)
            with torch.no_grad():
                img_syn, latent_code_init = g_ema([samples], return_latents=True,
                                            truncation=0.7, truncation_latent=mean_latent)
            sample_grid = make_grid(img_syn, nrow=int(np.sqrt(sample_batch_size)), normalize=True, scale_each=True, range=(-1, 1), padding=0)
            writer.add_image("sampling/Predictor-corrector",sample_grid,epoch)

        # Update the checkpoint after each epoch of training
        if epoch % 500 == 0:
            torch.save(score_model.state_dict(), os.path.join('checkpoints', model_dir,id_name+'.pth') )

if __name__ == "__main__":
    sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
    main()