from torch.nn.modules.loss import MSELoss
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch import optim

class GaborModel(nn.Module):
    def __init__(self, imgsize=(60,80), init_params=None):
        super().__init__()
        
        self.use_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if init_params is None:
            self.sigma = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
            self.theta = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
            self.Lambda = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
            self.psi = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
            self.gamma = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
            self.xoffset = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
            self.yoffset = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        else:
            init_params = dict([key, float(val)] for key, val in init_params.items())
            
            self.sigma = nn.Parameter(torch.tensor(init_params['sigma'], requires_grad=True, dtype=torch.float))
            self.theta = nn.Parameter(torch.tensor(init_params['theta'], requires_grad=True, dtype=torch.float))
            self.Lambda = nn.Parameter(torch.tensor(init_params['lambda'], requires_grad=True, dtype=torch.float))
            self.psi = nn.Parameter(torch.tensor(init_params['psi'], requires_grad=True, dtype=torch.float))
            self.gamma = nn.Parameter(torch.tensor(init_params['gamma'], requires_grad=True, dtype=torch.float))
            self.xoffset = nn.Parameter(torch.tensor(init_params['xoffset'], requires_grad=True, dtype=torch.float))
            self.yoffset = nn.Parameter(torch.tensor(init_params['yoffset'], requires_grad=True, dtype=torch.float))
            
        self.clipfreq = 5
        self.clip_params()    
        
        self.img_height = torch.tensor(imgsize[0], dtype=torch.float).to(self.use_device)
        self.img_width = torch.tensor(imgsize[1], dtype=torch.float).to(self.use_device)
        
        self.to(self.use_device)
        
        self.double()
        
    def clip_params(self):
        """ Restrict model parameters between ranges.
        """
        with torch.no_grad():
            self.sigma.clamp_(4, 20)
            self.theta.clamp_(0, torch.tensor(np.pi)*2)
            self.Lambda.clamp_(0.1, 1)
            self.psi.clamp_(0, torch.tensor(np.pi)*2)
            self.gamma.clamp_(0.5, 4)
            self.xoffset.clamp_(-0.3, 0.3)
            self.yoffset.clamp_(-0.3, 0.3)
    
    def generate_gabor(self):
        """
        """
        
        center = (self.yoffset*self.img_height, self.xoffset*self.img_width)
        Lambda = self.Lambda*200

        sigma_x = self.sigma
        sigma_y = self.sigma / self.gamma

        ymax, xmax = (self.img_height, self.img_width)
        xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
        xmin = -xmax
        ymin = -ymax
        (y, x) = torch.meshgrid(torch.arange(ymin, ymax+1).to(self.use_device), torch.arange(xmin, xmax+1).to(self.use_device))
        
        x_theta = (x-center[0]) * torch.cos(self.theta) + (y-center[1]) * torch.sin(self.theta)
        y_theta = -(x-center[0]) * torch.sin(self.theta) + (y-center[1]) * torch.cos(self.theta)

        gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(2 * torch.tensor(np.pi) / Lambda * x_theta + self.psi)
    
        return gb
    
    def fit(self, rf_input, epochs=10000):
        """ Fit gabor filter to true recetive field.

        Parameters:
        rf_input (np.array): 2D receptive field of a single neuron
        epochs (int): epochs over which to train model
        lr (float): learning rate

        Returns:
        gabor (np.array): solution as 2D array
        params (dict): dictionary of best gabor filter parameters
        """
        
        actual_rf = torch.tensor(rf_input, dtype=torch.double).to(self.use_device)
        
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        for t in range(epochs):
            gabor = self.generate_gabor()
            loss = loss_fn(gabor, actual_rf)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if t % self.clipfreq == 0:
                self.clip_params()
                
        params = {'sigma':float(self.sigma.cpu().detach()), 'theta':float(self.theta.cpu().detach()),
                  'lambda':float(self.Lambda.cpu().detach()), 'psi':float(self.psi.cpu().detach()),
                  'gamma':float(self.gamma.cpu().detach()), 'xoffset':float(self.xoffset.cpu().detach()),
                  'yoffset':float(self.yoffset.cpu().detach())}
        
        return gabor.cpu().detach().numpy(), params