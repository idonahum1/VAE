"""NICE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, latent_dim,device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )

    def sample(self,sample_size,mu=None,logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        with torch.no_grad():
            if mu==None:
                mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
            if logvar == None:
                logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)
            # first we generate z from the prior z ~ N(0,1)
            z = torch.randn((sample_size,self.latent_dim)).to(self.device)
            # then we upsample z
            z = self.upsample(z).view(-1, 64, 7, 7)
            # then we decode z
            recon = self.decoder(z)
            return recon

    def z_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def loss(self,x,recon,mu,logvar):
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
        # remember that kl between two gaussians is analytically solvable as
        # kl = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2)/2sigma2^2 - 1/2
        # since z ~ N(0,1) and we have N(mu,logvar) we can use the formula
        # kl = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) after some algebra
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def forward(self, x):
        x = self.encoder(x).view(-1, 64 * 7 * 7)
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.upsample(self.z_sample(mu, logvar)).view(-1, 64, 7, 7)
        recon = self.decoder(z)
        return recon, mu, logvar