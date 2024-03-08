"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from VAE import Model


def train(vae, trainloader, optimizer, epoch):
    vae.train()  # set to training mode
    epoch_total_loss = 0
    for data in tqdm(trainloader, desc=f"Training Epoch {epoch} - Batch:",
                     bar_format="{desc} {n_fmt}/{total_fmt}|{bar}|"):
        inputs, _ = data
        inputs = inputs.to(vae.device)
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(inputs)
        loss = vae.loss(inputs, x_recon, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_total_loss += loss.item()

    return epoch_total_loss / len(trainloader)


def test(vae, testloader, epoch):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        test_loss = 0
        for data in tqdm(testloader, desc=f"Testing Epoch {epoch} - Batch:",
                         bar_format="{desc} {n_fmt}/{total_fmt}|{bar}|"):
            inputs, _ = data
            inputs = inputs.to(vae.device)
            x_recon, mu, logvar = vae(inputs)
            loss = vae.loss(inputs, x_recon, mu, logvar)
            test_loss += loss.item()
        return test_loss / len(testloader)

def dequantize(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(dequantize), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)
    train_elbo, test_elbo = [], []
    for epoch in range(args.epochs):
        currect_train_elbo = train(vae, trainloader, optimizer, epoch+1)
        train_elbo.append(currect_train_elbo)
        current_test_elbo = test(vae, testloader, epoch+1)
        test_elbo.append(current_test_elbo)
        print(f"Summary epoch {epoch+1}/{args.epochs} - Train ELBO: {currect_train_elbo:.2f} , Test ELBO: {current_test_elbo:.2f}")
    fig, ax = plt.subplots()
    ax.plot(train_elbo)
    ax.plot(test_elbo)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ELBO')
    ax.legend(['Train ELBO','Test ELBO'])
    sample = vae.sample(args.sample_size)
    torchvision.utils.save_image(sample, './samples/' + filename + '.png', nrow=8)
    fig.savefig('./plots/'+ filename + '.png')
    torch.save(vae.state_dict(), './models/' + filename + '.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
