"""
run.py
"""
from tqdm import tqdm
import argparse, os, json, sys
sys.path.insert(0,'/home/niell_lab/Documents/github/FreelyMovingEphys/')
from project_analysis.inception_loop.base import CorePlusReadout2d
from project_analysis.inception_loop.datasets import WorldcamDataset3D
from torch.utils.data import Dataset, DataLoader
from project_analysis.inception_loop.core import Stacked2dCore
import torch.nn as nn
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='/home/niell_lab/data/freely_moving_ephys/inception_loop/inputs/train_history_dropnans.csv')
    parser.add_argument('--test_csv', type=str, default='/home/niell_lab/data/freely_moving_ephys/inception_loop/inputs/val_history_data.csv')
    parser.add_argument('--root_dir', type=str, default='/home/niell_lab/data/freely_moving_ephys/inception_loop/inputs/')
    args = parser.parse_args()
    return args

def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X = X.squeeze(dim=1)
        # compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred, y.squeeze().to(device))
        # print(pred.shape) # should be (64, 6, 128, 128)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main(train_csv, test_csv, root_dir):

    history_size = 0

    training_data = WorldcamDataset3D(train_csv, history_size, root_dir, transform=transforms.ToTensor())
    # testing_data = WorldcamDataset3D(test_csv, history_size, root_dir, transform=transforms.ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=64*38) # add num_workers
    # test_dataloader = DataLoader(testing_data, batch_size=64)

    input_channels = 3
    hidden_channels = 2
    input_kern = 7
    hidden_kern = 27

    learning_rate = 1e-5

    model = Stacked2dCore(input_channels, hidden_channels, input_kern, hidden_kern)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        # test_loop(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == '__main__':
    args = get_args()
    main(args.train_csv, args.test_csv, args.root_dir)
