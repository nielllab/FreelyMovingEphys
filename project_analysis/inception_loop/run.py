"""
run.py
"""
from tqdm import tqdm
from base import CorePlusReadout2d
from datasets import WorldcamDataset
from torch.utils.data import Dataset, DataLoader
from core import Stacked2dCore
import argparse, os, json
import torch.nn as nn
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='/home/niell_lab/data/freely_moving_ephys/inception_loop/inputs/WC_Train_Data.csv')
    parser.add_argument('--test_csv', type=str, default='/home/niell_lab/data/freely_moving_ephys/inception_loop/inputs/WC_Val_Data.csv')
    parser.add_argument('--root_dir', type=str, default='/home/niell_lab/data/freely_moving_ephys/inception_loop/inputs/')
    args = parser.parse_args()
    return args

def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    for batch, X in enumerate(dataloader):

        # compute prediction and loss
        pred = model(X.to(device))
        # loss = loss_fn(pred, y)
        print(pred.shape) # should be (64, 6, 128, 128)
        # backpropagation
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X in dataloader:
            pred = model(X.to(device))
            # test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main(train_csv, test_csv, root_dir):

    training_data = WorldcamDataset(train_csv, root_dir, transform=transforms.ToTensor())
    testing_data = WorldcamDataset(test_csv, root_dir, transform=transforms.ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=64) # add num_workers
    test_dataloader = DataLoader(testing_data, batch_size=64)

    input_channels = 3
    hidden_channels = 2
    input_kern = 7
    hidden_kern = 27

    learning_rate = 1e-3

    model = Stacked2dCore(input_channels, hidden_channels, input_kern, hidden_kern)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        # test_loop(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == '__main__':
    args = get_args()
    main(args.train_csv, args.test_csv, args.root_dir, args.phy_unit)
