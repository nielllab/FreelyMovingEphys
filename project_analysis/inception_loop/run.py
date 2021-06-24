"""
run.py
"""
from tqdm import tqdm
import argparse, os, json, sys
sys.path.insert(0,'/home/niell_lab/Documents/github/FreelyMovingEphys/')
from project_analysis.inception_loop.base import CorePlusReadout2d
from project_analysis.inception_loop.datasets import WorldcamDataset3D
from torch.utils.data import Dataset, DataLoader, Subset
from project_analysis.inception_loop.core import Stacked3dCore
import torch.nn as nn
import torch
from torchvision import transforms
from test_tube import Experiment
from torchvision import transforms
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='/home/niell_lab/data/freely_moving_ephys/inception_loop/inputs/train_history_dropnans.csv')
    parser.add_argument('--root_dir', type=str, default='/home/niell_lab/data/freely_moving_ephys/inception_loop/')
    parser.add_argument('--history_size', type=int, default=8)
    parser.add_argument('--split_frac', type=float, default=0.8)
    parser.add_argument('--input_channels', type=int, default=1) # 3
    parser.add_argument('--hidden_channels', type=int, default=2)
    parser.add_argument('--input_kern', type=int, default=7)
    parser.add_argument('--hidden_kern', type=int, default=7)# 27
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_resize', type=int, default=64)
    args = parser.parse_args()
    return args

def train_loop(dataloader, model, loss_fn, optimizer, exp_log):

    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y[:,-1,:].to(device))
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        exp_log.log({'train_loss': loss.item()})

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    

def test_loop(dataloader, model, loss_fn, exp_log):
    
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device); y = y[:,-1,:].to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= size
    correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")

    exp_log.log({'avg_test_loss': test_loss})

def main(args):

    input_path = os.path.join(args.root_dir, 'inputs')

    exp_log = Experiment(name='inceptionloop',
                    debug=False,
                    save_dir=args.root_dir,
                    autosave=True)

    checkpoint_path = os.path.join(args.root_dir,exp_log.name,f'version_{exp_log.version}','checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    configfile = os.path.join(args.root_dir,exp_log.name,f'version_{exp_log.version}','config.yaml')
    with open(configfile,'w') as file: 
        try:
            yaml.dump(args.__dict__, file)
        except yaml.YAMLError as exc:
            print(exc)
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((args.img_resize,args.img_resize)),
                                transforms.ToTensor(),
                                SetRange])

    dataset = WorldcamDataset3D(args.csv, args.history_size, input_path, transform=transform)

    num_units = len(eval(dataset.metadata['SR0'].iloc[0]))

    startind = dataset.metadata.index[0]; endind = dataset.metadata.index[-1]; splitind = int((endind - startind) * args.split_frac)
    
    training_data = Subset(dataset,torch.arange(startind,splitind))
    testing_data = Subset(dataset,torch.arange(splitind,endind))

    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, prefetch_factor=5)
    test_dataloader = DataLoader(testing_data, batch_size=args.batch_size, num_workers=args.num_workers)

    model = Stacked3dCore(args.input_channels, args.hidden_channels, args.input_kern, args.hidden_kern, num_units, spike_history_len=args.history_size, img_resize=args.img_resize)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    epochs = 100
    for t in range(epochs):
        savepath = os.path.join(checkpoint_path, 'spike_pred_epoch'+str(t)+'.pt')
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, exp_log)
        test_loop(test_dataloader, model, loss_fn, exp_log)
        if t % 10 == 0:
            torch.save({
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, savepath)

    print("Done!")

if __name__ == '__main__':
    args = get_args()
    main(args)
