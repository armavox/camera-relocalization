import argparse
import os
import time
from datetime import datetime as dt

import torch
import torch.optim as opt

from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import models, transforms
from PIL import Image
from model import PoseNet, GoogleNet, Loss
from data import *
import utils
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch cam location regresor')
parser.add_argument('--model', type=str, default='PoseNet', 
                    choices=['GoogleNet', 'PoseNet'])
parser.add_argument('--phase', type=str, default='train', 
                    choices=['train', 'test', 'infer'])
parser.add_argument('--epochs', '-e', metavar='NUM_EPOCHS', default=5,
                    type=int, help='number of epochs to train')
parser.add_argument('--batch-size', '-b', metavar='BATCH_SIZE', default=16,
                    type=int, help='train batch size')
parser.add_argument('--learning-rate', metavar='LEARNING_RATE', default=0.0001,
                    type=float, help='learning rate parameter for optimizer')
parser.add_argument('--small-dataset', action='store_true',
                    help='Use small dataset for experiments')
parser.add_argument('--csv-file', '-c', metavar='CSV_FILE_PATH', default='camera_relocalization_sample_dataset/info.csv',
                    type=str, help='path to csv file')
parser.add_argument('--img-folder', '-i', metavar='IMG_FOLDER_PATH', default='camera_relocalization_sample_dataset/images/',
                    type=str, help='path to folder with images')
parser.add_argument('--save-dir', metavar='CHECKPOINT_FOLDER_PATH', default='checkpoints/',
                    type=str, help='path to folder with checkpoints')
parser.add_argument('--save-freq', '-f', metavar='N', default=5,
                    type=int, help='Save checkpoint every N epochs')
parser.add_argument('--loss-beta', '-l', metavar='LOSS_BETA', default=750,
                    type=int, help='value of beta const in loss function')
parser.add_argument('--plot-loss', '-p', action='store_true',
                    help='Whether show plot of loss during trainng')
parser.add_argument('--no-cuda', action='store_true',
                    help='whether to disable GPU')
parser.add_argument('--resume', action='store_true',
                    help='Resume to last checkpoint and continue training')
parser.add_argument('--sample-img-path', metavar='IMG_PATH', default='',
                    type=str, help='Path to image for inference')


def main():

    global args
    args = parser.parse_args()

    if not args.no_cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Architecture
    inceptionV3 = models.inception_v3(pretrained=True)
    model_name = args.model
    if model_name == 'PoseNet':
        model = PoseNet(inceptionV3)
    elif model_name == 'GoogleNet':
        model = GoogleNet(inceptionV3)

    phase = args.phase
    if phase in ['train', 'test']:
        utils.to_cuda(model, device)

    # Dataset
    transform = transforms.Compose([
            ConvertPILMode(mode='RGB'),
            transforms.Resize((299, 299)),  # pretrained Inception net input is (3, 299, 299)
            transforms.ToTensor()
        ])

    if phase in ['train', 'test']:
        
        IMG_PATH = args.img_folder
        CSV_PATH = args.csv_file
        dataset = CameraDataset(CSV_PATH, IMG_PATH, transform=transform)
        
        ### SMALL DATASET FOR EXPERIMENTS ###
        if args.small_dataset:
            small_ds_inds = np.random.choice(range(len(dataset)), size=int(len(dataset)*0.1),
                        replace=False)
            small_dataset = Subset(dataset, small_ds_inds)
            dataset = small_dataset

        # normally it'll be better to generate and save random indices for every train session 
        # to file and pull them out on the following test phase, but to not to clutter up repo
        # we'll just state seed at this stage
        np.random.seed(0)  # TODO: saving indices for test phase
        train_inds, val_inds, test_inds = train_val_holdout_split(dataset, ratios=[0.8,0.1,0.1])  # TODO: custom ratios
        train_sampler = SubsetRandomSampler(train_inds)
        val_sampler = SubsetRandomSampler(val_inds)
        test_sampler = Subset(dataset, test_inds)

        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
        test_loader = DataLoader(test_sampler)

        # loss and optimizer
        loss_func = Loss(device, beta=args.loss_beta)

    # Checkpoints
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.resume and phase in ['train', 'test']:
        chk_list = sorted([f for f in os.listdir(save_dir) if model_name in f])
        try:
            checkpoint_file = os.path.join(save_dir, chk_list[-1])
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['state_dict'])

        except Exception:
            print('No checkpoint to resume for this model')
            return

        print('Resume checkpoint:', chk_list[-1])
    print('*' * 60)

    # Training, testing or inferring
    if phase == 'train':
        fit(model, args.epochs, loss_func, train_loader, val_loader, 
            device, save_dir)
    elif phase =='test':
        test_loss = test(model, loss_func, test_loader, device)
        print(f"Test loss is: {test_loss:.4f}")
    elif phase == 'infer':
        image_name = args.sample_img_path
        if not image_name:
            print('Provide sample image path for inference')
            return
        image = Image.open(image_name)
        result = infer(image, model, model_name, transform, save_dir=save_dir)
        return result
    else:
        raise KeyError('Unimplemented phase or wrong phase name')
        return


def fit(net, epochs, loss_func, train_loader, val_loader, device, save_dir):
    start_time = time.time()
    print(f'Start training {args.model} during {epochs} epochs:')
    if args.small_dataset:
        print('Training on small dataset')
    optim = opt.Adam(net.parameters(), lr=args.learning_rate)
    train_sum_loss = []
    val_sum_loss = []

    for epoch in range(epochs):
        train_loss = 0
        net.train()
        for i, batch in enumerate(train_loader):
            optim.zero_grad()

            img, coords = batch['image'].to(device), batch['coords'].to(device)
            pos_pred, ori_pred = net(img)
            pos_target, ori_target = coords[:, :3], coords[:, 3:] 
            loss = loss_func(pos_pred, ori_pred, pos_target, ori_target)
            loss.backward()
            optim.step()
            train_loss += loss
        train_loss /= len(train_loader)
        if epoch % 1 != 0:
            print('Epoch: %04d Train loss: %.4f' % (epoch, train_loss.item()))

        if epoch % 1 == 0:
            net.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in val_loader:
                    img, coords = batch['image'].to(device), batch['coords'].to(device)
                    pos_pred, ori_pred = net(img)
                    pos_target, ori_target = coords[:, :3], coords[:, 3:] 
                    val_loss += loss_func(pos_pred, ori_pred, pos_target, ori_target)
                val_loss /= len(val_loader)

                train_sum_loss.append(train_loss.item())
                val_sum_loss.append(val_loss.item())

            print('Epoch: %04d Train loss: %.4f | Val loss: %.4f' % (epoch, train_loss.item(), val_loss.item()))

        if epoch % args.save_freq == 0:            
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
                
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args},
                os.path.join(save_dir, '%s_epoch_%03d.ckpt' % (args.model, epoch)))

    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Finished in {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    if args.plot_loss:
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(train_sum_loss[1:])), train_sum_loss[1:], label='train')
        plt.plot(range(len(val_sum_loss[1:])), val_sum_loss[1:], label='valid')
        plt.title(f'{model_name} loss in {len(train_sum_loss[1:])} epochs')
        plt.legend(loc='upper right')
        now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'figs/{model_name}-{now}.png')

    return train_sum_loss, val_sum_loss


def test(net, loss_func, test_loader, device):
    net.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in test_loader:
            img, coords = batch['image'].to(device), batch['coords'].to(device)
            pos_pred, ori_pred = net(img)
            pos_target, ori_target = coords[:, :3], coords[:, 3:] 
            test_loss += loss_func(pos_pred, ori_pred, pos_target, ori_target)
        test_loss /= len(test_loader)
    return test_loss.item()


def infer(image, net, model_name, transform, save_dir=None):
    net.eval()
    with torch.no_grad():
        
        if save_dir:
            chk_list = sorted([f for f in os.listdir(save_dir) if model_name in f])
            try:
                checkpoint_file = os.path.join(save_dir, chk_list[-1])
                checkpoint = torch.load(checkpoint_file)
                net.load_state_dict(checkpoint['state_dict'])

            except Exception:
                print('WARNING. Using not pretrained model.')
                
        sample = transform(image).unsqueeze(0)
        pos, orient = net(sample)
        pos = pos.view(-1).detach().numpy() 
        orient = orient.view(-1).detach().numpy()
        print(f'POS_X:{pos[0]:.2f}, POS_Y:{pos[1]:.2f}, POS_Z:{pos[2]:.2f}, Q_W:{orient[0]:.2f}, Q_X{orient[1]:.2f}, Q_Y:{orient[2]:.2f}, Q_Z:{orient[3]:.2f}')
    return pos, orient


if __name__ == '__main__':

    main()
