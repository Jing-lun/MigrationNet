import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import pytorch_msssim
from metric import mean_iou, intersection_and_union, accuracy

from eval import eval_net
from migrationnet import MigrationNet
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from utils.dataset_mat import BasicDataset,BasicDataset_mat
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image


dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(args.img, args.mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                # if global_step % (len(dataset) // (1 * batch_size)) == 0:
        val_score = eval_net(net, val_loader, device, n_val)
        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, global_step)

        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))
            writer.add_scalar('Dice/test', val_score, global_step)

        # writer.add_images('images', imgs, global_step)
        if net.n_classes == 1:
            writer.add_images('masks/true', true_masks, global_step)
            writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
            temp_img = (torch.sigmoid(masks_pred) > 0.5)
            temp_img = temp_img.squeeze(0)
            temp_img = temp_img.cpu()
            temp_img = temp_img.to(torch.float)
            out_fn = '/home/jinglun/Data/migration/iros2020/predict_train_test/' + str(global_step) + '.png'
            save_image(temp_img, out_fn)

        if epoch % 20 == 0:
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the MigrationNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-x', '--img', default='/home/jinglun/Data/DATA/crackData/training/image/',
                        type=str, metavar='PATH', help='path to img dataset')
    parser.add_argument('-x1', '--img1', default='/home/jinglun/Data/DATA/crackData/training/image/',
                        type=str, metavar='PATH', help='path to img dataset')
    parser.add_argument('-x2', '--img2', default='/home/jinglun/Data/DATA/crackData/training/image/',
                        type=str, metavar='PATH', help='path to img dataset')
    parser.add_argument('-x3', '--img3', default='/home/jinglun/Data/DATA/crackData/training/image/',
                        type=str, metavar='PATH', help='path to img dataset')
    parser.add_argument('-y', '--mask', default='/home/jinglun/Data/DATA/crackData/training/bw_image/',
                        type=str, metavar='PATH', help='path to mask dataset')
    parser.add_argument('-c', '--channel', metavar='C', type=int, default=3,
                        help='Number of channels', dest='channels')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = MigrationNet(n_channels=256, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
