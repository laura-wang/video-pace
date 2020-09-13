import os
import time
import argparse
import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models import r21d, r3d, c3d, s3d_g
from datasets.ucf101 import ucf101_pace_pretrain
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor
from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='gpu id')
    parser.add_argument('--height', type=int, default=256, help='resize height')
    parser.add_argument('--width', type=int, default=256, help='resize width')
    parser.add_argument('--clip_len', type=int, default=64, help='64, input clip length')
    parser.add_argument('--crop_sz', type=int, default=224, help='crop size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='32, batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers')
    parser.add_argument('--epoch', type=int, default=18, help='total epoch')
    parser.add_argument('--max_sr', type=int, default=4, help='largest sampling rate')
    parser.add_argument('--num_classes', type=int, default=4, help='num of classes')
    parser.add_argument('--max_save', type=int, default=3, help='max save epoch num')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/k400')
    parser.add_argument('--pf', type=int, default=1, help='print frequency')
    parser.add_argument('--model', type=str, default='s3d', help='s3d/r21d/r3d/c3d, pretrain model')
    parser.add_argument('--data_list', type=str, default='list/train_ucf101_split1.list', help='data list')
    parser.add_argument('--rgb_prefix', type=str, default='/home/user/dataset/ucf101_jpegs_256/', help='dataset dir')

    args = parser.parse_args()

    return args


def train(args):
    torch.backends.cudnn.benchmark = True

    exp_name = '{}_sr_{}_{}_lr_{}_len_{}_sz_{}'.format(args.dataset, args.max_sr, args.model, args.lr, args.clip_len, args.crop_sz)


    print(exp_name)

    pretrain_cks_path = os.path.join('pretrain_cks', exp_name)
    log_path = os.path.join('visual_logs', exp_name)

    if not os.path.exists(pretrain_cks_path):
        os.makedirs(pretrain_cks_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ## 1. dataset
    #data_list = '/home/user/code/ucf101_list_ab1/train_ucf101_num_frames.list'
    #rgb_prefix = '/home/user/dataset/ucf101_jpegs_256/'

    transforms_ = transforms.Compose(
        [ClipResize((args.height, args.width)),  # h x w
         RandomCrop(args.crop_sz),
         RandomHorizontalFlip(0.5)]
    )

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    color_jitter = transforms.RandomApply([color_jitter], p=0.8)

    train_dataset = ucf101_pace_pretrain(args.data_list, args.rgb_prefix, clip_len=args.clip_len, max_sr=args.max_sr,
                                   transforms_=transforms_, color_jitter_=color_jitter)

    print("len of training data:", len(train_dataset))
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)

    ## 2. init model
    if args.model == 'r21d':
        model = r21d.R2Plus1DNet(num_classes=args.num_classes)
    elif args.model == 'r3d':
        model = r3d.R3DNet(num_classes=args.num_classes)
    elif args.model == 'c3d':
        model = c3d.C3D(num_classes=args.num_classes)
    elif args.model == 's3d':
        model = s3d_g.S3D(num_classes=args.num_classes, space_to_depth=False)

    # 3. define loss and lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    # 4. multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    criterion.to(device)

    writer = SummaryWriter(log_dir=log_path)
    iterations = 1

    model.train()

    for epoch in range(args.epoch):
        start_time = time.time()

        for i, sample in enumerate(dataloader):
            rgb_clip, labels = sample
            rgb_clip = rgb_clip.to(device, dtype=torch.float)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgb_clip)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            accuracy = torch.sum(preds == labels.data).detach().cpu().numpy().astype(np.float)
            accuracy = accuracy / args.bs

            iterations += 1

            if i % args.pf == 0:
                writer.add_scalar('data/train_loss', loss, iterations)
                writer.add_scalar('data/Acc', accuracy, iterations)

                print("[Epoch{}/{}] Loss: {} Acc: {} Time {} ".format(
                    epoch + 1, i, loss, accuracy, time.time() - start_time))

            start_time = time.time()

        scheduler.step()
        model_saver(model, optimizer, epoch, args.max_save, pretrain_cks_path)

    writer.close()


def model_saver(net, optimizer, epoch, max_to_keep, model_save_path):
    tmp_dir = os.listdir(model_save_path)
    print(tmp_dir)
    tmp_dir.sort()
    if len(tmp_dir) >= max_to_keep:
        os.remove(os.path.join(model_save_path, tmp_dir[0]))

    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'opt_dict': optimizer.state_dict(),
    }, os.path.join(model_save_path, 'epoch-' + '{:02}'.format(epoch + 1) + '.pth.tar'))


if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train(args)
