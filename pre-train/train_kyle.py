import argparse
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from src.data.data import load_dataset
from src.models import load_model
from src.utils import loss
from src.utils.loss import CrossEntropyLabelSmooth

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import time
from datetime import datetime
from tqdm import tqdm 


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])



if __name__ == "__main__":
    torch.backends.cudnn.enable = True
    parser = argparse.ArgumentParser(description='train_source_os')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='9', help="device id to run") # 默认使用最后一个gpu
    parser.add_argument('--s', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=200, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    parser.add_argument('--worker', type=int, default=12, help="number of workers") # 线程 4->12
    parser.add_argument('--dset', type=str, default='cifar10', # 数据集CIFAR-10
                        choices=['VISDA-C', 'office', 'officehome', 'office-caltech', 'domainnet126', 'cifar10',
                                 'cifar100'])
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--net', type=str, default='ResNet18_10', # 'ResNet50_10' -> ResNet18_10
                        help="vgg16, ResNet50_10, resnet101, vit, WideResNet_8,ResNet18_8,ResNet18_10")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='/home/ubuntu/stamp_ln/pre-train') # 原来是 '../ckpt/models' 这个没用上
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--data_dir', type=str, default='/mnt/d/stamp_lib/datasets') # 数据集路径
    parser.add_argument('--ckpt', type=str, default='/home/ubuntu/stamp_ln/pre-train') # 训练完后数据集保存路径 
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['cosine', 'multistep'])
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--steps', type=list, default=[60, 120, 160])
    args = parser.parse_args()

    train_dataset, train_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker, split='train',
                                               transforms=transform_train)
    test_dataset, test_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker, split='test',
                                             transforms=transform_test)

    model_name = args.net
    ckpt_path = os.path.join(args.ckpt, 'models', model_name + '.pt')

    model = load_model(model_name).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epoch)
    elif args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.steps, gamma=args.gamma)

    CELOSS = nn.CrossEntropyLoss()
    model.train()
    max_acc = 0

    # Early stop
    early_stop_patience = 10
    no_improve_epochs =0


    print("\n========== Training Start ==========")
    for epoch in range(args.max_epoch):
        print(f"\n[Epoch {epoch + 1}/{args.max_epoch}] | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        epoch_loss = 0.0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)):
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()

            loss = CELOSS(model(inputs), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 打印每 100 个 batch 的训练信息
            if (i + 1) % 100 == 0:
                print(f"  [Batch {i + 1}/{len(train_loader)}] Loss: {loss.item():.4f} | Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch + 1} Summary: Average Loss: {epoch_loss / len(train_loader):.4f} | Time: {epoch_time:.2f}s")

        # 测试阶段
        model.eval()
        acc = 0
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            acc += (torch.max(outputs, 1)[1] == labels).float().sum().item()
        acc = acc / len(test_loader.dataset)

        print(f"  Validation Accuracy: {acc:.4f}")
        scheduler.step()

        # 保存模型
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Model Saved at Epoch {epoch + 1} with Accuracy: {acc:.4f}")
            no_improve_epochs = 0 # Reset counter for upbreaking
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping at Epoch {epoch +1}, Best Validation Accuracy:{max_acc:.4f}")
            break     

        model.train()

    print("\n========== Training Finished ==========")
    print(f"Best Accuracy: {max_acc:.4f}")