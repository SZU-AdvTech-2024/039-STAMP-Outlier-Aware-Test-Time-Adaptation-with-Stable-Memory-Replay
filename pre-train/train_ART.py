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



# Define the UGD loss function
def unknown_activation_loss(logits, labels, num_classes):
    """
    Computes the Unknown Activation Loss (L_UA).
    logits: Predicted logits from the model [batch_size, num_classes + 1].
    labels: Ground truth labels [batch_size].
    num_classes: Number of known classes.
    """
    f_u = logits[:, num_classes - 1 ]  # Logit for unknown class
    
    # f_k = torch.cat([
    #     logits[:, :num_classes -1  ].scatter_(1, labels.view(-1, 1), -1e12), # Set true class to -inf 注意就地操作_导致logits_copy浅拷贝失效
    #     logits[:, num_classes -1 ].unsqueeze(1)
    # ], dim=1)

    f_k = logits.clone()
    f_k.scatter_(1, labels.view(-1,1),-1e12)

    ua_loss = -torch.log(torch.exp(f_u) / torch.exp(f_k).sum(dim=1))
    ua_loss_mean = ua_loss.mean()
    # print(f"---------------The ua_loss_mean is {ua_loss_mean} -----------------")
    return ua_loss_mean

def smoothed_ce_loss(logits, labels, num_classes, tau=1.2, lambda_=0.05):
    """
    Computes the Smoothed Cross-Entropy Loss (L_SCE).
    logits: Predicted logits from the model [batch_size, num_classes + 1].
    labels: Ground truth labels [batch_size].
    num_classes: Number of known classes.
    tau: Temperature scaling parameter.
    lambda_: Weight for L2 regularization on logits.
    """
    scaled_logits = logits / tau
    ce_loss = F.cross_entropy(scaled_logits[:, :num_classes], labels)
    # l2_penalty = lambda_ * (logits ** 2).sum(dim=1).mean()
    l2_penalty = lambda_ * (logits ** 2).mean()
    return ce_loss + l2_penalty

def ugd_loss(logits, labels, num_classes):
    """
    Combines L_UA and L_SCE into the final UGD loss.
    """
    logits_ua = logits.clone()
    logits_sce = logits.clone()
    ua_loss = unknown_activation_loss(logits_ua, labels, num_classes)
    sce_loss = smoothed_ce_loss(logits_sce, labels, num_classes)
    print(f"------- ua_loss is {ua_loss:.2f} - sce_loss is {sce_loss:.2f} ------------")
    return ua_loss + args.loss_balance* sce_loss

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
    parser.add_argument('--batch_size', type=int, default=512, help="batch_size") # 256->512
    parser.add_argument('--worker', type=int, default=12, help="number of workers") # 线程 4->12
    parser.add_argument('--dset', type=str, default='cifar10', # 数据集CIFAR-10
                        choices=['VISDA-C', 'office', 'officehome', 'office-caltech', 'domainnet126', 'cifar10',
                                 'cifar100'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--net', type=str, default='ResNet18_11', # 'ResNet50_10' -> ResNet18_11
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
    parser.add_argument('--loss_balance', type=float, default=1.5)
    args = parser.parse_args()

    train_dataset, train_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker, split='train',
                                               transforms=transform_train)
    test_dataset, test_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker, split='test',
                                             transforms=transform_test)

    model_name = args.net
    ckpt_path = os.path.join(args.ckpt, 'models', model_name + '_UA.pt')

    model = load_model(model_name).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=2e-4)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epoch)
    elif args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.steps, gamma=args.gamma)

    max_acc = 0

    # Early stop
    early_stop_patience = 10
    no_improve_epochs = 0

    print("\n========== Training Start ==========")
    for epoch in range(args.max_epoch):
        print(f"\n[Epoch {epoch + 1}/{args.max_epoch}] | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        epoch_loss = 0.0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)):
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()

            logits = model(inputs)
            loss = ugd_loss(logits, labels, num_classes=model.num_classes)
            # print(f"The caculated ugd_loss is {loss}")
            loss.backward()

            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
