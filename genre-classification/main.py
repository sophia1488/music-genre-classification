import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import json
from train import training, validating
import utils
import os
from dataset import AudioDataset
from torch.utils.data import DataLoader
from model import DenseNet

def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not valid boolean string')
    return s == 'True'


def get_args():
    parser = argparse.ArgumentParser()
    # path #
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--data_dir", default="../Data", type=str)

    # parameter #
    parser.add_argument("--cuda", default=0, type=int, help='Specify cuda number')
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--train_batch", default=8, type=int)
    parser.add_argument("--valid_batch", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=int)
    parser.add_argument("--weight_decay", default=1e-3, type=int)

    # model #
    parser.add_argument("--model", default="densenet", type=str)
    parser.add_argument("--pretrained", default=True, type=boolean_string)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    save_dir = f"result/{args.name}"
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(comment="GTZAN")

    print('Creating dataloader...')
    Trainset = AudioDataset(f"{args.data_dir}/train_128mel.pkl")
    train_loader = DataLoader(Trainset, shuffle=True, batch_size=args.train_batch, num_workers=args.num_workers)

    Validset = AudioDataset(f"{args.data_dir}/valid_128mel.pkl")
    val_loader = DataLoader(Validset, shuffle=True, batch_size=args.valid_batch, num_workers=args.num_workers)

    print('Loading model...')
    if args.model=="densenet":
        model = DenseNet(args.pretrained).to(device)
    elif args.model=="resnet":
        model = ResNet(args.pretrained).to(device)
    elif args.model=="inception":
        model = Inception(args.pretrained).to(device) 

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    #train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, i, scheduler)
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = training(model, device, train_loader, optimizer, loss_fn)
        valid_loss, valid_acc = validating(model, device, val_loader, loss_fn)

        print(f"Epoch {epoch} Train Loss:{train_loss:.3f} Train Acc:{train_acc:.3f} Valid Loss:{valid_loss:.3f} Valid Acc:{valid_acc:.3f}")

        is_best = (valid_acc > best_acc)
        if is_best: best_acc = valid_acc
        
        scheduler.step()
        # save_checkpoint(state, is_best, split, checkpoint):
        utils.save_checkpoint({"epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    'best_acc': best_acc,
                    "optimizer": optimizer.state_dict()}, 
                    is_best, 
                    "{}".format(save_dir))
        writer.add_scalar("train loss", train_loss, epoch)
        writer.add_scalar("valid accuracy", valid_acc, epoch)

        with open(os.path.join(save_dir, 'log.log'), 'a') as outfile:
            outfile.write('Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n'.format(
                epoch+1, train_loss, train_acc, valid_loss, valid_acc))
    writer.close()


if __name__ == '__main__':
    main()
