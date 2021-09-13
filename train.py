from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn

def training(model, device, data_loader, optimizer, loss_fn):
    model.train()
    train_loss, total_acc, total_cnt = 0, 0, 0

    pbar = tqdm(data_loader, disable = False)
    for data in pbar:
        pbar.set_description("Training batch")
        inputs = data[0].to(device)         # [batch, 3, 128, 1500] 
        target = data[1].squeeze(1).to(device)    # ([1, ...,]), shape: [batch]

        outputs = model(inputs)

        loss = loss_fn(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
         _, pred_label = torch.max(y_hat, dim=1)
        acc = torch.sum((pred_label == y).float()).item()
        total_acc += acc
        total_cnt += target.size(0) 
        
    return train_loss / len(data_loader), total_acc / total_cnt

def validating(model, device, test_loader, loss_fn):
    train_loss, total_acc, total_cnt = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)

            outputs = model(inputs)
                  loss = loss_fn(outputs, target)
                  train_loss += loss.item()

            _, pred_label = torch.max(outputs.data, 1)
                  total_acc += torch.sum((pred_label == y).float()).item()
            total_cnt += target.size(0)

    return train_loss / len(data_loader), total_acc / total_cnt
