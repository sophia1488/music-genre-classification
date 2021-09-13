import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import json
from train import training, validating

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config/config.json', type=str)
    parser.add_argument('--cuda', default=0, type=int, help='Specify cuda number')
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    save_dir = f"result/{args.name}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(args.config) as f:
		config = json.load(f)
    
    writer = SummaryWriter(comment=params.dataset_name)

    print('Creating dataloader...')
    Trainset = AudioDataset(f"{config.data_dir}/train_128mel.pkl")
    train_loader = DataLoader(Trainset, shuffle=True, batch_size=config.train_batch_size, num_workers=config.num_workers)

    Validset = AudioDataset(f"{config.data_dir}/valid_128mel.pkl")
    val_loader = DataLoader(Validset, shuffle=True, batch_size=config.valid_batch_size, num_workers=config.num_workers)

    print('Loading model...')
    if config.model=="densenet":
        model = DenseNet(config.pretrained).to(device)
    elif config.model=="resnet":
        model = ResNet(config.pretrained).to(device)
    elif config.model=="inception":
        model = Inception(config.pretrained).to(device) 

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    #train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, i, scheduler)
    best_acc = 0.0

    for epoch in range(config['epochs']):
        train_loss, train_acc = training(model, device, train_loader, optimizer, loss_fn)
        valid_loss, valid_acc = validating(model, device, val_loader, loss_fn)

        print(f"Epoch {epoch} Train Loss:{train_loss} Valid Acc:{valid_acc}")

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
