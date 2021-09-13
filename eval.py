from sklearn.metrics import confusion_matrix
from utils import save_cm_fig
import argparse
import torch
from model import DenseNet
from dataset import AudioDataset
from torch.utils.data import DataLoader
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, type=int, help='Specify cuda number')
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_dir", default="../Data", type=str)
    args = parser.parse_args()
    return args


def Eval(model, device, test_loader):
    CM = np.zeros((10,10))
    train_loss, total_acc, total_cnt = 0, 0, 0
    model.eval()
    
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)

            outputs = model(inputs)

            _, pred_label = torch.max(outputs.data, 1)
            total_acc += torch.sum((pred_label == target).float()).item()
            total_cnt += target.size(0)

            target = target.cpu()
            pred_label = pred_label.cpu()
            cm = confusion_matrix(target, pred_label, labels=[i for i in range(10)])
            CM = np.add(CM, cm)
        
    return total_acc / total_cnt, CM


def main():
    args = get_args()
    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print('Creating dataloader...')
    Testset = AudioDataset(f"{args.data_dir}/test_128mel.pkl")
    test_loader = DataLoader(Testset, batch_size=16)

    
    print('Loading model...')
    model = DenseNet()
    checkpoint = torch.load(f"{args.save_dir}/model_best.ckpt", map_location='cpu')
    model.load_state_dict(checkpoint['state_dict']) # already is ['state_dict']
    model = model.to(device)
    model.eval()
    
    # Predicted result
    print('Predicting...')
    test_acc, cm = Eval(model, device, test_loader)
    print('test acc:', test_acc)
    target_names=['rock', 'country', 'disco', 'blues', 'metal', 'hiphop', 'pop', 'jazz', 'classical', 'reggae']
    save_cm_fig(cm, classes=target_names, normalize=True, title="genre_classification", save_dir=args.save_dir)

if __name__ == '__main__':
    main()
