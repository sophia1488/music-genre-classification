import torch
from torch.utils.data import DataLoader
import librosa
import argparse
import pickle as pkl
from preprocess import extract_spectrogram
from dataset import AudioDataset
from model import DenseNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, type=int, help='Specify cuda number')
    parser.add_argument("--save_dir", type=str, required=True, help='the directory with the best model checkpoint')
    parser.add_argument("--audio", type=str, required=True)
    args = parser.parse_args()
    return args


def Predict(model, device, test_loader):
    model.eval()
    
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)

            outputs = model(inputs)

            _, pred_label = torch.max(outputs.data, 1)

            pred_label = pred_label[0].item()
        
    return pred_label 


def main():
    args = get_args()
    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    audio = args.audio
    
    print(f'Loading audio file {audio}')
    clip, sr = librosa.load(audio)
    values = extract_spectrogram(sr, [], clip, 10)     # randomly set class_idx

    audio = audio.split('/')[-1]
    with open(f"{args.save_dir}/{audio.split('.')[0]}.pkl","wb") as handler:
        pkl.dump(values, handler, protocol=pkl.HIGHEST_PROTOCOL)
    
    print('Creating dataloader...')
    Testset = AudioDataset(f"{args.save_dir}/{audio.split('.')[0]}.pkl")
    test_loader = DataLoader(Testset, batch_size=16)

    
    print('Loading model...')
    model = DenseNet()
    checkpoint = torch.load(f"{args.save_dir}/model_best.ckpt", map_location='cpu')
    model.load_state_dict(checkpoint['state_dict']) # already is ['state_dict']
    model = model.to(device)
    model.eval()
    
    # Predicted result
    MAP = {0:'blues', 1:'pop', 2:'jazz', 3:'country', 4:'classical', 5:'reggae', 6:'rock', 7:'disco', 8:'metal', 9:'hiphop'}
    print('Predicting...')
    pred_label = Predict(model, device, test_loader)
    label = MAP[pred_label]
    print(f'{audio} belongs to class "{label}"')


if __name__ == '__main__':
    main()
