import librosa
import argparse
import pickle as pkl
from preprocess import extract_spectrogram

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, type=int, help='Specify cuda number')
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_dir", default="../Data", type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    audio = args.audio
    
    clip, sr = librosa.load(audio)
    values = extract_spectrogram(sampling_rate, v, clip, 10)     # randomly set class_idx

    with open(f"{audio.split('.')[0]}.pkl","wb") as handler:
        pkl.dump(values, handler, protocol=pkl.HIGHEST_PROTOCOL)
    
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
