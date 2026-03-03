import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from model import SpeechCNN, calculate_model_metrics
from melbanks import LogMelFilterBanks

class YesNoDataset(Dataset):
    def __init__(self, root, subset, n_mels=80, kwargs=None):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        self.dataset = SPEECHCOMMANDS(root, subset=subset, download=True)
        # Filter yes/no classes according to exercise constraint
        self.indices = []
        for i in range(len(self.dataset)):
            label = self.dataset[i][2]
            if label in ['yes', 'no']:
                self.indices.append(i)
        
        self.log_mel = LogMelFilterBanks(n_mels=n_mels, **(kwargs or {}))
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[self.indices[idx]]
        # PAD waveform to exactly 16000
        padding = 16000 - waveform.shape[1]
        if padding > 0:
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :16000]
            
        mel_spec = self.log_mel(waveform)
        # Binary target conversion
        target = 1 if label == 'yes' else 0
        return mel_spec.squeeze(0), target

def main():
    parser = argparse.ArgumentParser(description="Pytorch SPEECHCOMMANDS Classification")
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    # We should setup data
    import os
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_dataset = YesNoDataset(root_dir, 'training', n_mels=args.n_mels)
    val_dataset = YesNoDataset(root_dir, 'validation', n_mels=args.n_mels)
    test_dataset = YesNoDataset(root_dir, 'testing', n_mels=args.n_mels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeechCNN(in_channels=args.n_mels, num_classes=2, groups=args.groups).to(device)
    
    # Check dimensions for logs
    # Assume 1 second sequences (shape based on hop_length) -> hop_length=160 -> frames = 100 
    params, flops = calculate_model_metrics(model, input_size=(1, args.n_mels, 101))
    print(f"Model parameters: {params}")
    print(f"Model FLOPs: {flops}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                
        val_acc = val_correct / len(val_dataset)
        
        print(f"Epoch: {epoch+1:02d} | Train Loss: {train_loss:.4f} | Validation Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            
    print(f"Best Validation Acc: {best_acc:.4f}")
    
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
    test_acc = test_correct / len(test_dataset)
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    # Record to a output file to track metric parameters
    with open('training_log.csv', 'a') as f:
        f.write(f"{args.n_mels},{args.groups},{params},{flops},{test_acc},{train_loss},{epoch_time}\\n")
        
if __name__ == '__main__':
    main()
