import argparse
import os
import ssl
import time

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS

import soundfile as sf

# Bypass macOS SSL cert issues for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

# Use soundfile backend to avoid torchcodec dependency
def _sf_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath)
    tensor = torch.from_numpy(data).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor, sr

torchaudio.load = _sf_load

from model import SpeechCNN, calculate_model_metrics
from melbanks import LogMelFilterBanks

_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_CLASSES = {'yes': 1, 'no': 0}


class YesNoDataset(Dataset):
    def __init__(self, root: str, subset: str, n_mels: int = 80):
        self.dataset = SPEECHCOMMANDS(root, subset=subset, download=True)
        self.indices = [
            i for i, path in enumerate(self.dataset._walker)
            if os.path.basename(os.path.dirname(path)) in _CLASSES
        ]
        self.log_mel = LogMelFilterBanks(n_mels=n_mels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        waveform, _, label, *_ = self.dataset[self.indices[idx]]
        waveform = waveform[:, :16000]
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        return self.log_mel(waveform).squeeze(0), _CLASSES[label]


def run(args):
    train_ds = YesNoDataset(_DATA_ROOT, 'training',   args.n_mels)
    val_ds   = YesNoDataset(_DATA_ROOT, 'validation', args.n_mels)
    test_ds  = YesNoDataset(_DATA_ROOT, 'testing',    args.n_mels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SpeechCNN(in_channels=args.n_mels, groups=args.groups).to(device)
    params, flops = calculate_model_metrics(model, input_size=(1, args.n_mels, 101))
    print(f"[n_mels={args.n_mels}, groups={args.groups}] params={params:,}  FLOPs={flops:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    last_train_loss = 0.0
    last_epoch_time = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        last_train_loss = total_loss
        last_epoch_time = time.time() - t0

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += model(x).argmax(1).eq(y).sum().item()
        val_acc = correct / len(val_ds)
        print(f"  Epoch {epoch:02d} | loss={total_loss:.4f} | val_acc={val_acc:.4f} | time={last_epoch_time:.1f}s")

        # per-epoch log for plotting
        epoch_log = os.path.join(os.path.dirname(__file__), 'epoch_log.csv')
        with open(epoch_log, 'a') as f:
            f.write(f"{args.n_mels},{args.groups},{epoch},{total_loss:.6f},{val_acc:.6f},{last_epoch_time:.2f}\n")

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
    test_acc = correct / len(test_ds)
    print(f"  Test accuracy: {test_acc:.4f}")

    log_path = os.path.join(os.path.dirname(__file__), 'training_log.csv')
    with open(log_path, 'a') as f:
        f.write(f"{args.n_mels},{args.groups},{params},{flops},{test_acc:.4f},{last_train_loss:.4f},{last_epoch_time:.2f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_mels',     type=int, default=80)
    parser.add_argument('--groups',     type=int, default=1)
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
