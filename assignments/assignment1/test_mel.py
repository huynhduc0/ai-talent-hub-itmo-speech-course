import torch
import torchaudio # check if module imported
from melbanks import LogMelFilterBanks

num_frames = 1 * 16000 # 1 second
signal = torch.randn(1, num_frames)

melspec = torchaudio.transforms.MelSpectrogram(
    hop_length=160,
    n_mels=80
)(signal)
logmelbanks = LogMelFilterBanks()(signal)

assert torch.log(melspec + 1e-6).shape == logmelbanks.shape, f"Shape mismatch: {torch.log(melspec + 1e-6).shape} vs {logmelbanks.shape}"
assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks, atol=1e-5), "Value mismatch"
print("Tests passed successfully")
