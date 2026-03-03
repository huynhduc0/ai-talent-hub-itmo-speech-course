from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()
        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        # Do correct initialization of stft params below:
        # hop_length, n_mels, center, return_complex, onesided, normalize_stft, pad_mode, power
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power

        # Do correct initialization of mel fbanks params below:
        # f_min_hz, f_max_hz, norm_mel, mel_scale
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        # finish parameters initialization
        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        # To access attributes, use self.<parameter_name>
        return F.melscale_fbanks(
            n_freqs=(self.n_fft // 2) + 1 if self.onesided else self.n_fft,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz if self.f_max_hz is not None else float(self.samplerate // 2),
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale
        )

    def spectrogram(self, x):
        # x - is an input signal
        return torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window.to(x.device),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex
        )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        # <YOUR CODE GOES HERE>
        # get complex spectrogram
        spec = self.spectrogram(x)
        
        # compute power spectrum: |stft|^2
        if self.return_complex:
            spec_power = spec.abs() ** self.power
        else:
            # For pytorch versions < 1.8 where stft returns last dimension as 2 for real/imag
            spec_power = spec.pow(2).sum(-1)
            if self.power != 2.0:
                spec_power = spec_power.pow(self.power / 2.0)
                
        # Move mel_fbanks to the correct device
        mel_fbanks = self.mel_fbanks.to(x.device)
        
        # apply mel filterbanks: multiplication along frequencies
        # spec_power: (batch, n_freqs, time)
        # mel_fbanks: (n_freqs, n_mels)
        # result needs to be: (batch, n_mels, time)
        # we can do matrix multiplication: (batch, time, n_freqs) @ (n_freqs, n_mels) => (batch, time, n_mels) => transpose
        spec_power = spec_power.transpose(1, 2)
        mel_spec = torch.matmul(spec_power, mel_fbanks)
        mel_spec = mel_spec.transpose(1, 2)
        
        # Return log mel filterbanks matrix
        log_mel_spec = torch.log(mel_spec + 1e-6)
        
        return log_mel_spec
