import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

def subsample4(wav):  
    # This function only works for k = 4 as of now.
    k = 4
    channels, dim= np.shape(wav) 

    dim1 = dim // k - 192  # 192 is used to correct the size of the sampled data, you can change it
    wav1, wav2 = np.zeros([channels, dim1]), np.zeros([channels, dim1])
    #print("wav1:", wav1.shape)
    #print("wav2:", wav2.shape)

    wav_cpu = wav.cpu()
    for channel in range(channels):
        for i in range(dim1):
            i1 = i * k
            num = np.random.choice([0, 1, 2, 3])
            if num == 0:
                wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1], wav_cpu[channel, i1+1]
            elif num == 1:
                wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1+1], wav_cpu[channel, i1+2]
            elif num == 2:
                wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1+2], wav_cpu[channel, i1+3]
            else:
                wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1+3], wav_cpu[channel, i1]   

    return torch.from_numpy(wav1).cuda(), torch.from_numpy(wav2).cuda()

def subsample2(wav):  
    # This function only works for k = 2 as of now.
    k = 2
    channels, dim= np.shape(wav) 

    dim1 = dim // k -128     # 128 is used to correct the size of the sampled data, you can change it
    wav1, wav2 = np.zeros([channels, dim1]), np.zeros([channels, dim1])   # [2, 1, 32640]
    #print("wav1:", wav1.shape)
    #print("wav2:", wav2.shape)

    wav_cpu = wav.cpu()
    for channel in range(channels):
        for i in range(dim1):
            i1 = i * k
            num = np.random.choice([0, 1])
            if num == 0:
                wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1], wav_cpu[channel, i1+1]
            elif num == 1:
                wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1+1], wav_cpu[channel, i1]

    return torch.from_numpy(wav1).cuda(), torch.from_numpy(wav2).cuda()

class SpeechDataset(Dataset):
    def __init__(self, noisy_files, clean_files, n_fft=N_FFT, hop_length=HOP_LENGTH):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)

        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.len_ = len(self.noisy_files)

        # fixed len
        self.max_len = 65280

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform

    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]

        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)

        return output

    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])   #list[n]
        x_noisy = self.load_sample(self.noisy_files[index])

        # padding/cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)

        # compute inputs and targets (g1x and g2x are subsampled form x_noisy)
        g1_wav, g2_wav = subsample2(x_noisy)
        g1_wav, g2_wav = g1_wav.type(torch.FloatTensor),g2_wav.type(torch.FloatTensor)
        
        # STFT
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        g1_stft = torch.stft(input=g1_wav, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        
        return x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft
