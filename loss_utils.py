import torch
from preprocess import TorchSignalToFrames
from scipy import linalg
import numpy as np
import scipy

class mse_loss(object):
    def __call__(self, outputs, labels, loss_mask):
        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        loss = torch.sum((masked_outputs - masked_labels)**2.0) / torch.sum(loss_mask)
        return loss


class stftm_loss(object):
    def __init__(self, frame_size=512, frame_shift=256, loss_type='mae'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        #self.device = device
        self.frame = TorchSignalToFrames(frame_size=self.frame_size,
                                         frame_shift=self.frame_shift)
        D = linalg.dft(frame_size)
        W = np.hamming(self.frame_size)
        DR = np.real(D)
        DI = np.imag(D)
        self.DR = torch.from_numpy(DR).float().cuda()  # to(self.device)
        self.DR = self.DR.contiguous().transpose(0, 1)
        self.DI = torch.from_numpy(DI).float().cuda()  # to(self.device)
        self.DI = self.DI.contiguous().transpose(0, 1)
        self.W = torch.from_numpy(W).float().cuda()  # to(self.device)

    def __call__(self, outputs, labels, loss_mask):
        outputs = self.frame(outputs)
        labels = self.frame(labels)
        loss_mask = self.frame(loss_mask)
        outputs = self.get_stftm(outputs)
        labels = self.get_stftm(labels)

        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        if self.loss_type == 'mse':
            loss = torch.sum((masked_outputs - masked_labels)**2) / torch.sum(loss_mask)
        elif self.loss_type == 'mae':
            loss = torch.sum(torch.abs(masked_outputs - masked_labels)) / torch.sum(loss_mask)

        return loss

    def get_stftm(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        stftm = torch.abs(stft_R) + torch.abs(stft_I)
        return stftm

class reg_loss(object):
    def __call__(self, fg1, g2, g1fx, g2fx):
        return torch.mean((fg1-g2-g1fx+g2fx)**2)