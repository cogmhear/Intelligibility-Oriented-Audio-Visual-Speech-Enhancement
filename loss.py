import librosa
import numpy as np
import torch
from pystoi.stoi import BETA, DYN_RANGE, MINFREQ, N, NUMBAND
from pystoi.utils import thirdoct
from torch import nn
from torch.nn.functional import unfold

from config import stft_size, window_shift, window_size

eps = torch.finfo(torch.float32).eps


class STOILoss(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 extended: bool = True):
        super().__init__()
        # Independant from FS
        self.sample_rate = sample_rate
        self.extended = extended
        self.intel_frames = N
        self.beta = BETA
        self.dyn_range = DYN_RANGE
        self.eps = torch.finfo(torch.float32).eps
        self.win_len = window_size
        self.nfft = stft_size
        win = torch.from_numpy(np.hanning(self.win_len + 2)[1:-1]).float()
        self.win = nn.Parameter(win, requires_grad=False)
        obm_mat = thirdoct(sample_rate, self.nfft, NUMBAND, MINFREQ)[0]
        self.OBM = nn.Parameter(torch.from_numpy(obm_mat).float(),
                                requires_grad=False)

    def forward(self, est_targets: torch.Tensor,
                targets: torch.Tensor, ) -> torch.Tensor:
        if targets.shape != est_targets.shape:
            raise RuntimeError('targets and est_targets should have '
                               'the same shape, found {} and '
                               '{}'.format(targets.shape, est_targets.shape))
        x_spec = targets.reshape(-1, 256, 256, 1)
        y_spec = est_targets.reshape(-1, 256, 256, 1)
        # Apply OB matrix to the spectrograms as in Eq. (1)
        x_tob = torch.matmul(self.OBM, torch.norm(x_spec, 2, -1) ** 2 + self.eps).pow(0.5)
        y_tob = torch.matmul(self.OBM, torch.norm(y_spec, 2, -1) ** 2 + self.eps).pow(0.5)
        # Perform N-frame segmentation --> (batch, 15, N, n_chunks)
        batch = targets.shape[0]
        x_seg = unfold(x_tob.unsqueeze(2),
                       kernel_size=(1, self.intel_frames),
                       stride=(1, 1)).view(batch, x_tob.shape[1], N, -1)
        y_seg = unfold(y_tob.unsqueeze(2),
                       kernel_size=(1, self.intel_frames),
                       stride=(1, 1)).view(batch, y_tob.shape[1], N, -1)
        # Compute mask if use_vad
        mask_f = None
        if self.extended:
            # Normalize rows and columns of intermediate intelligibility frames
            x_n = self.rowcol_norm(x_seg, mask=mask_f)
            y_n = self.rowcol_norm(y_seg, mask=mask_f)
            corr_comp = x_n * y_n
            correction = self.intel_frames * x_n.shape[-1]
        else:
            # Find normalization constants and normalize
            norm_const = (masked_norm(x_seg, dim=2, keepdim=True, mask=mask_f) /
                          (masked_norm(y_seg, dim=2, keepdim=True, mask=mask_f)
                           + self.eps))
            y_seg_normed = y_seg * norm_const
            # Clip as described in [1]
            clip_val = 10 ** (-self.beta / 20)
            y_prim = torch.min(y_seg_normed, x_seg * (1 + clip_val))
            # Mean/var normalize vectors
            y_prim = meanvar_norm(y_prim, dim=2, mask=mask_f)
            x_seg = meanvar_norm(x_seg, dim=2, mask=mask_f)
            # Matrix with entries summing to sum of correlations of vectors
            corr_comp = y_prim * x_seg
            # J, M as in [1], eq.6
            correction = x_seg.shape[1] * x_seg.shape[-1]
        # Compute average (E)STOI w. or w/o VAD.
        sum_over = list(range(1, x_seg.ndim))  # Keep batch dim
        # Return -(E)STOI to optimize for
        return torch.mean(1 - torch.sum(corr_comp, dim=sum_over) / correction)

    @staticmethod
    def rowcol_norm(x, mask=None):
        """ Mean/variance normalize axis 2 and 1 of input vector"""
        for dim in [2, 1]:
            x = meanvar_norm(x, mask=mask, dim=dim)
        return x


def meanvar_norm(x, mask=None, dim=-1):
    x = x - masked_mean(x, dim=dim, mask=mask, keepdim=True)
    x = x / (masked_norm(x, p=2, dim=dim, keepdim=True, mask=mask) + eps)
    return x


def masked_mean(x, dim=-1, mask=None, keepdim=False):
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    return (x * mask).sum(dim=dim, keepdim=keepdim) / (
            mask.sum(dim=dim, keepdim=keepdim) + eps
    )


def masked_norm(x, p=2, dim=-1, mask=None, keepdim=False):
    if mask is None:
        return torch.norm(x, p=p, dim=dim, keepdim=keepdim)
    return torch.norm(x * mask, p=p, dim=dim, keepdim=keepdim)


if __name__ == '__main__':
    y, sr = librosa.load(librosa.ex('trumpet'), sr=16000)
    noisy_speech = torch.from_numpy(y[:40900])
    clean_speech = noisy_speech
    sample_rate = 16_000
    loss_func = STOILoss(sample_rate=sample_rate)
    S = torch.from_numpy(np.abs(librosa.stft(y[:40900], win_length=window_size, n_fft=stft_size, hop_length=window_shift, window="hann", center=True)))
    S_new = torch.cat((S.unsqueeze(0), S.unsqueeze(0)), dim=0)
    loss_batch = loss_func(S.unsqueeze(0), S.unsqueeze(0))
    print(loss_batch)
