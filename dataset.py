#  Copyright (c) 2021 Mandar Gogate, All rights reserved.

import logging
import math
import os
import random
from os.path import join

import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from config import GRID_IMAGES_ROOT_sq, GRID_ROOT, SEED, img_height, img_width, nb_channels, sampling_rate, stft_size, window_shift, window_size
from utils.data import get_images
from utils.generic import subsample_list


def get_transform():
    transform_list = [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


test_transform = get_transform()


class GridDataset(Dataset):
    def __init__(self, speakers, raw_data_root, images_root, shuffle=True, seed=SEED, audio_prefix="audio_16000", subsample=1, add_channel_dim=False, a_only=True, return_stft=False):
        self.return_stft = return_stft
        self.a_only = a_only
        self.images_root = images_root
        self.add_channel_dim = add_channel_dim
        self.speakers = speakers
        self.raw_data_root = raw_data_root
        self.audio_prefix = audio_prefix
        self.files_list = self.build_files_list
        self.rgb = True if nb_channels == 3 else False
        if shuffle:
            random.seed(SEED)
            random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
        logging.info("Found {} utterances".format(len(self.files_list)))
        self.data_count = len(self.files_list)
        self.batch_index = 0
        self.total_batches_seen = 0
        self.batch_input = {"noisy": None}
        self.index = 0
        self.max_len = len(self.files_list)
        self.max_cache = 0
        self.seed = seed
        self.window = "hann"
        self.fading = False

    @property
    def build_files_list(self):
        files_list = []
        for speaker in self.speakers:
            clean_root = join(self.raw_data_root, speaker, self.audio_prefix)
            for audio_file in os.listdir(clean_root):
                clean_file = join(clean_root, audio_file)
                file_id = audio_file.split(".")[0]
                files_list.append([speaker, file_id, clean_file])
        return files_list

    def __len__(self):
        if self.return_stft:
            return len(self.files_list)
        else:
            return len(self.files_list) * 2

    def __getitem__(self, idx):
        data = {}
        (speaker, file_id, clean_file), (_, _, noise_file) = random.sample(self.files_list, 2)
        if not self.a_only:
            images_root = join(self.images_root, speaker, file_id)
            data["lip_images"] = self.get_lip_images(images_root)
        if self.return_stft:
            data["noisy_audio_spec"], data["mask"], data["clean"], data["noisy_stft"] = self.get_audiofeat(clean_file, noise_file)
        else:
            data["noisy_audio_spec"], data["mask"] = self.get_audiofeat(clean_file, noise_file)
        return data

    def get_noisy_features(self, noisy):
        audio_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=self.window, center=True)
        if self.add_channel_dim:
            return np.abs(audio_stft).astype(np.float32)[np.newaxis, ...]
        else:
            return np.abs(audio_stft).astype(np.float32)

    def get_lip_images(self, images_root, rgb=False):
        lip_image = np.zeros((64, img_height, img_width)).astype(np.float32)
        try:
            img = get_images(images_root, rgb=rgb)
            if img is not None:
                img = img.astype(np.float32)
                img = img / 255
                mean = [0.5]
                std = [0.5]
                img = (img - mean) / std
                if lip_image.shape[0] <= img.shape[0]:
                    lip_image = img[:lip_image.shape[0]]
                else:
                    lip_image[:img.shape[0]] = img
        except Exception as e:
            print(e)
        return lip_image[np.newaxis, ...]

    def get_audiofeat(self, clean_file, noise_file):
        noise, _ = librosa.load(noise_file, sr=sampling_rate)
        clean, _ = librosa.load(clean_file, sr=sampling_rate)
        clean, noise = clean[:40900], noise[:40900]
        if noise.shape[0] > clean.shape[0]:
            clean = np.pad(clean, pad_width=[0, noise.shape[0] - clean.shape[0]], mode="constant")
        else:
            noise = np.pad(noise, pad_width=[0, clean.shape[0] - noise.shape[0]], mode="constant")
        noise_db = random.randint(0, 20)
        clean_power = np.linalg.norm(clean, 2)
        noise_power = np.linalg.norm(noise, 2)
        snr = math.exp(noise_db / 10)
        scale = snr * noise_power / clean_power
        noisy = (scale * clean + noise) / 2
        if self.return_stft:
            clean_audio = clean
            noisy_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=self.window, center=True)
            return self.get_noisy_features(noisy), self.get_noisy_features(
                clean), clean_audio[:-100], noisy_stft

        else:
            return self.get_noisy_features(noisy), self.get_noisy_features(clean)


class GridDataModule(LightningDataModule):
    def __init__(self, batch_size=16, add_channel_dim=False, a_only=False):
        super(GridDataModule, self).__init__()
        train_speakers_ids, val_speakers_ids, test_speakers_ids = [4, 7, 11, 16, 23, 24, 25, 29, 31, 33, 34, 3, 5, 6, 9, 10, 13, 14, 17, 19, 26, 27, 28], [1, 32, 2, 30], [18, 20, 22, 26]
        train_speakers = ["S{}".format(speaker) for speaker in train_speakers_ids]
        val_speakers = ["S{}".format(speaker) for speaker in val_speakers_ids]
        test_speakers = ["S{}".format(speaker) for speaker in test_speakers_ids]
        self.train_dataset = GridDataset(train_speakers, GRID_ROOT, GRID_IMAGES_ROOT_sq, add_channel_dim=add_channel_dim, a_only=a_only)
        self.val_dataset = GridDataset(val_speakers, GRID_ROOT, GRID_IMAGES_ROOT_sq, add_channel_dim=add_channel_dim, a_only=a_only, return_stft=True)
        self.test_dataset = GridDataset(test_speakers, GRID_ROOT, GRID_IMAGES_ROOT_sq, add_channel_dim=add_channel_dim, a_only=a_only, return_stft=True)
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


if __name__ == '__main__':
    mask = "IRM"
    dataset = GridDataset(speakers=["S3"], raw_data_root=GRID_ROOT, images_root=GRID_IMAGES_ROOT_sq, a_only=False, return_stft=True)
    for i in range(10):
        data = dataset[i]
        for k, v in data.items():
            print(k, v.shape, np.min(v), np.max(v), np.mean(v))
    print(dataset.files_list[:10])
