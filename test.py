#  Copyright (c) 2021 Mandar Gogate, All rights reserved.

from argparse import ArgumentParser
from os import makedirs
from os.path import isfile, join

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from config import sampling_rate, window_shift, window_size
from dataset import GridDataModule
from model import IO_AVSE_DNN, build_visualfeat_net, build_audio_unet
from utils.generic import str2bool


def main(args):
    clean_root = join(args.save_root, "clean")
    noisy_root = join(args.save_root, "noisy")
    enhanced_root = join(args.save_root, args.model_uid)
    makedirs(args.save_root, exist_ok=True)
    makedirs(clean_root, exist_ok=True)
    makedirs(noisy_root, exist_ok=True)
    makedirs(enhanced_root, exist_ok=True)

    datamodule = GridDataModule(add_channel_dim=True, a_only=args.a_only)
    test_dataset = datamodule.test_dataset

    audio_unet = build_audio_unet(filters=64, a_only=args.a_only, visual_feat_dim=1024)
    visual_net = build_visualfeat_net(extract_feats=True) if not args.a_only else None

    if args.ckpt_path.endswith("ckpt") and isfile(args.ckpt_path):
        model = IO_AVSE_DNN.load_from_checkpoint(args.ckpt_path, nets=(visual_net, audio_unet), args=args)
    else:
        raise FileNotFoundError("Cannot load model weights: {}".format(args.ckpt_path))

    model.eval()
    model.to("cuda:0")
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            filename = f"{str(i).zfill(5)}.wav"
            clean_path = join(clean_root, filename)
            noisy_path = join(noisy_root, filename)
            enhanced_path = join(enhanced_root, filename)

            data = test_dataset[i]
            if not isfile(clean_path):
                sf.write(clean_path, data["clean"], samplerate=sampling_rate)
            if not isfile(noisy_path):
                noisy = librosa.istft(data["noisy_stft"], win_length=window_size, hop_length=window_shift, window="hann")
                sf.write(noisy_path, noisy, samplerate=sampling_rate)
            if not isfile(enhanced_path):
                inputs = {"noisy_audio_spec": torch.from_numpy(data["noisy_audio_spec"][np.newaxis, ...]).to(model.device)}
                if not args.a_only:
                    inputs["lip_images"] = torch.from_numpy(data["lip_images"][np.newaxis, ...]).to(model.device)
                pred_mag = model(inputs)[0][0].cpu().numpy()
                noisy_phase = np.angle(data["noisy_stft"])
                estimated = pred_mag * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                estimated_audio = librosa.istft(estimated, win_length=window_size, hop_length=window_shift, window="hann")
                sf.write(enhanced_path, estimated_audio, samplerate=sampling_rate)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--a_only", type=str2bool, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--model_uid", type=str, required=True)
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--lr", type=float, default=0.00158)
    args = parser.parse_args()
    main(args)
