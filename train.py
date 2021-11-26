#  Copyright (c) 2021 Mandar Gogate, All rights reserved.

from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import GridDataModule
from model import *
from utils.generic import str2bool


def main(args):
    checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch")

    datamodule = GridDataModule(batch_size=args.batch_size, add_channel_dim=True, a_only=args.a_only)

    audio_unet = build_audio_unet(filters=64, a_only=args.a_only, visual_feat_dim=1024)
    visualfeat_net = build_visualfeat_net(extract_feats=True) if not args.a_only else None
    model = IO_AVSE_DNN((visualfeat_net, audio_unet), args, datamodule.test_dataset)

    trainer = Trainer.from_argparse_args(args, default_root_dir=args.log_dir, callbacks=[checkpoint_callback])

    if args.tune:
        trainer.tune(model, datamodule)
    else:
        trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--a_only", type=str2bool, default=True)
    parser.add_argument("--tune", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00158)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--loss", type=str, default="l1", required=True)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
