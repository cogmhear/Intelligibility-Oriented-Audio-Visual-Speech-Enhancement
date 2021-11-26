import random

import librosa
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import window_shift, window_size
from loss import STOILoss
from utils.models.resnet import BasicBlock, ResNet
from utils.nn import TCN, conv_block, threeD_to_2D_tensor, unet_conv, unet_upconv, up_conv, \
    weights_init


class VisualFeatNet(nn.Module):
    def __init__(self, tcn_options, hidden_dim=256, num_classes=500,
                 relu_type='prelu', extract_feats=False):
        super(VisualFeatNet, self).__init__()
        self.extract_feats = extract_feats
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.tcn = TCN(input_size=self.backend_out,
                       num_channels=[hidden_dim * len(tcn_options['kernel_size']) * tcn_options['width_mult']] *
                       tcn_options['num_layers'],
                       num_classes=num_classes,
                       tcn_options=tcn_options,
                       dropout=tcn_options['dropout'],
                       relu_type=relu_type,
                       dwpw=tcn_options['dwpw'],
                       )

    def forward(self, x, lengths):
        B, C, T, H, W = x.size()
        if type(lengths) == int:
            lengths = [lengths] * B
        x = self.frontend3D(x)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        return self.tcn(x, lengths, B, self.extract_feats)


class UNet(nn.Module):
    def __init__(self, filters=64, input_nc=2, output_nc=2, av_embedding=1024, a_only=True, activation='Sigmoid'):
        super(UNet, self).__init__()
        self.a_only = a_only
        self.conv1 = unet_conv(input_nc, filters)
        self.conv2 = unet_conv(filters, filters * 2)
        self.conv3 = conv_block(filters * 2, filters * 4)
        self.conv4 = conv_block(filters * 4, filters * 8)
        self.conv5 = conv_block(filters * 8, filters * 8)
        self.frequency_pool = nn.MaxPool2d([2, 1])
        if not a_only:
            self.upconv1 = up_conv(av_embedding, filters * 8)
        else:
            self.upconv1 = up_conv(filters * 8, filters * 8)
        self.upconv2 = up_conv(filters * 16, filters * 8)
        self.upconv3 = up_conv(filters * 16, filters * 4)
        self.upconv4 = up_conv(filters * 8, filters * 2)
        self.upconv5 = unet_upconv(filters * 4, filters)
        self.upconv6 = unet_upconv(filters * 2, output_nc, True)
        if activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'ReLu':
            self.activation = nn.ReLU()

    def forward(self, mix_spec, visual_feat=None):
        conv1feat = self.conv1(mix_spec)
        conv2feat = self.conv2(conv1feat)
        conv3feat = self.conv3(conv2feat)
        conv3feat = self.frequency_pool(conv3feat)
        conv4feat = self.conv4(conv3feat)
        conv4feat = self.frequency_pool(conv4feat)
        conv5feat = self.conv5(conv4feat)
        conv5feat = self.frequency_pool(conv5feat)
        if self.a_only:
            av_feat = conv5feat
        else:
            upsample_visuals = F.interpolate(visual_feat, (8, 64))
            av_feat = torch.cat((conv5feat, upsample_visuals), dim=1)
        upconv1feat = self.upconv1(av_feat)
        upconv2feat = self.upconv3(torch.cat((upconv1feat, conv4feat), dim=1))
        upconv3feat = self.upconv4(torch.cat((upconv2feat, conv3feat), dim=1))
        upconv4feat = self.upconv5(torch.cat((upconv3feat, conv2feat), dim=1))
        predicted_mask = self.upconv6(torch.cat((upconv4feat, conv1feat), dim=1))
        pred_mask = self.activation(predicted_mask)
        return torch.mul(pred_mask, mix_spec)


def build_audio_unet(filters=64, input_nc=1, output_nc=1, visual_feat_dim=1280, weights='', a_only=False, activation="Sigmoid"):
    net = UNet(filters, input_nc, output_nc, visual_feat_dim, a_only=a_only, activation=activation)
    net.apply(weights_init)

    if len(weights) > 0:
        print('Loading weights for UNet')
        net.load_state_dict(torch.load(weights))
    return net


def build_visualfeat_net(weights='', extract_feats=True):
    net = VisualFeatNet(tcn_options=dict(num_layers=4, kernel_size=[3], dropout=0.2, dwpw=False, width_mult=2),
                        relu_type="prelu",
                        extract_feats=extract_feats)
    if len(weights) > 0:
        print('Loading weights for lipreading stream')
        net.load_state_dict(torch.load(weights))
    return net


class IO_AVSE_DNN(LightningModule):
    def __init__(self, nets, args, val_dataset=None):
        super(IO_AVSE_DNN, self).__init__()
        self.args = args
        self.net_visualfeat, self.net_audio_unet = nets
        loss = args.loss
        if loss.lower() == "l1":
            self.loss = F.l1_loss
        elif loss.lower() == "l2":
            self.loss = F.mse_loss
        elif loss.lower() == "stoi":
            self.loss = STOILoss()
        else:
            raise NotImplementedError("{} is currently unavailable as loss function. Select one of l1, l2 and stoi".format(loss))
        self.lr = args.lr
        self.val_dataset = val_dataset

    def forward(self, input):
        noisy_audio_spec = input['noisy_audio_spec']
        if self.args.a_only:
            pred_mask = self.net_audio_unet(noisy_audio_spec)
        else:
            lip_images = input['lip_images']
            visual_feat = self.net_visualfeat(lip_images.float(), 64)
            pred_mask = self.net_audio_unet(noisy_audio_spec, visual_feat)
        return pred_mask

    def training_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        if self.val_dataset is not None:
            with torch.no_grad():
                tensorboard = self.logger.experiment
                rand_int = random.randint(0, len(self.val_dataset))
                data = self.val_dataset[rand_int]
                inputs = {"noisy_audio_spec": torch.from_numpy(data["noisy_audio_spec"][np.newaxis, ...]).to(self.device)}
                if not self.args.a_only:
                    inputs["lip_images"] = torch.from_numpy(data["lip_images"][np.newaxis, ...]).to(self.device)
                pred_mag = self(inputs)[0][0].cpu().numpy()
                noisy_phase = np.angle(data["noisy_stft"])
                estimated = pred_mag * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                estimated_audio = librosa.istft(estimated, win_length=window_size, hop_length=window_shift, window="hann")
                noisy = librosa.istft(data["noisy_stft"], win_length=window_size, hop_length=window_shift, window="hann")
                tensorboard.add_audio("{}/clean".format(self.current_epoch), data["clean"][np.newaxis, ...], sample_rate=16000)
                tensorboard.add_audio("{}/noisy".format(self.current_epoch), noisy[np.newaxis, ...], sample_rate=16000)
                tensorboard.add_audio("{}/enhanced".format(self.current_epoch), estimated_audio[np.newaxis, ...], sample_rate=16000)

    def cal_loss(self, batch_inp):
        mask = batch_inp["mask"]
        pred_mask = self(batch_inp)
        loss = self.loss(pred_mask, mask)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.66, patience=2),
                "monitor": "val_loss_epoch",
            },
        }


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    net = build_audio_unet(a_only=True)
    test_audio_data = torch.rand((1, 1, 256, 256))
    pred_mask = net(test_audio_data).detach().numpy()
    print("Audio-only UNet", pred_mask.shape)
    print(np.min(pred_mask), np.max(pred_mask))
    test_visual_data = torch.rand([1, 1, 64, 88, 88])
    visual_net = build_visualfeat_net(extract_feats=True)
    visual_feat = visual_net(test_visual_data, 64)
    print("Visual feat", visual_feat.shape)
    net = build_audio_unet(filters=64, a_only=False, visual_feat_dim=1024)
    print("Audio-visual UNet", net(test_audio_data, visual_feat).shape)
