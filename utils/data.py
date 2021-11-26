#  Copyright (c) 2018 Mandar Gogate, All rights reserved.

import logging

import cv2
import numpy as np

from config import img_height, img_width
from utils.generic import get_files

GRID_TRANSCRIPTION = {
    0: dict(b="bin", l="lay", p="place", s="set"),
    1: dict(b="blue", g="green", r="red", w="white"),
    2: dict(a="at", b="by", i="in", w="with"),
    4: dict(z="zero"),
    5: dict(a="again", n="now", p="please", s="soon")
}


def load_npz(path: str):
    return np.load(path)['arr_0']


def preprocess_feature(data: np.ndarray, TVF: int, repetition: int) -> np.ndarray:
    padding = [[repetition * (TVF - 1), 0]] + [[0, 0]] * (len(data.shape) - 1)
    padded_data = np.pad(data, padding, "constant", constant_values=0)
    input_length = padded_data.shape[0]
    appended_input = [np.expand_dims(padded_data[0:input_length - repetition * (TVF - 1)], axis=1)]
    for i in range(1, TVF):
        appended_input.append(np.expand_dims(padded_data[repetition * i:input_length - repetition * (TVF - 1 - i)], axis=1))
    return np.concatenate(tuple(appended_input), axis=1)


def get_upsampled_images(images_root: str, image_format, upsample_by: int = 3, dct_features: bool = False, rgb: bool = False):
    combined_images = get_images(images_root, rgb)
    return np.repeat(combined_images, upsample_by, axis=0)


def get_images(images_root: str, rgb: bool = False):
    try:
        images = sorted(get_files(images_root))
        images_list = []
        for image in images:
            try:
                img = cv2.imread(image, int(rgb))
                img = cv2.resize(img, (img_height, img_width))
            except Exception as e:
                logging.debug(e)
                img = None
            if img is None:
                if int(rgb) == 1:
                    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                else:
                    img = np.zeros((img_height, img_width), dtype=np.uint8)
            img = img[np.newaxis, ...]
            images_list += [img]
        combined_images = np.concatenate(images_list, axis=0)
        return combined_images
    except Exception as e:
        logging.debug("Error {} in {}".format(e, images_root))


def align_frames(interpolated_images: np.ndarray, ibm: np.ndarray) -> np.ndarray:
    if interpolated_images.shape[0] > ibm.shape[0]:
        interpolated_images = interpolated_images[:ibm.shape[0]]
    else:
        last_image = interpolated_images[-1]
        last_image = np.expand_dims(last_image, axis=0)
        for _ in range(ibm.shape[0] - interpolated_images.shape[0]):
            interpolated_images = np.vstack((interpolated_images, last_image))
    return interpolated_images


def get_concatenated(ibm_list, dtype):
    out_list = []
    for ibm_path in ibm_list:
        ibm = np.array(load_npz(ibm_path)[0], dtype=dtype)
        out_list.append(ibm[np.newaxis, ...])
    output_mask = np.concatenate(tuple(out_list), axis=0)
    return output_mask


