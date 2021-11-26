#  Copyright (c) 2021 Mandar Gogate, All rights reserved.
import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime
from itertools import repeat
from os import makedirs
from os.path import isdir, isfile, join
from pathlib import Path

import numpy as np


def load_json(json_fp: str):
    with open(json_fp, 'r') as f:
        json_content = json.load(f)
    return json_content


def subsample_list(inp_list: list, sample_rate: float):
    random.shuffle(inp_list)
    return [inp_list[i] for i in range(int(len(inp_list) * sample_rate))]


def tempdir() -> str:
    return tempfile.gettempdir()


def ensure_exists(path: str):
    makedirs(path, exist_ok=True)


def multicore_processing(func, parameters: list, processes=None):
    from multiprocessing import Pool
    pool = Pool(processes=processes)
    result = pool.map(func, parameters)
    pool.close()
    pool.join()
    return result


def config_logging(level=logging.INFO):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(level)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    return logger


def get_files(files_root: str) -> list:
    assert isdir(files_root)
    return [
        join(files_root, file)
        for file in sorted(os.listdir(files_root))
    ]


def shuffle_arr(a, b):
    if a.shape[0] != b.shape[0]: raise RuntimeError("Both arrays should have the same elements in the first axis")
    p = np.random.permutation(len(a))
    return a[p], b[p]


def shuffle_lists(a: list, b: list, seed: int = None):
    c = list(zip(a, b))
    if seed is not None:
        random.seed(seed)
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def get_utc_time():
    return datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def subsample_list(inp_list: list, sample_rate: float):
    random.shuffle(inp_list)
    return [inp_list[i] for i in range(int(len(inp_list) * sample_rate))]


def save_dict(path: str, dict_obj: dict):
    with open(path, "w") as f:
        json.dump(dict_obj, f, indent=4, sort_keys=True)


class DisablePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def check_repo():
    from git import Repo
    repo = Repo(os.getcwd())
    assert not repo.is_dirty(), "Please commit the changes and then run the code"


def save_json(json_data, json_path, overwrite=True):
    if isfile(json_path) and not overwrite:
        raise Exception("JSON path: {} already exists".format(json_path))
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4, sort_keys=True)


def execute(command):
    subprocess.call(command, shell=True, stdout=None)


def inf_loop(data_loader):
    """ wrapper function for endless data loader. """
    for loader in repeat(data_loader):
        yield from loader

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def multicore_processing(func, parameters: list, processes=None):
    from multiprocessing import Pool
    pool = Pool(processes=processes)
    result = pool.map(func, parameters)
    pool.close()
    pool.join()
    return result
