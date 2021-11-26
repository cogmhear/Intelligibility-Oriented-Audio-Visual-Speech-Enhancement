# Towards Intelligibility-Oriented Audio-Visual Speech Enhancement

[![arXiv](https://img.shields.io/badge/arXiv-2111.09642-green.svg)](https://arxiv.org/abs/2111.09642) 

This paper has been accepted for publication in the Clarity Workshop on Machine Learning Challenges for Hearing Aids 

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* pytorch-lightning
* tensorboard >= 1.14

You can install all requirements using 

```bash
pip install -r requirements.txt
```

## Usage
Update config.py with your dataset path
Try `python train.py --log_dir ./logs --a_only False --gpu 1 --max_epochs 15 --loss stoi` to run code.

## License
The code in this repository is CC BY 4.0 licensed, as found in the LICENSE file.

## Acknowledgements
Parts of the code are adapted from [Co-Separation](https://github.com/rhgao/co-separation), [Lipreading using Temporal Convolutional Networks](https://github.
com/mpc001/Lipreading_using_Temporal_Convolutional_Networks) and [VisualVoice](https://github.com/facebookresearch/VisualVoice).
