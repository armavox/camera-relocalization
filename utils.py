import torch
import torch.nn as nn


def to_cuda(model, device):
    if torch.cuda.device_count() > 1:
        print("Using CUDA")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = model.to(device)
