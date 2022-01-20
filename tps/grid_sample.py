"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# encoding: utf-8

import torch.nn.functional as F
from torch.autograd import Variable

def grid_sample(input, grid, canvas = None, padding_mode='zeros', mode='bilinear'):
    g_sample = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if canvas is not None:
        source_region = Variable(input.data.new(input.size()).fill_(1))
        target_region = F.grid_sample(source_region, grid)
        padded_output = g_sample * target_region + canvas * (1 - target_region)
        return padded_output
    else:
        return g_sample
