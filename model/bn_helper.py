import torch
import functools

import torch.distributed as dist

def init_distributed_mode():
    rank = 1
    world_size = 2
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set the current device to the current rank

#init_distributed_mode()

BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d #SyncBatchNorm
relu_inplace = True
