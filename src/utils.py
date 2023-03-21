import random
import numpy as np
import torch
            

def set_seed(seed: int):
    """Function for setting the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Function for setting the device in PyTorch."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"
