import io
import json
import os
import random

import numpy as np
import torch


def startupCheck(file):
    """Creates an empty json file if a file with the provided path does not exist 
    or is not readable."""
    if not (os.path.isfile(file) and os.access(file, os.R_OK)):
        print ("Creating json file...")
        with io.open(file, 'w') as db_file:
            db_file.write(json.dumps({}))
            

def set_seed(seed):
    """Function for setting the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)