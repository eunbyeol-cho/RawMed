import numpy as np
import tqdm
import os
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from functools import partial
import torch
    
def get_first_sep_idx(event, value):
    """Return the separator index for the given event and value."""
    idx = np.where(event == value)[0]
    return idx[0] if idx.size else None

def extract_sep_idx_from_event(i_event, t_event):
    """Process an event and return the separator index."""
    if i_event[0] == 0:
        return 0
    else:
        i_sep_idx = get_first_sep_idx(i_event, 102)
        t_sep_idx = get_first_sep_idx(t_event, 6)
        if i_sep_idx is None and t_sep_idx is None:
            sep_idx = None
        elif i_sep_idx is not None and t_sep_idx is None:
            sep_idx = i_sep_idx
        elif i_sep_idx is None and t_sep_idx is not None:
            sep_idx = t_sep_idx
        else:
            sep_idx = min(i_sep_idx, t_sep_idx)
    return sep_idx