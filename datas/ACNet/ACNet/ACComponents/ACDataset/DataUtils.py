import json
import re
import pandas as pd


class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        self.mixed = True
        self.random_sample_negative = False
        self.random_sample_negative_seed = 8
        self.discard_extreme_imbalance = False  # Discard the subsets that are extremely imbalanced. Default False.
        self.pn_rate_threshold = 0.2           # The threshold of Pos/Neg for extremely imbalanced subsets. Default 0.1.
        self.discard_few_pos = True            # Discard the subsets that have only few positive samples. Default True.
        self.few_pos_threshold = 10            # The threshold to identify few positive
        self.large_thres = 20000
        self.medium_thres = 1000
        self.small_thres = 100




def SaveJson(Addr, object):
    with open(Addr, 'w') as f:
        json.dump(object,f)

def LoadJson(Addr):
    with open(Addr, 'r') as f:
        content = json.load(f)
    return content