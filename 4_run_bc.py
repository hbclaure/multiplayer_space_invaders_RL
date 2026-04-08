import torch
import numpy as np
import pandas as pd
import os


bc_folder = '/Users/houstonclaure/Desktop/multiplayer_space_invaders_RL/analysis_results/behavior_cloning_splits'
#upload dataset
train_data = torch.load(os.path.join(bc_folder, "eval.pt"), weights_only=False)
test_data = torch.load(os.path.join(bc_folder, "test.pt"), weights_only=False)
eval_data = torch.load(os.path.join(bc_folder, "train.pt"), weights_only=False)



