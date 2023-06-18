import os
import json
from pathlib import Path

import torch

from utils import data_utils
from models import cbm
from plots import plots


# change this to the correct model dir, everything else should be taken care of
load_dir = "saved_models/doctor_nurse_full_cbm_2023_06_05_11_27"
device = "cuda"

with open(os.path.join(load_dir, "args.txt"), "r") as f:
    args = json.load(f)
dataset = args["dataset"]
target_model, target_preprocess = data_utils.get_target_model(args["backbone"], device)
model = cbm.load_cbm(load_dir, device)