# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np
from PIL import Image

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from src.utils.distributed import (
    init_distributed,
    AllReduce
)

from src.helper import load_target_encoding_checkpoint, init_encoder
from src.transforms import make_inference_transforms
from pickle import dump


# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint']
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        logger.info("Using CPU")
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        logger.info("Using Cuda")

    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    patch_size = args['mask']['patch_size']
    # --

    # -- LOGGING
    folder = args['logging']['folder']
    
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    assert r_file is not None
    load_path = os.path.join(folder, r_file)

    # -- init model
    target_encoder = init_encoder(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name)

    target_encoder = load_target_encoding_checkpoint(
        device=device,
        r_path=load_path,
        target_encoder=target_encoder,
    )
    target_encoder.to(device)

    # -- load data
    image_folder = os.path.join(root_path, image_folder)
    images = [Image.open(os.path.join(image_folder, file)) for file in os.listdir(image_folder) if file.endswith(".png") or file.endswith(".jpg")]
    labels = [file for file in os.listdir(image_folder) if file.endswith(".png") or file.endswith(".jpg")]


    # -- transform data
    transforms = make_inference_transforms(
        crop_size=crop_size,
        crop_scale=(1,1),
    )

    input_tensor = [transforms(img) for img in images]
    input_tensor = torch.stack(input_tensor).to(device)

    resulting_tensors = []

    # -- get encodings from data
    for i in range(0, input_tensor.shape[0], batch_size):
        with torch.no_grad():
            logger.info("%d/%d" % (i, input_tensor.shape[0]))
            encoded_rep = target_encoder(input_tensor[i:min(i+batch_size, input_tensor.shape[0])])
            resulting_tensors.append(torch.mean(encoded_rep, 1).cpu())
    resulting_tensors = torch.concat(resulting_tensors).numpy().tolist()

    # Save encodings
    labeled_encodings = {label: tensor for label, tensor in zip(labels, resulting_tensors)}
    dump(labeled_encodings, open(os.path.join(image_folder, "jepa_encodings.pkl"), "wb"))
    

if __name__ == "__main__":
    main()
