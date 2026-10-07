import os
from contextlib import nullcontext
from pathlib import Path

import jdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import math

from config import (
    BLOCK_SIZE,
    TARGET_SIZE,
    BATCH_SIZE,
    MAX_ITERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LOG_FREQ,
    EMA_DECAY,
    L_RELEVANCE,
    L_DECODABILITY,
    BUFFER
)

from jdh import JDH
from jdh import JDHConfig
from decoder_model import Decoder
from loadenv import LoadEnv

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_loader = LoadEnv()
    env_loader._load_env_file()

    dec_nn = Decoder(JDHConfig())
    dec_nn_state_dict = torch.load(env_loader._resolve_path("DECODER_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model_decoder.pt"))
    dec_nn_state_dict = {k.replace("module.", ""): v for k, v in dec_nn_state_dict.items()}
    dec_nn_state_dict = {k.replace("_orig_mod.", ""): v for k, v in dec_nn_state_dict.items()} 
    dec_nn.load_state_dict(dec_nn_state_dict)
    dec_nn.to(device).eval().compile()

    src_enc_nn = JDH(JDHConfig())
    src_enc_nn_state_dict = torch.load(env_loader._resolve_path("DECODER_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model_encoder.pt"))
    src_enc_nn_state_dict = {k.replace("module.", ""): v for k, v in src_enc_nn_state_dict.items()}
    src_enc_nn_state_dict = {k.replace("_orig_mod.", ""): v for k, v in src_enc_nn_state_dict.items()} 
    src_enc_nn.load_state_dict(src_enc_nn_state_dict)
    src_enc_nn.to(device).eval().compile()

    tgt_enc_nn = copy.deepcopy(src_enc_nn).to(device).eval().compile()

