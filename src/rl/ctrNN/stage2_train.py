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

from decoder_model import Decoder
from loadenv import LoadEnv

env_loader = LoadEnv()
env_loader._load_env_file()

