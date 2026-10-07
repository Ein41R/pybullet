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
import loadenv
loadenv.LoadEnv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
) 

scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JDH_CONFIG = jdh.JDHConfig()
INPUT_FILE_PATH = loadenv._resolve_path("JDH_INPUT_FILE", Path(__file__).resolve().parent / "input.txt")
MODEL_PATH = loadenv._resolve_path("JDH_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model.pt")

def fetch_data():
    if not INPUT_FILE_PATH.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        INPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INPUT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)

def get_batch():
    data = torch.memmap.mmap(INPUT_FILE_PATH.open("r+b").fileno(), 0)
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    # for now batches of fixed length
    x = torch.stack([torch.tensor(data[i : i + BLOCK_SIZE], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i : i + BLOCK_SIZE + TARGET_SIZE], dtype=torch.long) for i in ix])
    x, y = x.to(device), y.to(device)


if __name__ == "__main__":
    if not PROJECT_ROOT.input.txt.exists():
        fetch_data()
    get_batch()
