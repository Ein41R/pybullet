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
INPUT_FILE_PATH = env_loader._resolve_path("JDH_INPUT_FILE", Path(__file__).resolve().parent / "input.txt")
MODEL_PATH = env_loader._resolve_path("JDH_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model.pt")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def fetch_data():
    if not INPUT_FILE_PATH.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        INPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INPUT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)

def get_batch(mode="encode"):
    data = np.memmap(INPUT_FILE_PATH, dtype=np.uint8, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    # for now batches of fixed length
    x = torch.stack([torch.from_numpy(data[i : i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i : i + BLOCK_SIZE + TARGET_SIZE].astype(np.int64)) for i in ix])
    
    if torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
        device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y

if __name__ == "__main__":
    fetch_data()

    decoder = Decoder(JDH_CONFIG).to(device)#.compile()
    src_encoder = jdh.JDH(JDH_CONFIG).to(device)
    ema_encoder = copy.deepcopy(src_encoder).to(device).eval()
    # [p.require_grad_(False) for p in ema_encoder.parameters()]

    src_encoder.compile()

    enc_opt = torch.optim.AdamW(src_encoder.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    dec_opt = torch.optim.AdamW(decoder.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    @torch.no_grad()
    def ema_update():
        for p_online, p_target in zip(src_encoder.parameters(), ema_encoder.parameters()):
            p_target.mul_(EMA_DECAY).add_(p_online.detach(), alpha=1 - EMA_DECAY)

    """
    STAGE 1:
    Train enc and dec only
    """
    dec_acc = 0
    x,_ = get_batch(mode="encode")
    for step in range(MAX_ITERS):
        enc_opt.zero_grad(set_to_none=True)
        dec_opt.zero_grad(set_to_none=True)
        
        with ctx: src_emb = src_encoder(x)
        with ctx, torch.no_grad(): tgt_emb = ema_encoder(x)
        
        dec_logits, dec_loss = decoder(src_emb.detach(), x)

        feature_loss = F.mse_loss(src_emb, tgt_emb)

        scaler.scale(dec_loss).backward()
        scaler.scale(feature_loss).backward()
        scaler.step(dec_opt)
        scaler.step(enc_opt)
        scaler.update()
        ema_update()

        if step % LOG_FREQ == 0:
            dec_acc = (dec_logits.argmax(dim=-1) == x).float().mean().item()
            feature_loss_value = feature_loss.detach().item()
            dec_loss_value = dec_loss.detach().item()
            print(
                f"step {step:>4d} | "
                f"feature loss {feature_loss_value:.3} | "
                f"decoder ce {dec_loss_value:.3} acc {dec_acc:.3} ppl {math.exp(min(dec_loss_value, 20)):.1} |"
            )
            dec_acc = 0

        print("STAGE 1: training completed, saving models...")

        #not saving ema since you could just use encoder for both
        torch.save(src_encoder.state_dict(), MODEL_PATH.parent / "JDH_model_encoder.pt")
        torch.save(decoder.state_dict(), MODEL_PATH.parent / "JDH_model_decoder.pt")

    """
    STAGE 2:
    Training the predictor
    """
#NVM do that in separate file
