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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE_PATH = env_loader._resolve_path("JDH_INPUT_FILE", Path(__file__).resolve().parent / "input.txt")


def fetch_data():
    if not INPUT_FILE_PATH.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        INPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INPUT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)

def get_batch():
    data = np.memmap(INPUT_FILE_PATH, dtype=np.uint8, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([
        torch.from_numpy(data[i:i + BLOCK_SIZE - 32].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64))
        for i in ix
    ])
    if torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
        device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y

if __name__ == "__main__":
    dec_nn = Decoder(JDHConfig())
    dec_nn_state_dict = torch.load(env_loader._resolve_path("DECODER_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model_decoder.pt"))
    dec_nn_state_dict = {k.replace("module.", ""): v for k, v in dec_nn_state_dict.items()}
    dec_nn_state_dict = {k.replace("_orig_mod.", ""): v for k, v in dec_nn_state_dict.items()} 
    dec_nn.load_state_dict(dec_nn_state_dict)
    dec_nn.to(device).eval().compile()

    src_enc_nn = JDH(JDHConfig())
    src_enc_nn_state_dict = torch.load(env_loader._resolve_path("JDH_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model_encoder.pt"))
    src_enc_nn_state_dict = {k.replace("module.", ""): v for k, v in src_enc_nn_state_dict.items()}
    src_enc_nn_state_dict = {k.replace("_orig_mod.", ""): v for k, v in src_enc_nn_state_dict.items()} 
    src_enc_nn.load_state_dict(src_enc_nn_state_dict)
    src_enc_nn.to(device).compile()

    tgt_enc_nn = copy.deepcopy(src_enc_nn).to(device).eval()
    tgt_enc_nn.compile()

    optimizer = torch.optim.AdamW(src_enc_nn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    print(f"Using device: {device} with dtype {dtype}")


    for step in range(MAX_ITERS):
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch()
        
        with ctx, torch.no_grad():
            src_emb = tgt_enc_nn(x)
            tgt_emb = tgt_enc_nn(y) #y has 1 token more

        with ctx:
            tgt_pred = src_enc_nn.predict(src_emb)

        tgt_emb = tgt_emb[:, -1, :]
        tgt_pred = tgt_pred[:, -1:, :]

        loss = F.mse_loss(tgt_pred, tgt_emb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % LOG_FREQ == 0:
            print(f"Step [{step:4d}/{MAX_ITERS}]: Loss = {loss.item():.6f}")

    print("STAGE 2:Training completed. Saving the model...")

    torch.save(src_enc_nn.state_dict(), env_loader._resolve_path("MODEL_PATH", Path(__file__).resolve().parent / "parameters") / "JDH_model_with_predictor.pt")
    print(f"Model saved to {env_loader._resolve_path('MODEL_PATH', Path(__file__).resolve().parent / 'parameters') / 'JDH_model_with_predictor.pt'}")