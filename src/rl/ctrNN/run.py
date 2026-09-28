import torch
from pathlib import Path
import os
from loadenv import LoadEnv

env = LoadEnv()
if __name__ == "__main__":
    model = torch.load(env._resolve_path("CTRNN_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model.pt"))