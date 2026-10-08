import torch
from pathlib import Path
import os
from loadenv import LoadEnv
from jdh import JDHConfig
from jdh import JDH
from decoder_model import Decoder

from config import MAX_NEW_TOKENS
env_loader = LoadEnv()

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_float32_matmul_precision("high")

    model = JDH(JDHConfig())
    model_state_dict = torch.load(env_loader._resolve_path("JDH_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model_with_predictor.pt"))
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()} 
    model.load_state_dict(model_state_dict)
    model.to(device).compile()
    model.eval()

    decoder = Decoder(JDHConfig())
    decoder_state_dict = torch.load(env_loader._resolve_path("DECODER_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model_decoder.pt"))
    decoder_state_dict = {k.replace("module.", ""): v for k, v in decoder_state_dict.items()}
    decoder_state_dict = {k.replace("_orig_mod.", ""): v for k, v in decoder_state_dict.items()} 
    decoder.load_state_dict(decoder_state_dict)
    decoder.to(device).eval().compile()
