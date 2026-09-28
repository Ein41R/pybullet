import torch
from pathlib import Path
import os
from loadenv import LoadEnv
from jdh import JDHConfig
from jdh import JDH
from decoder_model import Decoder

env = LoadEnv()
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = JDH(JDHConfig())
    model.to(device)
    model_state_dict = torch.load(env._resolve_path("MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model.pt"))
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}  # Remove 'module.' prefix if present
    model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}  # Remove '_orog_mod.' prefix if present
    
    model.load_state_dict(model_state_dict)
    model.eval()

    decoder = Decoder(JDHConfig())
    decoder_state_dict = torch.load(env._resolve_path("DECODER_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model_decoder.pt"))
    decoder_state_dict = {k.replace("module.", ""): v for k, v in decoder_state_dict.items()}  # Remove 'module.' prefix if present
    decoder_state_dict = {k.replace("_orig_mod.", ""): v for k, v in decoder_state_dict.items()}  # Remove '_orog_mod.' prefix if present
    decoder.load_state_dict(decoder_state_dict)
    decoder.to(device)
    decoder.eval()
    while True:
        user_prompt = input("\033[36mEnter a prompt (or 'exit' to quit): \033[0m")
        tmp = user_prompt
        if user_prompt.lower() == "exit":
            break
        prompt = torch.tensor(
            bytearray(user_prompt, "utf-8"), dtype=torch.long, device=device
        ).unsqueeze(0)
        x = model.forward(prompt)
        y = model.predict(x)
        out = decoder.generate(y, prompt, max_new_tokens=100, top_k=3)
        out_str = bytes(out.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
        print(out_str)