import torch
from pathlib import Path
import os
from loadenv import LoadEnv
from jdh import JDHConfig
from jdh import JDH
from decoder_model import Decoder

max_new_tokens = 10

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
    collapse_count = 0
    while True:
        user_prompt = input("\033[36mEnter a prompt (or 'exit' to quit): \033[0m")
        tmp = user_prompt
        if user_prompt.lower() == "exit":
            break
        prompt = torch.tensor(
            bytearray(user_prompt, "utf-8"), dtype=torch.long, device=device
        ).unsqueeze(0)

        if prompt.size(1) > 512:
            print("\033[31mPrompt is too long. Please enter a shorter prompt (max 512 characters).\033[0m")
            continue
        elif prompt.size(1) < 1:
            print("\033[31mPrompt is too short. Please enter a longer prompt (min 8 characters).\033[0m")
            continue

        print("\033[32mGenerating text...\033[0m")
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # re-encode the full (growing) context, predict the next chunk,
                # then decode only the last predicted embedding -> one token
                x = model.forward(prompt)
                y = model.predict(x)
                print(f"Standart deviation: {torch.std(y).item():.4f}")
                idx_next = decoder.generate(y, top_k=3)
                token = bytes(idx_next.item())
                if  bytes("\x00", "utf-8") in token:
                    collapse_count += 1
                    if collapse_count > 5:
                        print("\033[31mGeneration collapsed. Stopping.\033[0m")
                        break
                else:
                    collapse_count = 0
                token = token.decode("utf-8", errors="ignore")
                # stream each token so output is visible immediately
                print(token, end="", flush=True)
                prompt = torch.cat((prompt, idx_next), dim=1)

        print()