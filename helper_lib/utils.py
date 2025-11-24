import torch
import os

def save_model(model, path):
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
