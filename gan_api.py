from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch
from io import BytesIO
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from helper_lib.model import get_model
from helper_lib.utils import load_model

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = get_model("gan")[0]
generator = load_model(generator, "generator.pth", device)

def generate_image():
    z = torch.randn(16, 100, device=device)
    with torch.no_grad():
        imgs = generator(z).cpu()

    grid = make_grid(imgs, nrow=4, normalize=True, pad_value=1)
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf

@app.get("/generate")
def generate_endpoint():
    buf = generate_image()
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
