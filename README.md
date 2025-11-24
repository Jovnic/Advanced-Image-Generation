# GAN Image Generator API

## 1. Clone the repository
```bash
git clone https://github.com/Jovnic/Advanced-Image-Generation.git
cd Advanced-Image-Generation
```

## 2. Create and sync environment with uv
```bash
uv sync
```

## 3. Run training (optional)
```bash
uv run python main.py
```

This will train the models and save:
```
cnn.pth
generator.pth
discriminator.pth
diffusion.pth

```

## 4. Run FastAPI server
```bash
uv run python gan_api.py
```

## 5. Open API documentation
Go to:
```
http://127.0.0.1:8000/docs
```

Click `/generate` → “Try it out” → Execute.
You can find the picture in the output folder.
## 6. Docker deployment
Build and run the container:
```bash
docker build -t gan-api .
docker run -p 8000:8000 gan-api
```
Then open again:
```
http://127.0.0.1:8000/docs
```

## 7. Note
- If `generator.pth` is missing, the API still runs using the initialized model.
- Environment is fully reproducible with `pyproject.toml` and `uv.lock`.
