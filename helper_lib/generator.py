import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def generate_gan_samples(generator, device="cpu", num_samples=16, nrow=4):
    generator.eval()
    z = torch.randn(num_samples, 100, device=device)

    with torch.no_grad():
        imgs = generator(z).cpu()

    grid = make_grid(imgs, nrow=nrow, normalize=True, pad_value=1)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.show()

    generator.train()

def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    return torch.linspace(beta_start, beta_end, T, device=device)

@torch.no_grad()
def diffusion_sample(model, device="cpu", num_steps=200, num_samples=16):

    betas = linear_beta_schedule(num_steps, device=device)
    alphas = 1.0 - betas
    alphabar = torch.cumprod(alphas, dim=0)

    x = torch.randn(num_samples, 1, 28, 28, device=device)

    for t in reversed(range(num_steps)):
        t_batch = torch.tensor([t], device=device).float() / (num_steps - 1)
        t_batch = t_batch.repeat(num_samples)

        eps_pred = model(x, t_batch)

        coef = betas[t] / torch.sqrt(1 - alphabar[t])
        x = (1 / torch.sqrt(alphas[t])) * (x - coef * eps_pred)

        if t > 0:
            x += torch.sqrt(betas[t]) * torch.randn_like(x)

    x = x.clamp(-1, 1).cpu()

    grid = make_grid(x, nrow=4, normalize=True)
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

    return x

def langevin_sample(energy_model, x_init, steps=30, step_size=0.1, noise_scale=0.005):
    x = x_init.clone().detach().requires_grad_(True)

    for _ in range(steps):
        e = energy_model(x).sum()
        grad = torch.autograd.grad(e, x, create_graph=False)[0]
        x = x - step_size * grad + noise_scale * torch.randn_like(x)
        x = x.clamp(-1, 1).detach().requires_grad_(True)

    return x.detach()

@torch.no_grad()
def ebm_generate(energy_model, device="cpu", num_samples=16):

    x_init = torch.randn(num_samples, 1, 28, 28, device=device)
    x = langevin_sample(energy_model, x_init)

    x = x.cpu()
    grid = make_grid(x, nrow=4, normalize=True)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

    return x
