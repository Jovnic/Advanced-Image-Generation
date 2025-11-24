import torch
import torch.nn.functional as F

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, epochs, device):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        g_loss_total, d_loss_total = 0.0, 0.0

        for imgs, _ in dataloader:
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            optimizer_G.zero_grad()

            z = torch.randn(batch_size, 100, device=device)
            gen_imgs = generator(z)

            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | D loss: {d_loss_total/len(dataloader):.4f} | G loss: {g_loss_total/len(dataloader):.4f}")

    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")

def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    return torch.linspace(beta_start, beta_end, T, device=device)

def train_diffusion(noise_model, dataloader, optimizer, device, epochs=5, T=200):
    noise_model.train()

    betas = linear_beta_schedule(T, device=device)
    alphas = 1.0 - betas
    alphabar = torch.cumprod(alphas, dim=0)

    for epoch in range(epochs):
        total_loss = 0.0

        for x0, _ in dataloader:
            x0 = x0.to(device)
            b = x0.size(0)

            t = torch.randint(0, T, (b,), device=device).long()
            a_bar_t = alphabar[t].view(-1, 1, 1, 1)

            eps = torch.randn_like(x0)
            xt = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * eps

            t_norm = t.float() / (T - 1)

            optimizer.zero_grad()
            eps_pred = noise_model(xt, t_norm)
            loss = F.mse_loss(eps_pred, eps)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | diffusion loss: {total_loss/len(dataloader):.4f}")

def langevin_sample(energy_model, x_init, steps=30, step_size=0.1, noise_scale=0.005):
    x = x_init.clone().detach().requires_grad_(True)

    for _ in range(steps):
        e = energy_model(x).sum()
        grad = torch.autograd.grad(e, x, create_graph=False)[0]
        x = x - step_size * grad + noise_scale * torch.randn_like(x)
        x = x.clamp(-1, 1).detach().requires_grad_(True)

    return x.detach()

def train_ebm(energy_model, dataloader, optimizer, device, epochs=5, steps=30, step_size=0.1, noise_scale=0.005):
    energy_model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for x_real, _ in dataloader:
            x_real = x_real.to(device)

            x_neg_init = torch.randn_like(x_real)
            x_neg = langevin_sample(energy_model, x_neg_init, steps=steps, step_size=step_size, noise_scale=noise_scale)

            optimizer.zero_grad()
            e_real = energy_model(x_real).mean()
            e_neg = energy_model(x_neg).mean()
            loss = e_real - e_neg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | ebm loss: {total_loss/len(dataloader):.4f}")
