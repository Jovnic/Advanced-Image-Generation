import torch
from torch import optim, nn
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model, train_gan, train_diffusion, train_ebm
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import save_model, load_model
from helper_lib.generator import generate_gan_samples, diffusion_sample, ebm_generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_data_loader(data_dir="data", batch_size=64, train=True)
test_loader = get_data_loader(data_dir="data", batch_size=64, train=False)

run_cnn = False
run_gan = False
run_diffusion = True
run_ebm = False

if run_cnn:
    model = get_model("cnn").to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        loss = train_model(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    test_loss, acc = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {acc:.2f}%")

    save_model(model, "cnn.pth")
    model = load_model(model, "cnn.pth", device)

if run_gan:
    generator, discriminator = get_model("gan")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    train_gan(generator, discriminator, train_loader, optimizer_G, optimizer_D, criterion, epochs=20, device=device)

    save_model(generator, "generator.pth")
    save_model(discriminator, "discriminator.pth")

    generator = load_model(generator, "generator.pth", device)
    generate_gan_samples(generator, device=device)

if run_diffusion:
    diffusion_model = get_model("diffusion").to(device)
    optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-3)

    train_diffusion(diffusion_model, train_loader, optimizer, device=device, epochs=3, T=200)

    save_model(diffusion_model, "diffusion.pth")

    diffusion_model = load_model(diffusion_model, "diffusion.pth", device)
    diffusion_sample(diffusion_model, device=device, num_steps=200, num_samples=16)

if run_ebm:
    ebm = get_model("ebm").to(device)
    optimizer = optim.Adam(ebm.parameters(), lr=1e-4)

    train_ebm(ebm, train_loader, optimizer, device=device, epochs=3, steps=30, step_size=0.1, noise_scale=0.01)

    save_model(ebm, "ebm.pth")

    ebm = load_model(ebm, "ebm.pth", device)
    ebm_generate(ebm, device=device, num_samples=16)
