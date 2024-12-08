from torchvision import utils
from model import Generator
from tqdm import tqdm
from glob import glob
from PIL import Image
from losses import PerceptualLossVGG

import torch
import numpy as np
import os

if __name__ == "__main__":
    device = "cuda"

    g_ema = Generator(1024, 512, 8, channel_multiplier=2).to(device)
    checkpoint = torch.load("./stylegan2-ffhq-config-f.pt")
    g_ema.load_state_dict(checkpoint["g_ema"])
    g_ema.eval()

    img_path_list = sorted(glob("./data/*.png")) + sorted(glob("./data/*.jpg"))
    criterion_vgg = PerceptualLossVGG().cuda()

    for img_path in img_path_list:
        # z = g_ema.mean_latent(4096)
        z = g_ema.style(checkpoint["latent_avg"].clone().unsqueeze(0).to(device))
        z = z.unsqueeze(0).repeat(1, 18, 1).detach().clone().to(device)
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=0.01)

        pbar = tqdm(range(1000))

        img = Image.open(img_path)
        img = img.resize((1024, 1024))
        img = np.array(img)
        img = (img/127.5 - 1.0).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        for i in pbar:
            optimizer.zero_grad()
            sample, _ = g_ema([z], input_is_latent=True, randomize_noise=False)

            loss = torch.nn.functional.mse_loss(sample, img)
            loss += criterion_vgg(sample, img) * 0.1
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                i_str = str(i).zfill(4)
                pbar.set_description(f"Loss: {loss.item()}")
                utils.save_image(sample, f"./sample/sample_{i_str}.png", nrow=1, normalize=True, value_range=(-1, 1))
        latent_name = os.path.basename(img_path).split(".")[0]
        torch.save(z.detach(), f"./latents/{latent_name}.pt")