import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from glob import glob
import os
import subprocess

def catmull_rom_spline(P0, P1, P2, P3, num_points=100):
    t = torch.linspace(0, 1, num_points)
    t2 = t * t
    t3 = t2 * t

    f1 = -0.5 * t3 + t2 - 0.5 * t
    f2 = 1.5 * t3 - 2.5 * t2 + 1.0
    f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    f4 = 0.5 * t3 - 0.5 * t2

    return f1[:, None] * P0 + f2[:, None] * P1 + f3[:, None] * P2 + f4[:, None] * P3

def generate_n_dimensional_catmull_rom_spline(points, num_points):
    curve_points = []

    for i in range(1, len(points) - 2):
        P0, P1, P2, P3 = points[i - 1], points[i], points[i + 1], points[i + 2]
        spline_points = catmull_rom_spline(P0, P1, P2, P3, num_points[i])
        curve_points.append(spline_points)

    return torch.vstack(curve_points).reshape(-1, 18, 512)


def create_video_from_images(output_path, frame_rate=48):
    command = [
        'ffmpeg', '-framerate', str(frame_rate), '-i', './vid/sample_%04d.jpg',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_path, "-y"
    ]
    subprocess.run(command, check=True)

def prepare_z(g_ema):
    z = torch.randn(1, 512).to("cuda")
    z = g_ema.style(z)
    z = z.unsqueeze(1).repeat(1, 18, 1)
    return z

if __name__ == "__main__":
    # No grad
    torch.set_grad_enabled(False)
    device = "cuda"

    os.makedirs("./vid", exist_ok=True)

    g_ema = Generator(1024, 512, 8, channel_multiplier=2).to(device)
    checkpoint = torch.load("./stylegan2-ffhq-config-f.pt")
    g_ema.load_state_dict(checkpoint["g_ema"])

    z_path_list = glob("./latents/*.pt")    

    z_list = []
    for z_path in z_path_list:
        z_list.append(torch.load(f"./latents/{z_path}")) 

    # last point for the catmull-rom spline
    z_list.append(prepare_z(g_ema))
    z_list = [z.flatten().cpu() for z in z_list]

    num_points_list = [50 for _ in range(len(z_list) - 1)]

    num_points_list[-3] = 20
    num_points_list[-2] = 20

    z_list = generate_n_dimensional_catmull_rom_spline(z_list, num_points=num_points_list).cuda()

    velocity = z_list[-1] - z_list[-2]
    target_pos = z_list[-1].clone()
    current_pos = z_list[-1].clone()

    # Use tqdm for progress tracking in the dynamic sequence generation loop
    for _ in tqdm(range(100), desc="Generating dynamic sequence"):
        current_pos += velocity
        velocity += (target_pos - current_pos) * 0.05
        velocity *= 0.9
        z_list = torch.cat([z_list, current_pos.unsqueeze(0)], dim=0)
            
    # Save each generated image with tqdm progress bar
    for i, z in enumerate(tqdm(z_list, desc="Saving images")):
        i_str = str(i).zfill(4)
        sample, _ = g_ema([z.unsqueeze(0)], input_is_latent=True, randomize_noise=False)
        utils.save_image(sample, f"./vid/sample_{i_str}.jpg", nrow=1, normalize=True, value_range=(-1, 1))

    # Create the video from saved images
    create_video_from_images('./output_video.mp4')
