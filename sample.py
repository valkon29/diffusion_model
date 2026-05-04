#!/usr/bin/env python3

import os
import time
import argparse

import torch
import numpy as np
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler, DDIMScheduler
from PIL import Image

from config import config


def make_grid(images, nrow=4, padding=2):
    if isinstance(images[0], Image.Image):
        w, h = images[0].size
        nrow_img = nrow
        ncol = (len(images) + nrow - 1) // nrow
        grid = Image.new("RGB", (ncol * w + (ncol - 1) * padding,
                                 nrow_img * h + (nrow_img - 1) * padding))
        for i, img in enumerate(images):
            x = (i % ncol) * (w + padding)
            y = (i // ncol) * (h + padding)
            grid.paste(img, (x, y))
        return grid
    else:
        images = np.array(images)
        n, h, w, c = images.shape
        nrow_img = nrow
        ncol = (len(images) + nrow - 1) // nrow
        grid_w = ncol * w + (ncol - 1) * padding
        grid_h = nrow_img * h + (nrow_img - 1) * padding
        grid = np.zeros((grid_h, grid_w, c), dtype=images.dtype)
        for i in range(n):
            col = i % ncol
            row = i // ncol
            x = col * (w + padding)
            y = row * (h + padding)
            grid[y:y+h, x:x+w] = images[i]
        return grid


def sample_ddpm(pipeline, num_images, num_steps, device, generator):
    scheduler = DDPMScheduler(
        num_train_timesteps=num_steps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    pipeline.scheduler = scheduler

    start_time = time.time()
    images = pipeline(
        batch_size=num_images,
        num_inference_steps=num_steps,
        generator=generator,
        output_type="pil",
    ).images
    elapsed = time.time() - start_time

    return images, elapsed


def sample_ddim(unet, num_images, num_steps, eta, device, generator):
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipeline = DDIMPipeline(unet=unet, scheduler=scheduler)
    pipeline.to(device)

    start_time = time.time()
    images = pipeline(
        batch_size=num_images,
        num_inference_steps=num_steps,
        eta=eta,
        generator=generator,
        output_type="pil",
    ).images
    elapsed = time.time() - start_time

    return images, elapsed


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Sample from trained diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--method", type=str, default="ddpm",
                        choices=["ddpm", "ddim", "both"],
                        help="Sampling method")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of inference steps")
    parser.add_argument("--num_images", type=int, default=16,
                        help="Number of images to generate")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta parameter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="samples",
                        help="Output directory for generated images")
    args = parser.parse_args()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print(f"Loading model from {args.checkpoint}...")
    pipeline = DDPMPipeline.from_pretrained(args.checkpoint)
    pipeline.to("mps" if torch.backends.mps.is_available() else "cpu")
    device = pipeline.device
    print(f"Model loaded on {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.method in ("ddpm", "both"):
        steps = args.steps if args.steps is not None else 1000
        print(f"\n--- DDPM Sampling ({steps} steps) ---")
        images, elapsed = sample_ddpm(pipeline, args.num_images, steps, device, generator)

        grid = make_grid(images, nrow=int(math.sqrt(args.num_images)))
        save_path = os.path.join(args.output_dir, f"ddpm_{steps}steps.png")
        grid.save(save_path)
        print(f"  Generated {len(images)} images in {elapsed:.2f}s ({elapsed/len(images):.2f}s per image)")
        print(f"  Saved to {save_path}")

    if args.method in ("ddim", "both"):
        steps = args.steps if args.steps is not None else 100
        print(f"\n--- DDIM Sampling ({steps} steps, eta={args.eta}) ---")
        images, elapsed = sample_ddim(pipeline.unet, args.num_images, steps, args.eta, device, generator)

        grid = make_grid(images, nrow=int(math.sqrt(args.num_images)))
        save_path = os.path.join(args.output_dir, f"ddim_{steps}steps_eta{args.eta}.png")
        grid.save(save_path)
        print(f"  Generated {len(images)} images in {elapsed:.2f}s ({elapsed/len(images):.2f}s per image)")
        print(f"  Saved to {save_path}")

if __name__ == "__main__":
    import math
    main()
