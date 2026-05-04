#!/usr/bin/env python3

import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

from config import config


class ImageDataset(Dataset):

    def __init__(self, root_dir, image_size=64, center_crop=True, random_flip=True,
                 color_jitter=True, color_jitter_strength=0.2, random_rotation=10):
        self.root_dir = Path(root_dir)
        self.image_paths = sorted([
            str(p) for p in self.root_dir.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ])
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        transform_list = [
            transforms.Resize(image_size + 8, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(image_size),
        ]
        if random_rotation > 0:
            transform_list.append(transforms.RandomRotation(random_rotation,
                interpolation=transforms.InterpolationMode.BILINEAR, fill=0))
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=min(color_jitter_strength, 0.2),
            ))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def validate(model, scheduler, accelerator, epoch, num_images=8, device="cpu"):
    model.eval()
    pipeline = DDPMPipeline(
        unet=accelerator.unwrap_model(model),
        scheduler=scheduler,
    )
    pipeline.to(device)

    gen_device = "cpu" if str(device) == "mps" else device
    generator = torch.Generator(device=gen_device).manual_seed(42)

    images = pipeline(
        batch_size=num_images,
        num_inference_steps=scheduler.config.num_train_timesteps,
        generator=generator,
        output_type="pil",
    ).images

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis("off")
    plt.suptitle(f"Epoch {epoch}")
    os.makedirs(config.output_dir, exist_ok=True)
    plt.savefig(os.path.join(config.output_dir, f"val_epoch_{epoch:04d}.png"),
                bbox_inches="tight", dpi=100)
    plt.close()
    print(f"  [Validation] Saved to {config.output_dir}/val_epoch_{epoch:04d}.png")
    model.train()


def train():
    parser = argparse.ArgumentParser(description="Train DDPM diffusion model")
    parser.add_argument("--dataset", type=str, default=config.dataset_path,
                        help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=config.num_epochs,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate,
                        help="Learning rate")
    parser.add_argument("--image_size", type=int, default=config.image_size,
                        help="Image size")
    parser.add_argument("--output_dir", type=str, default=config.output_dir,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=config.seed,
                        help="Random seed")
    args = parser.parse_args()

    config.dataset_path = args.dataset
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.image_size = args.image_size
    config.output_dir = args.output_dir
    config.seed = args.seed

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, config.logging_dir),
    )

    os.makedirs(config.output_dir, exist_ok=True)

    dataset = ImageDataset(
        root_dir=config.dataset_path,
        image_size=config.image_size,
        center_crop=config.center_crop,
        random_flip=config.random_flip,
        color_jitter=config.color_jitter,
        color_jitter_strength=config.color_jitter_strength,
        random_rotation=config.random_rotation if config.random_rotation else 0,
    )
    use_mps = torch.backends.mps.is_available()
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=not use_mps,
    )
    if accelerator.is_main_process:
        print(f"Dataset: {len(dataset)} images loaded from {config.dataset_path}")
        print(f"Image size: {config.image_size}x{config.image_size}")
        print(f"Batch size: {config.batch_size}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Total steps per epoch: {len(train_dataloader)}")

    model = UNet2DModel(**config.model_config)
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    noise_scheduler = DDPMScheduler(**config.noise_scheduler_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * config.num_epochs,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        if config.logging_dir is not None:
            accelerator.init_trackers("ddpm_cats", config={
                "dataset_size": len(dataset),
                "batch_size": config.batch_size,
                "epochs": config.num_epochs,
                "lr": config.learning_rate,
                "image_size": config.image_size,
                "model_params": total_params,
            })

    global_step = 0
    loss_history = []
    best_loss = float("inf")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{config.num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                clean_images = batch
                noise = torch.randn_like(clean_images)
                bs = clean_images.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bs,), device=clean_images.device,
                    dtype=torch.long,
                )
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_loss += loss.detach().item()
                loss_history.append(loss.detach().item())

                if accelerator.is_main_process:
                    log_dict = {
                        "train_loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    accelerator.log(log_dict)

                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        checkpoint_dir = os.path.join(
                            config.output_dir, f"checkpoint_step_{global_step}"
                        )
                        pipeline = DDPMPipeline(
                            unet=accelerator.unwrap_model(model),
                            scheduler=noise_scheduler,
                        )
                        pipeline.save_pretrained(checkpoint_dir)
                        print(f"\n  [Checkpoint] Saved model to {checkpoint_dir}")

            progress_bar.set_postfix({"loss": loss.detach().item()})

        progress_bar.close()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{config.num_epochs} - Average loss: {avg_epoch_loss:.6f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            if accelerator.is_main_process:
                best_path = os.path.join(config.output_dir, "best_model")
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler,
                )
                pipeline.save_pretrained(best_path)
                print(f"  [Best Model] Saved with loss {best_loss:.6f}")

        if (epoch + 1) % config.validation_epochs == 0 and accelerator.is_main_process:
            validate(
                model,
                noise_scheduler,
                accelerator,
                epoch + 1,
                num_images=config.num_validation_images,
                device=accelerator.device,
            )

    if accelerator.is_main_process:
        final_path = os.path.join(config.output_dir, "final_model")
        pipeline = DDPMPipeline(
            unet=accelerator.unwrap_model(model),
            scheduler=noise_scheduler,
        )
        pipeline.save_pretrained(final_path)
        print(f"\nTraining complete! Final model saved to {final_path}")

        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.xlabel("Step")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss")
        plt.yscale("log")
        plt.savefig(os.path.join(config.output_dir, "loss_history.png"), dpi=100)
        plt.close()
        print(f"Loss plot saved to {config.output_dir}/loss_history.png")

    accelerator.end_training()


if __name__ == "__main__":
    train()
