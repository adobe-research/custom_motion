# Copyright 2024 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.


import logging
import inspect
import math
import os
import numpy as np
import argparse
import uuid
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video
from typing import Dict, Optional, Tuple

from utils_train import *



logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    dataset_types: Tuple[str] = ('json'),
    validation_steps: int = 10000,
    trainable_modules: Tuple[str] = None,
    trainable_text_modules: Tuple[str] = None,
    extra_unet_params = None,
    extra_text_encoder_params = None,
    train_batch_size: int = 1,
    max_train_steps: int = 2001,
    learning_rate: float = 5.0e-06,
    txt_learning_rate: float = 5.0e-06,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    text_encoder_gradient_checkpointing: bool = False,
    checkpointing_steps: int = 50000,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = 64,
    train_text_encoder: bool = False,
    use_offset_noise: bool = False,
    offset_noise_strength: float = 0.1,
    extend_dataset: bool = False,
    cache_latents: bool = False,
    cached_latent_dir = None,
    use_unet_lora: bool = False,
    use_text_lora: bool = False,
    unet_lora_modules: Tuple[str] = ["ResnetBlock2D"],
    text_encoder_lora_modules: Tuple[str] = ["CLIPEncoderLayer"],
    prior_preservation = False,
    prior_lambda = 0.,
    train_text_token = False,
    subject = '',
    test_seed=0,
    eval_script=None,
    sampling='uniform',
    class_name='',
    ignore_tokenizer_warnings=False,
    **kwargs
):

    *_, config = inspect.getargvalues(inspect.currentframe())
    

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        # log_with="wandb",
        project_dir=output_dir
    )


    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)
    # Handle the output folder creation
    if accelerator.is_main_process:
        out_dir_name = f'{subject}_{("+").join(trainable_modules)}_TOKEN{train_text_token}_PRIOR{prior_preservation}{prior_lambda}_lr{learning_rate}_SAMPLING{sampling}'
        output_dir = create_output_folders(output_dir, out_dir_name, config)

    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_model_path)

    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    optim_params = [
        param_optim(unet, trainable_modules != 'none', extra_params=extra_unet_params, negation=None),
    ]
    params = create_optimizer_params(optim_params, learning_rate)

  
    # Create Optimizer
    optimizer = optimizer_cls(params,lr=learning_rate,betas=(adam_beta1, adam_beta2),weight_decay=adam_weight_decay,eps=adam_epsilon)
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Get the training dataset based on types (json, single_video, image)
    train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)

    # Extend datasets that are less than the greatest one. This allows for more balanced training.
    attrs = ['train_data', 'frames', 'image_dir', 'video_files']
    extend_datasets(train_datasets, attrs, extend=extend_dataset)

    # Process one dataset
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    
    # Process many datasets
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets) 
        

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=True
    )

     # Latents caching
    cached_data_loader = handle_cache_latents(
        cache_latents, 
        output_dir,
        train_dataloader, 
        train_batch_size, 
        vae,
        cached_latent_dir
    ) 

    if cached_data_loader is not None: 
        train_dataloader = cached_data_loader
    # Prepare everything with our `accelerator`.
    unet, optimizer,train_dataloader, lr_scheduler, text_encoder = accelerator.prepare(
        unet, 
        optimizer, 
        train_dataloader, 
        lr_scheduler, 
        text_encoder
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet, 
        text_encoder, 
        gradient_checkpointing, 
        text_encoder_gradient_checkpointing
    )
    
    
    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2video-fine-tune")

    val_prompts = []
    # get val prompts
    for file in [validation_data.prompt]:
        f = open(file, 'r')
        prompts = []
        for idx, line in enumerate(f.readlines()):
            l = line.strip()
            if len(l) != 0:
                prompts.append([int(l.split(',')[0]), l.split(',')[1]])
        f.close()
        val_prompts.extend(prompts)


    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    text_encoder = text_encoder.to(torch.float32)

   
    def finetune_unet(batch, prior_preservation, prior_lambda, train_encoder=False, key=''):
        # Check if we are training the text encoder
        text_trainable = (train_text_encoder or use_text_lora)
        
        # Unfreeze UNET Layers
        if global_step == 0: 
            already_printed_trainables = False
            unet.train()
            handle_trainable_modules(
                unet, 
                trainable_modules, 
                is_enabled=True,
            )

        # Convert videos to latent space
        pixel_values = batch[f"pixel_values"]
        if prior_preservation:
            pixel_values_class = batch[f"pixel_valuesclass"]
            pixel_values = torch.cat([pixel_values, pixel_values_class], dim=0)
        if not cache_latents:
            latents = tensor_to_vae_latent(pixel_values, vae)
        else:
            latents = pixel_values

        # Get video length
        video_length = latents.shape[2]

        # Sample noise that we'll add to the latents
        if prior_preservation:
            noise = sample_noise(latents[0][None, :], offset_noise_strength, use_offset_noise)
            noise1 = sample_noise(latents[1][None, :], offset_noise_strength, use_offset_noise)
            noise = torch.cat([noise, noise1], dim=0)
        else:
            noise = sample_noise(latents, offset_noise_strength, use_offset_noise)

        bsz = latents.shape[0]
        update = batch['train_weights'][0]
        sampling = batch['sampling'][0]
        # Sample a random timestep for each video
        if sampling == 'uniform' or sampling is None:
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        elif sampling == 'high-level':
            def f(t, a=0.5, T=1000):
                return (1/T)*(1 - a*np.cos((np.pi*t)/T))
            probabilities = np.array([f(t) for t in range(noise_scheduler.config.num_train_timesteps)])
            probabilities = probabilities / probabilities.sum()
            timesteps = torch.tensor(np.random.choice(np.arange(noise_scheduler.config.num_train_timesteps), size=bsz, p=probabilities))
        elif sampling == 'low-level':
            def f(t, a=0.5, T=1000):
                return (1/T)*(1 + a*np.cos((np.pi*t)/T))
            probabilities = np.array([f(t) for t in range(noise_scheduler.config.num_train_timesteps)])
            probabilities = probabilities / probabilities.sum()
            timesteps = torch.tensor(np.random.choice(np.arange(noise_scheduler.config.num_train_timesteps), size=bsz, p=probabilities))
        timesteps = timesteps.long().to(latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
      
        # Fixes gradient checkpointing training.
        # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        if gradient_checkpointing or text_encoder_gradient_checkpointing:
            unet.eval()
            text_encoder.eval()
            
        # Encode text embeddings
        token_ids = batch[f'prompt_ids']
        if prior_preservation:
            token_ids_class = batch[f'prompt_idsclass']
            token_ids = torch.cat([token_ids, token_ids_class], dim=0)
        encoder_hidden_states = text_encoder(token_ids)[0]

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise

        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)

        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

        
        # Here we do two passes for video and text training.
        # If we are on the second iteration of the loop, get one frame.
        # This allows us to train text information only on the spatial layers.
        losses = []
        should_truncate_video = (video_length > 1 and text_trainable)

        # We detach the encoder hidden states for the first pass (video frames > 1)
        # Then we make a clone of the initial state to ensure we can train it in the loop.
        detached_encoder_state = encoder_hidden_states.clone().detach()
        trainable_encoder_state = encoder_hidden_states.clone()

        
        should_detach  = False
        if update == 'spatial':
            should_truncate_video = True
        elif update == 'temporal' or update == 'all':
            should_truncate_video = False
        
        if global_step == 0: 
            print(f'update: {update}', "should_truncate_video: ", should_truncate_video)
        if should_truncate_video:
            noisy_latents = noisy_latents[:,:,1,:,:].unsqueeze(2)
            target = target[:,:,1,:,:].unsqueeze(2)
                    
        encoder_hidden_states = (
            detached_encoder_state if should_detach else trainable_encoder_state
        )

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        
        if not prior_preservation:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss_dict = {'loss': loss, 'prior_loss': torch.tensor(0.0)}

        else:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            loss_dict = {'loss': loss, 'prior_loss': prior_loss}
            loss = loss + prior_lambda * prior_loss

        losses.append(loss)
    

        loss = losses[0] if len(losses) == 1 else losses[0] + losses[1] 
        return loss, latents, loss_dict

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet) ,accelerator.accumulate(text_encoder):                
                with accelerator.autocast():
                    loss, latents, loss_dict = finetune_unet(batch, prior_preservation, prior_lambda, train_encoder=train_text_encoder)
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
            
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
            
                if global_step in checkpointing_steps:
                    save_pipe(
                        pretrained_model_path, 
                        global_step, 
                        accelerator, 
                        unet, 
                        text_encoder,
                        tokenizer, 
                        vae, 
                        output_dir,
                        is_checkpoint=True,
                        logger=logger,
                    )
            

                if global_step in validation_steps:
                    if accelerator.is_main_process:

                        with accelerator.autocast():
                            unet.eval()
                            text_encoder.eval()
                            unet_and_text_g_c(unet, text_encoder, False, False)
                            pipeline = TextToVideoSDPipeline.from_pretrained(
                                pretrained_model_path,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                vae=vae,
                                unet=unet
                            )

                            diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                            pipeline.scheduler = diffusion_scheduler

                            
                            os.makedirs(f"{output_dir}/samples/step_{global_step:05d}", exist_ok=True)
                            os.makedirs("tmp", exist_ok=True)
                            for _, prompt in enumerate(val_prompts):
                                seed, prompt = prompt[0], prompt[1]
                                _ = seed
                                save_filename = f"{prompt.replace(' ', '_')}"

                                out_file = f"{output_dir}/samples/step_{global_step:05d}/{save_filename}_seed{_}mp4"
                                if not out_file.endswith('.mp4'):
                                    out_file = out_file.replace('mp4', '.mp4')
                                with torch.no_grad():
                                    generator = torch.Generator(device=latents.device).manual_seed(_)
                                    video_frames = pipeline(
                                        prompt,
                                        width=validation_data.width,
                                        height=validation_data.height,
                                        num_frames=validation_data.num_frames,
                                        num_inference_steps=validation_data.num_inference_steps,
                                        guidance_scale=validation_data.guidance_scale,
                                        generator=generator,
                                    ).frames    
                                    rnd = str(uuid.uuid4())
                                    export_to_video((255*video_frames).astype(np.uint8)[0], f'tmp/{rnd}.mp4', 8)
                                    command = f'ffmpeg -y -i "tmp/{rnd}.mp4" -c:v libx264 "{out_file}"'
                                    os.system(command)
                                    os.system(f'rm tmp/{rnd}.mp4')                            
                            del pipeline
                            torch.cuda.empty_cache()

                    logger.info(f"Saved a new sample to {out_file}")

                    unet_and_text_g_c(
                        unet, 
                        text_encoder, 
                        gradient_checkpointing, 
                        text_encoder_gradient_checkpointing
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
             "prior_loss":loss_dict['prior_loss'].detach().item(), "instance_loss": loss_dict['loss'].detach().item()}
            accelerator.log({"training_loss": loss.detach().item()}, step=step)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
                pretrained_model_path, 
                global_step, 
                accelerator, 
                unet, 
                text_encoder, 
                tokenizer,
                vae, 
                output_dir,
                is_checkpoint=False,
                logger=logger,
        )     
    accelerator.end_training()
    os.system(f'rm tmp')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/my_config.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))

