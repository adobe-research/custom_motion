# Copyright 2024 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.


import os
import decord
import numpy as np
import random
import json
import torchvision
import torchvision.transforms as T
import torch

from glob import glob
from PIL import Image
from itertools import islice
from pathlib import Path
from .bucketing import sensible_buckets
from itertools import cycle, islice

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange, repeat
from vidaug import augmentors as va
from itertools import chain
import cv2


def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids

def read_caption_file(caption_file):
        with open(caption_file, 'r', encoding="utf8") as t:
            return t.read()

    
def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices

def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch, effective_idx=None):
    # breakpoint()
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video, _, effective_idx = get_frame_batch(vr, resize=resize, effective_idx=effective_idx)

    else:
        vr = decord.VideoReader(vid_path, width=w, height=h, effective_idx=effective_idx)
        video, _, effective_idx  = get_frame_batch(vr)

    return video, vr, effective_idx



class SingleVideoDataset(Dataset):
    def __init__(
        self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            frame_step: int = 1,
            single_video_path: str = "",
            single_video_prompt: str = "",
            use_caption: bool = False,
            use_bucketing: bool = False,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing
        self.frames = []
        self.index = 1

        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.n_sample_frames = n_sample_frames
        self.frame_step = frame_step

        self.single_video_path = single_video_path
        self.single_video_prompt = single_video_prompt

        self.width = width
        self.height = height
    def create_video_chunks(self):
        # Create a list of frames separated by sample frames
        # [(1,2,3), (4,5,6), ...]
        vr = decord.VideoReader(self.single_video_path)
        vr_range = range(1, len(vr), self.frame_step)

        self.frames = list(self.chunk(vr_range, self.n_sample_frames))

        # Delete any list that contains an out of range index.
        for i, inner_frame_nums in enumerate(self.frames):
            for frame_num in inner_frame_nums:
                if frame_num > len(vr):
                    print(f"Removing out of range index list at position: {i}...")
                    del self.frames[i]

        return self.frames

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def get_frame_batch(self, vr, resize=None):
        index = self.index
        frames = vr.get_batch(self.frames[self.index])
        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize
    
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        
        return video, vr 

    def single_video_batch(self, index):
        train_data = self.single_video_path
        self.index = index

        if train_data.endswith(self.vid_types):
            video, _ = self.process_video_wrapper(train_data)

            prompt = self.single_video_prompt
            prompt_ids = get_prompt_ids(prompt, self.tokenizer)

            return video, prompt, prompt_ids
        else:
            raise ValueError(f"Single video is not a video type. Types: {self.vid_types}")
    
    @staticmethod
    def __getname__(): return 'single_video'

    def __len__(self):
        
        return len(self.create_video_chunks())

    def __getitem__(self, index):

        video, prompt, prompt_ids = self.single_video_batch(index)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }

        return example
    
class ImageDataset(Dataset):
    
    def __init__(
        self,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption:     bool = False,
        image_dir: str = '',
        single_img_prompt: str = '',
        use_bucketing: bool = False,
        fallback_prompt: str = '',
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.use_bucketing = use_bucketing

        self.image_dir = self.get_images_list(image_dir)
        self.fallback_prompt = fallback_prompt

        self.use_caption = use_caption
        self.single_img_prompt = single_img_prompt

        self.width = width
        self.height = height

    def get_images_list(self, image_dir):
        if os.path.exists(image_dir):
            imgs = [x for x in os.listdir(image_dir) if x.endswith(self.img_types)]
            full_img_dir = []

            for img in imgs: 
                full_img_dir.append(f"{image_dir}/{img}")

            return sorted(full_img_dir)

        return ['']

    def image_batch(self, index):
        train_data = self.image_dir[index]
        img = train_data

        try:
            img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        except:
            img = T.transforms.PILToTensor()(Image.open(img).convert("RGB"))

        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            width, height = sensible_buckets(width, height, w, h)
              
        resize = T.transforms.Resize((height, width), antialias=True)

        img = resize(img) 
        img = repeat(img, 'c h w -> f c h w', f=1)

        prompt = self.fallback_prompt
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return img, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        if os.path.exists(self.image_dir[0]):
            return len(self.image_dir)
        else:
            return 0

    def __getitem__(self, index):
        img, prompt, prompt_ids = self.image_batch(index)
        example = {
            "pixel_values": (img / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt, 
            'dataset': self.__getname__()
        }

        return example

class VideoFolderDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        fps: int = 8,
        path: str = "./data",
        class_path = "",
        class_prompt = "",
        fallback_prompt: str = "",
        use_bucketing: bool = False,
        add_token: bool = False,
        train_weights=None,
        sampling=None,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing
        self.video_files = glob(f"{path}/*.mp4")
        self.train_weights = train_weights

        self.sampling = sampling
        self.fallback_prompt = fallback_prompt
        self.class_video_files = glob(f"{class_path}/*.mp4")
        self.class_prompt = class_prompt

        print('Num videos:', len(self.video_files))
        print('Num class videos:', len(self.class_video_files))
        self.video_files = list(islice(cycle(self.video_files), len(self.class_video_files)))
        print('After padding Num videos:', len(self.video_files))
        print('After padding Num class videos:', len(self.class_video_files))

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps

        x = self.__getitem__(0)['text_prompt']
        y = self.__getitem__(0)['class_text_prompt']
        print('Using this prompt:', x, '\n Class prompt:', y)

    def get_frame_buckets(self, vr):
        h, w, _ = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)
        return resize

    def get_frame_batch(self, vr, resize=None, effective_idx=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()
        
        every_nth_frame = max(1, round(native_fps / self.fps))
        every_nth_frame = min(len(vr), every_nth_frame)
        
        effective_length = len(vr) // every_nth_frame
        if effective_length < n_sample_frames:
            n_sample_frames = effective_length

        if effective_idx is None:
            effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)

        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video, vr, effective_idx
        
    def process_video_wrapper(self, vid_path, effective_idx=None):
        video, vr, effective_idx = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch,
                effective_idx=effective_idx,
            )
        return video, vr, effective_idx
    
    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return max(len(self.video_files), len(self.class_video_files))



    def __getitem__(self, index):
        video, _, effective_idx = self.process_video_wrapper(self.video_files[index % len(self.video_files)])
        

        if os.path.exists(self.video_files[index % len(self.video_files)].replace(".mp4", ".txt")):
            with open(self.video_files[index % len(self.video_files)].replace(".mp4", ".txt"), "r") as f:
                prompt = f.read()
        elif os.path.exists(os.path.join(os.path.dirname(self.video_files[index % len(self.video_files)]), "text.json")):
            with open(os.path.join(os.path.dirname(self.video_files[index % len(self.video_files)]), "text.json"), "r") as f:
                prompt = np.random.choice(json.load(f)[os.path.basename(self.video_files[index % len(self.video_files)])])
                prompt = prompt.replace('  ', ' ')
        else:
            prompt = np.random.choice(self.fallback_prompt)
            


        prompt_ids = self.get_prompt_ids(prompt)
        
        
        
        if len(self.class_video_files) > 0:
            class_video, _, _ = self.process_video_wrapper(self.class_video_files[index % len(self.class_video_files)])
            if os.path.exists(self.class_video_files[index % len(self.class_video_files)].replace(".mp4", ".txt")):
                with open(self.class_video_files[index % len(self.class_video_files)].replace(".mp4", ".txt"), "r") as f:
                    class_prompt = f.read().replace('\n', ' ')
            elif self.class_prompt == "":
                class_prompt = self.class_video_files[index % len(self.class_video_files)].split('.')[0].replace('_', ' ')
            else:
                class_prompt = np.random.choice(self.class_prompt)
            class_prompt_ids = self.get_prompt_ids(class_prompt)
        else:
            class_video = video
            class_prompt_ids = prompt_ids
            class_prompt = ''
            
        train_weights = self.train_weights
        sampling = self.sampling
        if video.shape[0] != self.n_sample_frames:
            video = torch.cat([video, video[-1].unsqueeze(0).repeat(self.n_sample_frames - video.shape[0], 1, 1, 1)])
        
        if class_video.shape[0] != video.shape[0]:
            class_video = torch.cat([class_video, class_video[-1].unsqueeze(0).repeat(video.shape[0] - class_video.shape[0], 1, 1, 1)])


        return {"pixel_values": (video / 127.5 - 1.0), "prompt_ids": prompt_ids[0], 'class_text_prompt': class_prompt,
                  "text_prompt": prompt, 'dataset': self.__getname__(),
                  'train_weights': train_weights,'sampling': sampling,
            "pixel_valuesclass":  (class_video / 127.5 - 1.0), "prompt_idsclass": class_prompt_ids[0], 'path':self.video_files[index % len(self.video_files)]}

class CachedDataset(Dataset):
    def __init__(self,cache_dir: str = ''):
        self.cache_dir = cache_dir
        self.cached_data_list = self.get_files_list()

    def get_files_list(self):
        tensors_list = [f"{self.cache_dir}/{x}" for x in os.listdir(self.cache_dir) if x.endswith('.pt')]
        return sorted(tensors_list)

    def __len__(self):
        return len(self.cached_data_list)

    def __getitem__(self, index):
        cached_latent = torch.load(self.cached_data_list[index], map_location='cuda:0')
        return cached_latent

