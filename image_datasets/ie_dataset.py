import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2


# def c_crop(image):
#     width, height = image.size
#     new_size = min(width, height)
#     left = (width - new_size) / 2
#     top = (height - new_size) / 2
#     right = (width + new_size) / 2
#     bottom = (height + new_size) / 2
#     return image.crop((left, top, right, bottom))

def aspect_resize(image, max_length):
    width, height = image.size
    if width > height:
        ratio = max_length / width
    else:
        ratio = max_length / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return image.resize((new_width, new_height))

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, prompt=""):
        assert "input" in os.listdir(img_dir)
        assert "target" in os.listdir(img_dir)

        train_dir = os.path.join(img_dir, "input")
        target_dir = os.path.join(img_dir, "target")
        self.input_images = [os.path.join(img_dir,"input", i) for i in os.listdir(train_dir) if '.jpg' in i or '.png' in i]
        self.target_images = [os.path.join(img_dir,"target", i) for i in os.listdir(target_dir) if '.jpg' in i or '.png' in i]
        self.input_images.sort()
        self.target_images.sort()
        self.img_size = img_size
        self.prompt = prompt

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        try:
            input_img = Image.open(self.input_images[idx])
            input_img = aspect_resize(input_img, self.img_size)
            input_img = torch.from_numpy((np.array(input_img) / 127.5) - 1)
            input_img = input_img.permute(2, 0, 1)

            target_img = Image.open(self.target_images[idx])
            target_img = aspect_resize(target_img, self.img_size)
            target_img = torch.from_numpy((np.array(target_img) / 127.5) - 1)
            target_img = target_img.permute(2, 0, 1)

            return target_img, input_img, self.prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
