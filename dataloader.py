import os
from os.path import join
import csv
import cv2, copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
# import torchaudio
import sys
from scipy.io import wavfile
import json


class DHF1KDataset(Dataset):
    def __init__(self, path_data, len_snippet, mode="train", multi_frame=0, alternate=1):
        self.path_data = path_data
        self.len_snippet = len_snippet
        self.mode = mode
        self.multi_frame = multi_frame
        self.alternate = alternate
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        if self.mode == "train":
            self.video_names = os.listdir(path_data)
            self.list_num_frame = [len(os.listdir(os.path.join(path_data, d, 'images'))) for d in self.video_names]
        elif self.mode == "val":
            self.list_num_frame = []
            for v in os.listdir(path_data):
                for i in range(0, len(
                        os.listdir(os.path.join(path_data, v, 'images'))) - self.alternate * self.len_snippet,
                               32):
                    self.list_num_frame.append((v, i))
        else:
            self.list_num_frame = []
            for v in os.listdir(path_data):
                for i in range(0, len(
                        os.listdir(os.path.join(path_data, v, 'images'))) - self.alternate * self.len_snippet,
                               self.len_snippet):
                    self.list_num_frame.append((v, i))
                self.list_num_frame.append(
                    (v, len(os.listdir(os.path.join(path_data, v, 'images'))) - self.len_snippet))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        if self.mode == "train":
            file_name = self.video_names[idx]
            start_idx = np.random.randint(0, self.list_num_frame[idx] - self.alternate * self.len_snippet + 1)
        elif self.mode == "val" or self.mode == "save":
            (file_name, start_idx) = self.list_num_frame[idx]

        path_clip = os.path.join(self.path_data, file_name, 'images')
        path_annt = os.path.join(self.path_data, file_name, 'maps')

        clip_img = []
        clip_gt = []

        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, '%04d.png' % (start_idx + self.alternate * i + 1))).convert('RGB')
            sz = img.size

            if self.mode != "save":
                gt = np.array(
                    Image.open(os.path.join(path_annt, '%04d.png' % (start_idx + self.alternate * i + 1))).convert('L'))
                gt = gt.astype('float')

                if self.mode == "train":
                    gt = cv2.resize(gt, (384, 224))

                if np.max(gt) > 1.0:
                    gt = gt / 255.0
                clip_gt.append(torch.FloatTensor(gt))

            clip_img.append(self.img_transform(img))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        if self.mode != "save":
            clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))
        if self.mode == "save":
            return clip_img, start_idx, file_name, sz
        else:
            if self.multi_frame == 0:
                return clip_img, clip_gt[-1]
            return clip_img, clip_gt


class Hollywood_UCFDataset(Dataset):
    def __init__(self, path_data, len_snippet, mode="train", frame_no="last", multi_frame=0):
        self.path_data = path_data
        self.len_snippet = len_snippet
        self.mode = mode
        self.frame_no = frame_no
        self.multi_frame = multi_frame
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        if self.mode == "train":
            self.video_names = os.listdir(path_data)
            self.list_num_frame = [len(os.listdir(os.path.join(path_data, d, 'images'))) for d in self.video_names]
        elif self.mode == "val":
            self.list_num_frame = []
            for v in os.listdir(path_data):
                for i in range(0, len(os.listdir(os.path.join(path_data, v, 'images'))) - self.len_snippet,
                               self.len_snippet):
                    self.list_num_frame.append((v, i))
                if len(os.listdir(os.path.join(path_data, v, 'images'))) <= self.len_snippet:
                    self.list_num_frame.append((v, 0))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        if self.mode == "train":
            file_name = self.video_names[idx]
            start_idx = np.random.randint(0, max(1, self.list_num_frame[idx] - self.len_snippet + 1))
        elif self.mode == "val":
            (file_name, start_idx) = self.list_num_frame[idx]

        path_clip = os.path.join(self.path_data, file_name, 'images')
        path_annt = os.path.join(self.path_data, file_name, 'maps')

        clip_img = []
        clip_gt = []

        list_clips = os.listdir(path_clip)
        list_clips.sort()
        list_sal_clips = os.listdir(path_annt)
        list_sal_clips.sort()

        if len(list_sal_clips) < self.len_snippet:
            temp = [list_clips[0] for _ in range(self.len_snippet - len(list_clips))]
            temp.extend(list_clips)
            list_clips = copy.deepcopy(temp)

            temp = [list_sal_clips[0] for _ in range(self.len_snippet - len(list_sal_clips))]
            temp.extend(list_sal_clips)
            list_sal_clips = copy.deepcopy(temp)

            assert len(list_sal_clips) == self.len_snippet and len(list_clips) == self.len_snippet

        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, list_clips[start_idx + i])).convert('RGB')
            clip_img.append(self.img_transform(img))

            gt = np.array(Image.open(os.path.join(path_annt, list_sal_clips[start_idx + i])).convert('L'))
            gt = gt.astype('float')

            if self.mode == "train":
                gt = cv2.resize(gt, (384, 224))

            if np.max(gt) > 1.0:
                gt = gt / 255.0
            clip_gt.append(torch.FloatTensor(gt))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        if self.multi_frame == 0:
            gt = clip_gt[-1]
        else:
            gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))

        return clip_img, gt
