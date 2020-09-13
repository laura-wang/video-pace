import os
import cv2
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor

class ucf101_pace_pretrain(Dataset):

    def __init__(self, data_list, rgb_prefix, clip_len,  max_sr, transforms_=None, color_jitter_=None): # yapf: disable:
        lines = open(data_list)
        self.rgb_lines = list(lines) * 10
        self.rgb_prefix = rgb_prefix
        self.clip_len = clip_len
        self.max_sr = max_sr
        self.toPIL = transforms.ToPILImage()
        self.transforms_ = transforms_
        self.color_jitter_ = color_jitter_

    def __len__(self):
        return len(self.rgb_lines)

    def __getitem__(self, idx):
        rgb_line = self.rgb_lines[idx].strip('\n').split()
        sample_name, action_label, num_frames = rgb_line[0], int(rgb_line[1]), int(rgb_line[2])

        rgb_dir = os.path.join(self.rgb_prefix, sample_name)
        sample_rate = random.randint(1, self.max_sr)
        start_frame = random.randint(1, num_frames)

        rgb_clip = self.loop_load_rgb(rgb_dir, start_frame, sample_rate,
                                      self.clip_len, num_frames)

        label = sample_rate - 1

        trans_clip = self.transforms_(rgb_clip)

        ## apply different color jittering for each frame in the video clip
        trans_clip_cj = []
        for frame in trans_clip:
            frame = self.toPIL(frame)  # PIL image
            frame = self.color_jitter_(frame)  # tensor [C x H x W]
            frame = np.array(frame)
            trans_clip_cj.append(frame)

        trans_clip_cj = np.array(trans_clip_cj).transpose(3, 0, 1, 2)

        return trans_clip_cj, label

    def loop_load_rgb(self, video_dir, start_frame, sample_rate, clip_len,
                      num_frames):

        video_clip = []
        idx = 0

        for i in range(clip_len):
            cur_img_path = os.path.join(
                video_dir,
                "frame" + "{:06}.jpg".format(start_frame + idx * sample_rate))

            img = cv2.imread(cur_img_path)
            video_clip.append(img)

            if (start_frame + (idx + 1) * sample_rate) > num_frames:
                start_frame = 1
                idx = 0
            else:
                idx += 1

        video_clip = np.array(video_clip)

        return video_clip


if __name__ == '__main__':

    data_list = 'D:/dataset/hmdb51/list/hmdb51_test_split1_num_frames.list'
    rgb_prefix = 'D:/dataset/hmdb51/hmdb51_img'

    transforms_ = transforms.Compose([
        ClipResize((128,171)),
        CenterCrop(112),
        RandomHorizontalFlip(0.5)

    ])

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)

    train_dataset = ucf101_pace_pretrain(data_list, rgb_prefix, clip_len=16, max_sr=4,
                                   transforms_=transforms_, color_jitter_=rnd_color_jitter)

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    for iter, sample in enumerate(dataloader):

        rgb_clip, label = sample
        print(rgb_clip.shape)
        rgb_clip = rgb_clip[0].numpy()

        print(rgb_clip.shape)

        rgb_clip = rgb_clip.transpose(1, 2, 3, 0)
        for i in range(len(rgb_clip)):
            cur_frame = rgb_clip[i]

            cv2.imshow("img", cur_frame)
            cv2.waitKey()