# Video_Pace

This repository contains the code for the following paper:

[Jiangliu Wang](https://scholar.google.com/citations?user=q6bsitMAAAAJ&hl=en), [Jianbo Jiao](https://jianbojiao.com/) and [Yunhui Liu](http://ri.cuhk.edu.hk/yhliu), ["Self-Supervised Video Representation Learning by Pace Prediction"](http://www.robots.ox.ac.uk/~vgg/publications/2020/Wang20/wang20.pdf), In: ECCV (2020).


* [Short Presentation](https://www.youtube.com/watch?v=wYHteK4BHlk)
* [Full Presentation](https://www.youtube.com/watch?v=LCeJYkSFXSk)
* [arXiv](https://arxiv.org/pdf/2008.05861.pdf)

---
#### Main idea:

![teaser](https://github.com/JianboJiao/video-pace/blob/master/imgs/teaser.png)

#### Framework:

![framework](https://github.com/JianboJiao/video-pace/blob/master/imgs/framework.png)


# Requirements
- pytroch >= 1.3.0
- tensorboardX
- cv2
- scipy

# Usage

## Data preparation

UCF101 dataset
- Download the original UCF101 dataset from the [official website](https://www.crcv.ucf.edu/data/UCF101.php). And then extarct RGB images from videos.
- Or direclty download the pre-processed RGB and optical flow data of UCF101 [here](https://github.com/feichtenhofer/twostreamfusion) provided by feichtenhofer.

## Pre-train

Train with pace prediction task on S3D-G, the default clip length is 64 and input video size is 224 x 224.

`python train.py --rgb_prefix RGB_DIR --gpu 0,1,2,3  --bs 32 --lr 0.001 --height 256 --width 256 --crop_sz 224 --clip_len 64`

Train with pace prediction task on c3d/r3d/r21d, the default clip length is 16 and input video size is 112 x 112.

`python train.py --rgb_prefix RGB_DIR --gpu 0 --bs 30 --lr 0.001 --model c3d/r3d/r21d --height 128 --width 171 --crop_sz 112 --clip_len 16`


## Evaluation
To be updated...

# Citation
If you find this work useful or use our code, please consider citing:

```
@InProceedings{Wang20,
  author       = "Jiangliu Wang and Jianbo Jiao and Yunhui Liu",
  title        = "Self-Supervised Video Representation Learning by Pace Prediction",
  booktitle    = "European Conference on Computer Vision",
  year         = "2020",
}
```
# Acknowlegement
Part of our codes are adapted from [S3D-G HowTO100M](https://github.com/antoine77340/S3D_HowTo100M), we thank the authors for their contributions.


