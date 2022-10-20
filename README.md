# Rendered_Image_Harmonization_Dataset

Welcome to the official homepage of the rendered image harmonization dataset. **RdHarmony** is a large-scale Rendered Image Harmonization dataset containing pairs of ground-truth rendered images and composite rendered images with 11 novel categories, which is useful for supervised image harmonization methods.

<!--
<img src='examples/dataset_triplet.png' align="center" width=1024>
-->

## Highlights

- 11 categories, containing "human", "bottle", "cake", "motorcycle", "cow", "couch", "apple", "car",  "chair", "dog" and "knife"
- 15000 ground-truth rendered images for "human" category and 1000 ground-truth rendered images for each other, with image resolution of 1920*1080
- maximally 225000 pairs of ground-truth rendered images and composite rendered images
- accurate foreground masks
- automatic rendered image generation and composite image generation
- 30 indoor/outdoor 3D scenes
- 50 viewpoints(2D scene) for each 3D scene
- 10 representative capture conditions for each 2D scene

## Research Paper

[Deep Image Harmonization by Bridging the Reality Gap](https://arxiv.org/pdf/2103.17104.pdf)

Junyan Cao, Wenyan Cong, Li Niu, Jianfu Zhang, Liqing Zhang

Note that in our paper, RdHarmony contains rendered training pairs with 6 novel categories. We extend it to 5 more novel categories now.

## Downloads

For "human" novel category:

- 15000 ground-truth rendered images [[Baidu_Cloud\]](https://pan.baidu.com/s/1zOzLZe5BREXN26sn9NMmxg) (access code: m4de)
- 65000 rendered image pairs used in our paper [[Baidu_Cloud\]](https://pan.baidu.com/s/1eS_XE_bTDRvdGCUg9o42RQ) (access code: vnnj)

For other 10 novel categories ("bottle", "cake", "motorcycle", "cow", "couch", "apple", "car",  "chair", "dog" and "knife"):

- 10000 ground-truth rendered images [[Baidu_Cloud\]](https://pan.baidu.com/s/1ZPHIVXGS7z4Xv5TuVyzumw) (access code: z7mb)
- 90000 rendered image pairs [[Baidu_Cloud\]](https://pan.baidu.com/s/13tsKitKzblp40BLYsKMhwg) (access code: u65p)


## Details

- ### **Ground-truth Rendered Image Generation**

  We collect 30 3D scenes from Unity Asset Store and CG websites, including outdoor scenes (e.g., raceway, downtown, street, forest) and indoor scenes (e.g., bar, stadium, gym). For each 2D scene shot in 3D scenes, we sample 10 ground-truth rendered images with 10 different capture conditions (i.e., styles), including the the night style as well as styles of Clear/PartlyCloudy/Cloudy weather at sunrise&sunset/noon/other-times. Example scenes of 11 novel categories with all 10 ground-truth rendered images are shown below. Under each time of the day except “Night”, from top to bottom, we show rendered images captured under Clear, Partly Cloudy, and Cloudy weather.

<img src='examples/groundtruth_example.jpg' align="center" width=1024>

<img src='examples/dataset_examples_r5_1.png' align="center" width=1024>

<img src='examples/dataset_examples_r5_2.png' align="center" width=1024>

- ### **Composite Rendered Image Generation**

  For each 2D scene, there are 10 ground-truth rendered images with 10 different styles, where one 3D character is treated as the foreground and its foreground mask could be obtained effortlessly using Unity3D. We could generate pairs of ground-truth rendered images and composite rendered images by randomly selecting two different images and exchanging their foregrounds. Taking "human" category for an example, the illustration of composite rendered image generation process is shown below.
<img src='examples/dataset_generation.jpg' align="center" width=1024>
