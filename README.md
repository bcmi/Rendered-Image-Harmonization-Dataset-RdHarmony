# Rendered_Image_Harmonization_Datasets

Welcome to the official homepage of the rendered image harmonization dataset. This repo includes details of **RHHarmony** dataset and **R10Harmony** dataset.

## RHHarmony

RHHarmony is a large-scale Rendered Human Harmonization dataset containing pairs of ground-truth rendered images and composite rendered images, which is useful for supervised image harmonization methods.

<img src='examples/dataset_examples.jpg' align="center" width=1024>



### Highlights

+ 15,000 ground-truth rendered images with image resolution of 1920*1080

+ maximally 135,000 pairs of ground-truth rendered images and composite rendered images

+ accurate foreground masks

+ automatic rendered image generation and composite image generation

+ 30 indoor/outdoor 3D scenes

+ 50 viewpoints(2D scene) for each 3D scene

+ 10 representative capture conditions for each 2D scene

  

### Research Paper
[Deep Image Harmonization by Bridging the Reality Gap](https://arxiv.org/pdf/2103.17104.pdf)

Wenyan Cong, Junyan Cao, Li Niu, Jianfu Zhang, Xuesong Gao, Zhiwei Tang, Liqing Zhang



### Downloads

+ 15000 ground-truth rendered images [[Baidu_Cloud]]() (access code: ) [[Alternative_address]]()
+ 65000 rendered image pairs used in our paper [[Baidu_Cloud]]() (access code: ) [[Alternative_address]]()



### Details


+ #### **Ground-truth Rendered Image Generation**

  We collect 30 3D scenes from Unity Asset Store and CG websites, including outdoor scenes (e.g., raceway, downtown, street, forest) and indoor scenes (e.g., bar, stadium, gym). For each 2D scene shot in 3D scenes, we sample 10 ground-truth rendered images with 10 different capture conditions (i.e., styles), including the the night style as well as styles of Clear/PartlyCloudy/Cloudy weather at sunrise&sunset/noon/other-times. Example scenes with all 10 ground-truth rendered images are shown below. The left four columns are outdoor scenes (raceway, downtown, street, and forest) and the right two columns are indoor scenes (bar and stadium). Under each time of the day except “Night”, from top to bottom, we show rendered images captured under Clear, Partly Cloudy, and Cloudy weather.

<img src='examples/groundtruth_example.jpg' align="center" width=900>



+ #### **Composite Rendered Image Generation**
  
   For each 2D scene, there are 10 ground-truth rendered images with 10 different styles, where one person is treated as the foreground and its foreground mask could be obtained effortlessly using Unity3D. We could generate pairs of ground-truth rendered images and composite rendered images by randomly selecting two different images and exchanging their foregrounds. The illustration of composite rendered image generation process is shown below.

<img src='examples/dataset_generation.jpg' align="center" width=600>

## R10Harmony

Following the same procedure in [Details](#Details), we extend RHHarmony to **10** more common categories, including **car, motorbike, dog, cow, bottle, knife, apple, cake, chair, and sofa**, and construct another large rendered image harmonization dataset R10Harmony.

<img src='examples/R10Harmony_examples.jpg' align="center" width=1024>

### Highlights

+ 10,000 ground-truth rendered images with image resolution of 1920\*1080

+ maximally 90,000 pairs of ground-truth rendered images and composite rendered images

+ accurate foreground masks

+ automatic rendered image generation and composite image generation

+ 20 indoor/outdoor 3D scenes, 2 scenes for each category

+ 50 viewpoints(2D scene) for each 3D scene

+ 10 representative capture conditions for each 2D scene

### Downloads

[[Baidu_Cloud]](https://pan.baidu.com/s/1hwPGcllgiulN2E_qeLURsw) (access code: im7g) [[Alternative_address]]()