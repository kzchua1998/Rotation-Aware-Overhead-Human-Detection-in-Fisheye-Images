# Transfer Learning of RAPiD Model on MW-R and WEBDTOF 

**RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images** <br />

## Installation
**Requirements**:
- PyTorch >= 1.0. Installation instructions can be found at https://pytorch.org/get-started/locally/
- opencv-python
- [pycocotools](https://github.com/cocodataset/cocoapi) (for Windows users, please refer to [this repo](https://github.com/philferriere/cocoapi))
- tqdm
- tensorboard (optional, only for training)

## Datasets
The pre-trained model from RAPiD authors is further trained on MW-R and WEPDTOF datasets for 4000 iterations.
- MW-R [[Download Here](https://vip.bu.edu/projects/vsns/cossy/datasets/mw-r/)]
- WEPDTOF [[Download Here](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/)]

<p align="center">
<img src="https://vip.bu.edu/files/2021/07/wepdof_samples.png" width="500" height="500">
</p>

It should be noted that MW-R only provides [raw videos](https://www.youtube.com/playlist?list=PLKjRNrBNA-nzzv4KqqdeMHMtq26kue5ZR) and their corresponding annotations in COCO json format. Therefore, further processing is necessary to convert the videos into frames and name them appropriately.


**Instructions**:
- Convert the MW Train Set videos into frames. 

- `MW-18Mar-2` video should contains 297 frames, 788 frames for `MW-18Mar-3` video, and 451 frames each for other videos. 

- In total, there should be 297 + 788 + 17*451 = 8752 frames.

- Rename the MW frames using the following file names: `Mar#_******.jpg`, where # is the video number as in the original MW dataset, and ****** is the frame number in that video but zero-padded to 6 digits. 

For example, the first frame of the `MW-18Mar-3` video will be `Mar3_000001.jpg`, and the 10th frame of the `MW-18Mar-12` video will be `Mar12_000010.jpg`.

## Demo
The video was taken approximately 17 feet from the ground near Kolej 10, Universiti Teknologi Malaysia.

https://user-images.githubusercontent.com/64066100/180419839-38764a0a-ff4e-4acc-83ec-60359f2c1bfe.mp4

2. Directly run `python example.py`. Alternatively, `demo.ipynb` gives an example using jupyter notebook.

## Evaluation and Visualization
Here is an example of evaluating trained RAPiD on a single image in terms of the AP metric. Skip to step 2 for visualization only.

0. Modify line 41-42 to evaluate your trained weights. Default weight used is `rapid_pL1_dark53_COCO608_Jun18_4000.ckpt`.
```
rapid = Detector(model_name='rapid',
                     weights_path='./weights/rapid_pL1_dark53_COCO608_Jun18_4000.ckpt')
```
1. Run `python evaluate.py --metric AP`
2. Visualize the result by running `python demo.py` or using `demo.ipynb` provided.

## Training on COCO json data format
0. Download [the Darknet-53 weights](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/dark53_imgnet.pth) by RAPiD authors which is pre-trained on ImageNet. This is identical to the one provided by the official YOLOv3 authors but in PyTorch format.
1. Place the weights file under the /weights folder;
2. Download the COCO dataset and put it at `path/to/COCO`
3. Modify line 57-60 in train.py according to the following code snippet.
```
if args.dataset == 'COCO':
    train_img_dir = 'path/to/img/train'
    train_json = 'path/to/json/train.json'
    val_img_dir = 'path/to/img/val'
    val_json = 'path/to/json/val.json'
```
4. Run `python train.py --model rapid_pL1 --dataset COCO --batch_size 2`. Set the largest possible batch size that can fit in the GPU memory.

## Citation
```
Z. Duan, M.O. Tezcan, H. Nakamura, P. Ishwar and J. Konrad, 
“RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images”, 
in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 
Omnidirectional Computer Vision in Research and Industry (OmniCV) Workshop, June 2020.
```
