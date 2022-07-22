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
| Dataset | MW-R | WEPDTOF |
|:----------:|:----:|:------:|
| Download Link | [96.6](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_HBCP608_Apr14_6000.ckpt) |  [97.3](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWCP608_Apr14_5500.ckpt)  |
|    1024    | [96.7](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_HBCP1024_Apr14_3000.ckpt) |  [98.1](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWCP1024_Apr14_3000.ckpt)  |

## Demo
https://user-images.githubusercontent.com/64066100/180419839-38764a0a-ff4e-4acc-83ec-60359f2c1bfe.mp4

0. Clone the repository
1. Download the [pre-trained network weights](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWHB1024_Mar11_4000.ckpt), which is trained on COCO, MW-R and HABBOF, and place it under the RAPiD/weights folder.
2. Directly run `python example.py`. Alternatively, `demo.ipynb` gives an example using jupyter notebook.

## Evaluation and Visualization
Here is an example of evaluating trained RAPiD on a single image in terms of the AP metric.

0. Modify line 41-42 to evaluate your trained weights. Default weight used `rapid_pL1_dark53_COCO608_Jun18_4000.ckpt` is trained on COCO, CEPDOF, HABBOF, MW-R and WEPDTOF.
```
rapid = Detector(model_name='rapid',
                     weights_path='./weights/rapid_pL1_dark53_COCO608_Jun18_4000.ckpt')
```
1. Run `python evaluate.py --metric AP`

The same evaluation process holds for published fisheye datasets like CEPDOF. For example, `python evaluate.py --imgs_path path/to/cepdof/Lunch1 --gt_path path/to/cepdof/annotations/Lunch1.json --metric AP`

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
