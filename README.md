<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

<div align="center"><a title="" href="https://github.com/zjykzj/YOLOv2"><img align="center" src="./imgs/YOLOv2.png" alt=""></a></div>

<p align="center">
  «YOLOv2» reproduced the paper "YOLO9000: Better, Faster, Stronger"
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-7btt"><span style="font-style:normal">Original (darknet)</span></th>
    <th class="tg-7btt"><a href="https://github.com/tztztztztz" target="_blank" rel="noopener noreferrer"><span style="text-decoration:none">tztztztztz</span></a><span style="font-weight:400">/</span><a href="https://github.com/tztztztztz/yolov2.pytorch" target="_blank" rel="noopener noreferrer">yolov2.pytorch</a></th>
    <th class="tg-7btt"><a href="https://github.com/zjykzj" target="_blank" rel="noopener noreferrer"><span style="text-decoration:none">zjykzj</span></a><span style="font-weight:400">/</span><a href="https://github.com/zjykzj/YOLOv2" target="_blank" rel="noopener noreferrer">YOLOv2</a>(This)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">train</td>
    <td class="tg-c3ow">VOC2007+2012 Train</td>
    <td class="tg-c3ow">VOC2007+2012 Train</td>
    <td class="tg-c3ow">VOC2007+2012 Train</td>
  </tr>
  <tr>
    <td class="tg-7btt">val</td>
    <td class="tg-c3ow">VOC2007 Test </td>
    <td class="tg-c3ow">VOC2007 Test </td>
    <td class="tg-c3ow">VOC2007 Test </td>
  </tr>
  <tr>
    <td class="tg-7btt">VOC AP[IoU=0.50], inference</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">76.8</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">72.7</span></td>
    <td class="tg-c3ow">51.82</td>
  </tr>
  <tr>
    <td class="tg-7btt">conf_thre</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">0.005</td>
  </tr>
  <tr>
    <td class="tg-7btt">nms_thre</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">0.45</td>
    <td class="tg-c3ow">0.45</td>
  </tr>
  <tr>
    <td class="tg-7btt">input_size</td>
    <td class="tg-c3ow">416</td>
    <td class="tg-c3ow">416</td>
    <td class="tg-c3ow">416</td>
  </tr>
</tbody>
</table>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Latest News](#latest-news)
- [Background](#background)
- [Prepare Data](#prepare-data)
    - [Pascal VOC](#pascal-voc)
- [Installation](#installation)
    - [Requirements](#requirements)
    - [Container](#container)
- [Usage](#usage)
    - [Train](#train)
    - [Eval](#eval)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Latest News

* **01/05/2023: Update. Train using VOC2007+2012 train dataset and evaluate using VOC2007 test dataset.
  ** [3146100](https://github.com/zjykzj/YOLOv2/tree/314610053a741280e0c2e205c264ce4637f3bdd8)
* **30/04/2023: Initial version. Implement YOLOv2 network, YOLOv2Loss, and related training/evaluation implementations.
  ** [32df024](https://github.com/zjykzj/YOLOv2/tree/32df02449d611e41348f5365d1c990f27f6ee4ed)

## Background

YOLOv2 has made more innovations on the basis of YOLOv1. For the network, it has created Darknet-19 and added YOLO-layer
implementation; For the loss function, it adds anchor box settings to help network training with more fine-grained
features. Compared with YOLOv1, YOLOv2 is more modern and high-performance.

This repository references many repositories implementations,
including [tztztztztz/yolov2.pytorch](https://github.com/tztztztztz/yolov2.pytorch)
and [yjh0410/yolov2-yolov3_PyTorch](https://github.com/yjh0410/yolov2-yolov3_PyTorch), as well
as [zjykzj/YOLOv3](https://github.com/zjykzj/YOLOv3).

## Prepare Data

### Pascal VOC

Use this script [voc2yolov5.py](https://github.com/zjykzj/vocdev/blob/master/py/voc2yolov5.py) to obtain Pascal VOC
datasets:

```shell
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-train -l trainval-2007 trainval-2012
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-val -l test-2007
```

Then softlink the folder where the dataset is located to the specified location:

```shell
ln -s /path/to/voc /path/to/YOLOv2/../datasets/voc
```

## Installation

### Requirements

See [NVIDIA/apex](https://github.com/NVIDIA/apex)

### Container

Development environment (Use nvidia docker container)

```shell
docker run --gpus all -it --rm -v </path/to/YOLOv2>:/app/YOLOv2 -v </path/to/voc>:/app/datasets/voc nvcr.io/nvidia/pytorch:22.08-py3
```

## Usage

### Train

* One GPU

```shell
CUDA_VISIBLE_DEVICES=0 python main_amp.py -c configs/yolov2_voc07+12.cfg --opt-level=O1 ../datasets/voc
```

### Eval

```shell
python eval.py -c configs/yolov2_voc07+12.cfg -ckpt outputs/yolov2_voc07+12/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.5190
AP for bicycle = 0.5965
AP for bird = 0.5381
AP for boat = 0.3863
AP for bottle = 0.2475
AP for bus = 0.6076
AP for car = 0.6225
AP for cat = 0.6972
AP for chair = 0.3463
AP for cow = 0.4604
AP for diningtable = 0.4382
AP for dog = 0.6924
AP for horse = 0.6076
AP for motorbike = 0.6319
AP for person = 0.5853
AP for pottedplant = 0.1913
AP for sheep = 0.5036
AP for sofa = 0.5741
AP for train = 0.5450
AP for tvmonitor = 0.5733
Mean AP = 0.5182
```

### Demo

```shell
python demo.py -c 0.5 configs/yolov2_voc07+12.cfg outputs/yolov2_voc07+12/model_best.pth.tar assets/voc2007-test/
```

<p align="left"><img src="results/voc/000237.jpg" height="240"\>  <img src="results/voc/000386.jpg" height="240"\></p>

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [tztztztztz/yolov2.pytorch](https://github.com/tztztztztz/yolov2.pytorch)
* [yjh0410/yolov2-yolov3_PyTorch](https://github.com/yjh0410/yolov2-yolov3_PyTorch)
* [zjykzj/YOLOv3](https://github.com/zjykzj/YOLOv3)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/YOLOv2/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2023 zjykzj