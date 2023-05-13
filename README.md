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
    <th class="tg-7btt"><a href="https://github.com/zjykzj" target="_blank" rel="noopener noreferrer"><span style="text-decoration:none">zjykzj</span></a><span style="font-weight:400">/</span><a href="https://github.com/zjykzj/YOLOv2" target="_blank" rel="noopener noreferrer">YOLOv2</a>(This)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">arch</td>
    <td class="tg-c3ow">YOLOv2</td>
    <td class="tg-c3ow">YOLOv2</td>
    <td class="tg-c3ow">YOLOv2</td>
    <td class="tg-c3ow">YOLOv2-tiny</td>
  </tr>
  <tr>
    <td class="tg-7btt">train</td>
    <td class="tg-c3ow">VOC07+12 trainval</td>
    <td class="tg-c3ow">VOC07+12 trainval</td>
    <td class="tg-c3ow">VOC07+12 trainval</td>
    <td class="tg-c3ow">VOC07+12 trainval</td>
  </tr>
  <tr>
    <td class="tg-7btt">val</td>
    <td class="tg-c3ow">VOC2007 Test </td>
    <td class="tg-c3ow">VOC2007 Test </td>
    <td class="tg-c3ow">VOC2007 Test </td>
    <td class="tg-c3ow">VOC2007 Test </td>
  </tr>
  <tr>
    <td class="tg-7btt">VOC AP[IoU=0.50]</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">76.8</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">72.7</span></td>
    <td class="tg-c3ow">71.65</td>
    <td class="tg-c3ow">63.96</td>
  </tr>
  <tr>
    <td class="tg-7btt">conf_thre</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">0.005</td>
  </tr>
  <tr>
    <td class="tg-7btt">nms_thre</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">0.45</td>
    <td class="tg-c3ow">0.45</td>
    <td class="tg-c3ow">0.45</td>
  </tr>
  <tr>
    <td class="tg-7btt">input_size</td>
    <td class="tg-c3ow">416</td>
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
  - [COCO](#coco)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Container](#container)
- [Usage](#usage)
  - [Train](#train)
  - [Eval](#eval)
  - [Demo](#demo)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Latest News

***09/05/2023: Update Version ([v0.1.2](https://github.com/zjykzj/YOLOv2/releases/tag/v0.1.2)). Add COCO dataset result and update VOC dataset training results***

***03/05/2023: Fix Version ([v0.1.1](https://github.com/zjykzj/YOLOv2/releases/tag/v0.1.1)). Fix target transform and update `yolov2_voc.cfg` and `yolov2-tiny_voc.cfg` training results for VOC2007 Test***

***02/05/2023: Init Version ([v0.1.0](https://github.com/zjykzj/YOLOv2/releases/tag/v0.1.0)). Complete YOLOv2 training/evaluation/prediction, while providing the evaluation results of VOC2007 Test***

## Background

YOLOv2 has made more innovations on the basis of YOLOv1. For the network, it has created Darknet-19 and added YOLO-layer implementation; For the loss function, it adds anchor box settings to help network training with more fine-grained features. Compared with YOLOv1, YOLOv2 is more modern and high-performance.

This repository references many repositories implementations, including [tztztztztz/yolov2.pytorch](https://github.com/tztztztztz/yolov2.pytorch) and [yjh0410/yolov2-yolov3_PyTorch](https://github.com/yjh0410/yolov2-yolov3_PyTorch), as well as [zjykzj/YOLOv3](https://github.com/zjykzj/YOLOv3).

## Prepare Data

### Pascal VOC

Use this script [voc2yolov5.py](https://github.com/zjykzj/vocdev/blob/master/py/voc2yolov5.py)

```shell
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-train -l trainval-2007 trainval-2012
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-val -l test-2007
```

Then softlink the folder where the dataset is located to the specified location:

```shell
ln -s /path/to/voc /path/to/YOLOv2/../datasets/voc
```

### COCO

Use this script [get_coco.sh](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)

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
CUDA_VISIBLE_DEVICES=0 python main_amp.py -c configs/yolov2_voc.cfg --opt-level=O1 ../datasets/voc
```

* Multi-GPUs

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "32111" main_amp.py -c configs/yolov2_voc.cfg --opt-level=O1 ../datasets/voc
```

### Eval

```shell
python eval.py -c configs/yolov2_voc.cfg -ckpt outputs/yolov2_voc/model_best.pth.tar --traversal ../datasets/voc
Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.6635
Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.6878
Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.7016
Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7165
Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7278
Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7323
Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7360
Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7376
Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7387
Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7306
python eval.py -c configs/yolov2-tiny_voc.cfg -ckpt outputs/yolov2-tiny_voc/model_best.pth.tar --traversal ../datasets/voc
Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.5841
Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.6048
Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.6227
Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.6396
Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.6420
Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.6524
Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.6527
Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.6515
Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.6390
Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.6260
python eval.py -c configs/yolov2_coco.cfg -ckpt outputs/yolov2_coco/model_best.pth.tar --traversal ../datasets/coco
Input Size：[320x320] ap50_95: = 0.1274 ap50: = 0.2838
Input Size：[352x352] ap50_95: = 0.1372 ap50: = 0.3014
Input Size：[384x384] ap50_95: = 0.1470 ap50: = 0.3192
Input Size：[416x416] ap50_95: = 0.1551 ap50: = 0.3325
Input Size：[448x448] ap50_95: = 0.1598 ap50: = 0.3426
Input Size：[480x480] ap50_95: = 0.1640 ap50: = 0.3500
Input Size：[512x512] ap50_95: = 0.1658 ap50: = 0.3548
Input Size：[544x544] ap50_95: = 0.1690 ap50: = 0.3608
Input Size：[576x576] ap50_95: = 0.1691 ap50: = 0.3621
Input Size：[608x608] ap50_95: = 0.1683 ap50: = 0.3626
```

### Demo

```shell
python demo.py -c 0.6 configs/yolov2_voc.cfg outputs/yolov2_voc/model_best.pth.tar --exp voc assets/voc2007-test/
```

<p align="left"><img src="results/voc/000237.jpg" height="240"\>  <img src="results/voc/000386.jpg" height="240"\></p>

```shell
python demo.py -c 0.6 configs/yolov2_coco.cfg outputs/yolov2_coco/model_best.pth.tar --exp coco assets/coco/
```

<p align="left"><img src="results/coco/bus.jpg" height="240"\>  <img src="results/coco/zidane.jpg" height="240"\></p>

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [tztztztztz/yolov2.pytorch](https://github.com/tztztztztz/yolov2.pytorch)
* [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [yjh0410/yolov2-yolov3_PyTorch](https://github.com/yjh0410/yolov2-yolov3_PyTorch)
* [zjykzj/YOLOv3](https://github.com/zjykzj/YOLOv3)
* [zjykzj/anchor-boxes](https://github.com/zjykzj/anchor-boxes)
* [zjykzj/vocdev](https://github.com/zjykzj/vocdev)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

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