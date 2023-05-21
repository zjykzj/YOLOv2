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

* Train using the `VOC07+12 trainval` dataset and test using the `VOC2007 Test` dataset with an input size of `416x416`. give the result as follows

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-zkss{background-color:#FFF;border-color:inherit;color:#333;text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-fr9f{background-color:#FFF;border-color:inherit;color:#333;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-y5w1{background-color:#FFF;border-color:inherit;color:#00E;font-weight:bold;text-align:center;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-fr9f"></th>
    <th class="tg-fr9f"><span style="font-style:normal">Original (darknet)</span></th>
    <th class="tg-y5w1">tztztztztz/yolov2.pytorch</th>
    <th class="tg-y5w1">zjykzj/YOLOv2(This)</th>
    <th class="tg-y5w1">zjykzj/YOLOv2(This)</th>
    <th class="tg-c3ow">zjykzj/YOLOv2(This)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-fr9f">ARCH</td>
    <td class="tg-zkss">YOLOv2</td>
    <td class="tg-zkss">YOLOv2</td>
    <td class="tg-zkss">YOLOv2+Darknet53</td>
    <td class="tg-zkss">YOLOv2</td>
    <td class="tg-zkss">YOLOv2-tiny</td>
  </tr>
  <tr>
    <td class="tg-fr9f">VOC AP[IoU=0.50]</td>
    <td class="tg-zkss">76.8</td>
    <td class="tg-zkss">72.7</td>
    <td class="tg-zkss">76.27</td>
    <td class="tg-zkss">71.65</td>
    <td class="tg-c3ow">64.19</td>
  </tr>
</tbody>
</table>

* Train using the `COCO train2017` dataset and test using the `COCO val2017` dataset with an input size of `416x416`. give the result as follows (*Note: The results of the original paper were evaluated on the `COCO test-dev2015` dataset*)

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-zkss{background-color:#FFF;border-color:inherit;color:#333;text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-fr9f{background-color:#FFF;border-color:inherit;color:#333;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-y5w1{background-color:#FFF;border-color:inherit;color:#00E;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-9y4h{background-color:#FFF;border-color:inherit;color:#1F2328;text-align:center;vertical-align:middle}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-fr9f"></th>
    <th class="tg-fr9f"><span style="font-style:normal">Original (darknet)</span></th>
    <th class="tg-y5w1">tztztztztz/yolov2.pytorch</th>
    <th class="tg-y5w1">zjykzj/YOLOv2(This)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-fr9f">ARCH</td>
    <td class="tg-zkss">YOLOv2</td>
    <td class="tg-zkss">YOLOv2+Darknet53</td>
    <td class="tg-zkss">YOLOv2</td>
  </tr>
  <tr>
    <td class="tg-fr9f">COCO AP[IoU=0.50:0.95]</td>
    <td class="tg-zkss">21.6</td>
    <td class="tg-9y4h">25.33</td>
    <td class="tg-9y4h">21.96</td>
  </tr>
  <tr>
    <td class="tg-fr9f">COCO AP[IoU=0.50]</td>
    <td class="tg-c3ow">44.0</td>
    <td class="tg-9y4h">47.24</td>
    <td class="tg-9y4h">42.65</td>
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

* ***[2023/05/21]Upgrade Version ([v0.2.0](https://github.com/zjykzj/YOLOv2/releases/tag/v0.2.0)).  Reconstructed loss function and add Darknet53 as a backbone***
* ***[2023/05/09]Update Version ([v0.1.2](https://github.com/zjykzj/YOLOv2/releases/tag/v0.1.2)). Add COCO dataset result and update VOC dataset training results***
* ***[2023/05/03]Fix Version ([v0.1.1](https://github.com/zjykzj/YOLOv2/releases/tag/v0.1.1)). Fix target transform and update `yolov2_voc.cfg` and `yolov2-tiny_voc.cfg` training results for VOC2007 Test***
* ***[2023/05/02]Init Version ([v0.1.0](https://github.com/zjykzj/YOLOv2/releases/tag/v0.1.0)). Complete YOLOv2 training/evaluation/prediction, while providing the evaluation results of VOC2007 Test***

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
python eval.py -c configs/yolov2_d53_voc.cfg -ckpt outputs/yolov2_d53_voc/model_best.pth.tar --traversal ../datasets/voc
Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.7140
Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.7304
Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.7456
Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7627
Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7672
Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7691
Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7689
Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7704
Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7672
Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7685
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
Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.5863
Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.6102
Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.6314
Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.6419
Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.6560
Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.6570
Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.6572
Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.6511
Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.6422
Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.6329
python eval.py -c configs/yolov2_d53_coco.cfg -ckpt outputs/yolov2_d53_coco/model_best.pth.tar --traversal ../datasets/coco
Input Size：[320x320] ap50_95: = 0.2194 ap50: = 0.4139
Input Size：[352x352] ap50_95: = 0.2321 ap50: = 0.4368
Input Size：[384x384] ap50_95: = 0.2450 ap50: = 0.4553
Input Size：[416x416] ap50_95: = 0.2533 ap50: = 0.4724
Input Size：[448x448] ap50_95: = 0.2608 ap50: = 0.4844
Input Size：[480x480] ap50_95: = 0.2656 ap50: = 0.4961
Input Size：[512x512] ap50_95: = 0.2715 ap50: = 0.5060
Input Size：[544x544] ap50_95: = 0.2752 ap50: = 0.5103
Input Size：[576x576] ap50_95: = 0.2799 ap50: = 0.5208
Input Size：[608x608] ap50_95: = 0.2789 ap50: = 0.5209
python eval.py -c configs/yolov2_coco.cfg -ckpt outputs/yolov2_coco/model_best.pth.tar --traversal ../datasets/coco
Input Size：[320x320] ap50_95: = 0.1846 ap50: = 0.3692
Input Size：[352x352] ap50_95: = 0.1985 ap50: = 0.3898
Input Size：[384x384] ap50_95: = 0.2094 ap50: = 0.4078
Input Size：[416x416] ap50_95: = 0.2196 ap50: = 0.4265
Input Size：[448x448] ap50_95: = 0.2264 ap50: = 0.4400
Input Size：[480x480] ap50_95: = 0.2331 ap50: = 0.4481
Input Size：[512x512] ap50_95: = 0.2380 ap50: = 0.4590
Input Size：[544x544] ap50_95: = 0.2410 ap50: = 0.4635
Input Size：[576x576] ap50_95: = 0.2384 ap50: = 0.4668
Input Size：[608x608] ap50_95: = 0.2396 ap50: = 0.4722
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