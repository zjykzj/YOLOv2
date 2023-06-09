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
    <td class="tg-zkss">76.33</td>
    <td class="tg-zkss">73.27</td>
    <td class="tg-c3ow">65.44</td>
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
    <th class="tg-y5w1">zjykzj/YOLOv2(This)</th>
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
    <td class="tg-9y4h">24.98</td>
    <td class="tg-9y4h">22.01</td>
  </tr>
  <tr>
    <td class="tg-fr9f">COCO AP[IoU=0.50]</td>
    <td class="tg-c3ow">44.0</td>
    <td class="tg-9y4h">46.85</td>
    <td class="tg-9y4h">42.70</td>
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

* ***[2023/06/28][v0.2.1](https://github.com/zjykzj/YOLOv2/releases/tag/v0.2.1). Refactor data module.***
* ***[2023/05/21][v0.2.0](https://github.com/zjykzj/YOLOv2/releases/tag/v0.2.0). Reconstructed loss function and add Darknet53 as a backbone.***
* ***[2023/05/09][v0.1.2](https://github.com/zjykzj/YOLOv2/releases/tag/v0.1.2). Add COCO dataset result and update VOC dataset training results.***
* ***[2023/05/03][v0.1.1](https://github.com/zjykzj/YOLOv2/releases/tag/v0.1.1). Fix target transform and update `yolov2_voc.cfg` and `yolov2-tiny_voc.cfg` training results for VOC2007 Test.***
* ***[2023/05/02][v0.1.0](https://github.com/zjykzj/YOLOv2/releases/tag/v0.1.0). Complete YOLOv2 training/evaluation/prediction, while providing the evaluation results of VOC2007 Test.***

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
python eval.py -c configs/yolov2_d53_voc.cfg -ckpt outputs/yolov2_d53_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes                                                                                                                                                                                                  
AP for aeroplane = 0.7804                                                                                                                                                                                          
AP for bicycle = 0.8453                                                                                                                                                                                            
AP for bird = 0.7612                                                                                                                                                                                               
AP for boat = 0.6260                                                                                                                                                                                               
AP for bottle = 0.5240                                                                                                                                                                                             
AP for bus = 0.8261                                                                                                                                                                                                
AP for car = 0.8244                                                                                                                                                                                                
AP for cat = 0.8635
AP for chair = 0.5690
AP for cow = 0.8161
AP for diningtable = 0.7046
AP for dog = 0.8470
AP for horse = 0.8398
AP for motorbike = 0.8014
AP for person = 0.7673
AP for pottedplant = 0.5069
AP for sheep = 0.7639
AP for sofa = 0.7374
AP for train = 0.8268
AP for tvmonitor = 0.7581
Mean AP = 0.7495
python eval.py -c configs/yolov2_voc.cfg -ckpt outputs/yolov2_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.7396
AP for bicycle = 0.7876
AP for bird = 0.7264
AP for boat = 0.6345
AP for bottle = 0.4606
AP for bus = 0.7885
AP for car = 0.7927
AP for cat = 0.8630
AP for chair = 0.5502
AP for cow = 0.8029
AP for diningtable = 0.7024
AP for dog = 0.8457
AP for horse = 0.8374
AP for motorbike = 0.8048
AP for person = 0.7514
AP for pottedplant = 0.4933
AP for sheep = 0.7716
AP for sofa = 0.7068
AP for train = 0.8618
AP for tvmonitor = 0.7328
Mean AP = 0.7327
python eval.py -c configs/yolov2-tiny_voc.cfg -ckpt outputs/yolov2-tiny_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.6745
AP for bicycle = 0.7511
AP for bird = 0.6245
AP for boat = 0.5421
AP for bottle = 0.3319
AP for bus = 0.7508
AP for car = 0.7413
AP for cat = 0.8123
AP for chair = 0.4276
AP for cow = 0.7286
AP for diningtable = 0.6336
AP for dog = 0.7646
AP for horse = 0.8083
AP for motorbike = 0.7378
AP for person = 0.6835
AP for pottedplant = 0.3593
AP for sheep = 0.6390
AP for sofa = 0.6519
AP for train = 0.7772
AP for tvmonitor = 0.6479
Mean AP = 0.6544
python eval.py -c configs/yolov2_d53_coco.cfg -ckpt outputs/yolov2_d53_coco/model_best.pth.tar --traversal ../datasets/coco
Input Size：[320x320] ap50_95: = 0.2162 ap50: = 0.4133
Input Size：[352x352] ap50_95: = 0.2289 ap50: = 0.4323
Input Size：[384x384] ap50_95: = 0.2386 ap50: = 0.4485
Input Size：[416x416] ap50_95: = 0.2498 ap50: = 0.4685
Input Size：[448x448] ap50_95: = 0.2596 ap50: = 0.4839
Input Size：[480x480] ap50_95: = 0.2657 ap50: = 0.4950
Input Size：[512x512] ap50_95: = 0.2699 ap50: = 0.5047
Input Size：[544x544] ap50_95: = 0.2755 ap50: = 0.5115
Input Size：[576x576] ap50_95: = 0.2767 ap50: = 0.5164
Input Size：[608x608] ap50_95: = 0.2801 ap50: = 0.5215
python eval.py -c configs/yolov2_coco.cfg -ckpt outputs/yolov2_coco/model_best.pth.tar --traversal ../datasets/coco
Input Size：[320x320] ap50_95: = 0.1862 ap50: = 0.3671
Input Size：[352x352] ap50_95: = 0.2003 ap50: = 0.3926
Input Size：[384x384] ap50_95: = 0.2087 ap50: = 0.4082
Input Size：[416x416] ap50_95: = 0.2201 ap50: = 0.4270
Input Size：[448x448] ap50_95: = 0.2291 ap50: = 0.4412
Input Size：[480x480] ap50_95: = 0.2323 ap50: = 0.4476
Input Size：[512x512] ap50_95: = 0.2394 ap50: = 0.4607
Input Size：[544x544] ap50_95: = 0.2427 ap50: = 0.4663
Input Size：[576x576] ap50_95: = 0.2439 ap50: = 0.4692
Input Size：[608x608] ap50_95: = 0.2435 ap50: = 0.4765
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