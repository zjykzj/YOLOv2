# Training Records

## About LR?

Refer to [tztztztztz/yolov2.pytorch](https://github.com/tztztztztz/yolov2.pytorch), modify the ignore threshold,
learning rate, and stop warmup:

```text
CRITERION :
    IGNORE_THRESH: 0.6
OPTIMIZER :
    TYPE: SGD
    LR: 0.0001
LR_SCHEDULER :
# warmup
    IS_WARMUP: False
```

The implementation of this
warehouse [yolov2_voc.cfg](https://github.com/zjykzj/YOLOv2/blob/14b8f5f5f22f528a8fd739f192f2095d9c8c3203/configs/yolov2_voc.cfg)
is as follows:

```text
CRITERION :
    IGNORE_THRESH: 0.7
OPTIMIZER :
    TYPE: SGD
    LR: 0.001
LR_SCHEDULER :
# warmup
    IS_WARMUP: True
```

From the training results, higher learning rates and warmup can lead to better training results:

```text
# tztztztztz/yolov2.pytorch
VOC07 metric? Yes
AP for aeroplane = 0.6820
AP for bicycle = 0.7757
AP for bird = 0.6963
AP for boat = 0.5496
AP for bottle = 0.3795
AP for bus = 0.7603
AP for car = 0.7655
AP for cat = 0.8340
AP for chair = 0.4877
AP for cow = 0.7319
AP for diningtable = 0.6933
AP for dog = 0.8097
AP for horse = 0.7900
AP for motorbike = 0.7463
AP for person = 0.6931
AP for pottedplant = 0.3693
AP for sheep = 0.6836
AP for sofa = 0.6811
AP for train = 0.7991
AP for tvmonitor = 0.6902
Mean AP = 0.6809
# v2
VOC07 metric? Yes
AP for aeroplane = 0.7318
AP for bicycle = 0.7880
AP for bird = 0.6608
AP for boat = 0.5399
AP for bottle = 0.4175
AP for bus = 0.7553
AP for car = 0.7742
AP for cat = 0.8207
AP for chair = 0.5147
AP for cow = 0.7341
AP for diningtable = 0.7119
AP for dog = 0.7991
AP for horse = 0.8170
AP for motorbike = 0.7938
AP for person = 0.7281
AP for pottedplant = 0.4169
AP for sheep = 0.6964
AP for sofa = 0.7519
AP for train = 0.7930
AP for tvmonitor = 0.6937
Mean AP = 0.6969
```

## About Loss?

Two versions of the loss function have been implemented:

1. [v1](https://github.com/zjykzj/YOLOv2/blob/314610053a741280e0c2e205c264ce4637f3bdd8/yolo/model/yololoss.py): Add
   iou_loss training even if the iou of the prediction box and annotation box is greater than the ignore threshold.
2. [v2](https://github.com/zjykzj/YOLOv2/blob/14b8f5f5f22f528a8fd739f192f2095d9c8c3203/yolo/model/yololoss.py): If the
   iou of the prediction box and annotation box is greater than the ignore threshold, it will be regarded as a positive
   sample and will not be involved in the calculation of the loss function.

From the training results, v2 is better than v1:

```text
# v1
VOC07 metric? Yes
AP for aeroplane = 0.7228
AP for bicycle = 0.7829
AP for bird = 0.6714
AP for boat = 0.5838
AP for bottle = 0.4166
AP for bus = 0.7809
AP for car = 0.7673
AP for cat = 0.8210
AP for chair = 0.4947
AP for cow = 0.7463
AP for diningtable = 0.7065
AP for dog = 0.8221
AP for horse = 0.7976
AP for motorbike = 0.7599
AP for person = 0.7291
AP for pottedplant = 0.3799
AP for sheep = 0.6884
AP for sofa = 0.7120
AP for train = 0.7657
AP for tvmonitor = 0.6979
Mean AP = 0.6923
# v2
VOC07 metric? Yes
AP for aeroplane = 0.7318
AP for bicycle = 0.7880
AP for bird = 0.6608
AP for boat = 0.5399
AP for bottle = 0.4175
AP for bus = 0.7553
AP for car = 0.7742
AP for cat = 0.8207
AP for chair = 0.5147
AP for cow = 0.7341
AP for diningtable = 0.7119
AP for dog = 0.7991
AP for horse = 0.8170
AP for motorbike = 0.7938
AP for person = 0.7281
AP for pottedplant = 0.4169
AP for sheep = 0.6964
AP for sofa = 0.7519
AP for train = 0.7930
AP for tvmonitor = 0.6937
Mean AP = 0.6969
```