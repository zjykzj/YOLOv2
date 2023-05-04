# Training Records

## New Loss

The loss function I implemented earlier (#b54447e) is as follows:

$$
loss = \lambda_{coord}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}1_{ij}^{obj}[(x_i - \hat{x_{i}})^{2} + (y_{i} - \hat{y_{i}})^{2} + (w_{i} - \hat{w_{i}})^{2} + (y_{i} - \hat{y_{i}})^{2})] \\ 
+\lambda_{obj}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}1_{ij}^{obj}(C_{i}-\hat{C_{i}})^{2} \\ 
+\lambda_{noobj}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}1_{ij}^{noobj}(C_{i}-\hat{C_{i}})^{2} \\
+\lambda_{cls}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}1_{ij}^{obj}softmax(pi(c))
$$

In brief, the calculation of the loss function is as follows:

$$
loss = \lambda_{coord}loss_{coord}+\lambda_{obj}loss_{obj}+\lambda_{noobj}loss_{noobj}+\lambda_{cls}loss_{cls}
$$

With reference to [tztztztztz/yolov2.pytorch](https://github.com/tztztztztz/yolov2.pytorch), the loss function is implemented (#68b66ce) as follows

$$
loss = \lambda_{coord}loss_{coord}/2.0+\lambda_{obj}loss_{obj}/2.0+\lambda_{noobj}loss_{noobj}/2.0+\lambda_{cls}loss_{cls}
$$

From the training results, new loss is better than old:

```text
# New YOLOv2Loss
Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.6452
Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.6707
Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.6852
Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7096
Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7148
Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7243
Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7255
Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7285
Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7262
Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7269
# Old YOLOv2Loss
Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.6474
Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.6699
Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.6887
Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7039
Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7138
Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7146
Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7257
Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7232
Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7265
Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7222
```

## About anchors?

In the original implementation, anchors is fixed no matter what the input size is. Anchors is always be 

```text
[[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
```

I tried another way that make anchors scaled with input size. The original anchor is relative to the image size

```text
[[0.1017, 0.133188], [0.245596, 0.308418], [0.388913, 0.622994], [0.728548, 0.372348], [0.864338, 0.769777]]
```

For example, when input size is 416x416, so the anchors can scaled by 416 / 32 = 13 times.

```python
        B, C, H, W = outputs.shape[:4]
        ...
        ...
        # broadcast anchors to all grids
        # [num_anchors] -> [num_anchors, 1, 1] -> [num_anchors, H, W]
        w_anchors = torch.broadcast_to(self.anchors[:, 0].reshape(self.num_anchors, 1, 1) * W,
                                       [self.num_anchors, H, W]).to(dtype=dtype, device=device)
        h_anchors = torch.broadcast_to(self.anchors[:, 1].reshape(self.num_anchors, 1, 1) * H,
                                       [self.num_anchors, H, W]).to(dtype=dtype, device=device)
```

Using this way to train, I found that it reduces training stability and the final accuracy is not as good as the original setting.

```text
# Fixed anchors
Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.6474
Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.6699
Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.6887
Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.7039
Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7138
Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.7146
Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.7257
Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.7232
Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.7265
Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.7222
# Scaled anchors
Input Size：[320x320] ap50_95: = -1.0000 ap50: = 0.4442
Input Size：[352x352] ap50_95: = -1.0000 ap50: = 0.6025
Input Size：[384x384] ap50_95: = -1.0000 ap50: = 0.6661
Input Size：[416x416] ap50_95: = -1.0000 ap50: = 0.6986
Input Size：[448x448] ap50_95: = -1.0000 ap50: = 0.7056
Input Size：[480x480] ap50_95: = -1.0000 ap50: = 0.6970
Input Size：[512x512] ap50_95: = -1.0000 ap50: = 0.6662
Input Size：[544x544] ap50_95: = -1.0000 ap50: = 0.6078
Input Size：[576x576] ap50_95: = -1.0000 ap50: = 0.5001
Input Size：[608x608] ap50_95: = -1.0000 ap50: = 0.3460
```

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