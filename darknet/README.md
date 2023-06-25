
# Darknet19

## Train for Darknet19

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py --arch darknet19 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-4 --epochs 120 --opt-level O1 ./imagenet/
```

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port "31226" main_amp.py --arch darknet19 -b 256 --workers 4 --opt-level O1 --resume weights/darknet19_224/model_best.pth.tar --evaluate ./imagenet/
* Prec@1 73.980 Prec@5 91.790
```

## Train for FastDarknet19

### 448x448

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31326" main_amp.py --arch fastdarknet19 -b 128 --workers 4 --lr 0.1 --weight-decay 1e-5 --epochs 120 --opt-level O1 --input ./imagenet/
```

The training results are as follows:

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port "31226" main_amp.py --arch fastdarknet19 -b 256 --workers 4 --opt-level O1 --input --resume weights/fastdarknet19_448/model_best.pth.tar --evaluate ./imagenet/
 * Prec@1 70.776 Prec@5 89.114
```

### 224x224

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31326" main_amp.py --arch fastdarknet19 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-5 --epochs 120 --opt-level O1 ./imagenet/
```

The training results are as follows:

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port "31226" main_amp.py --arch fastdarknet19 -b 256 --workers 4 --opt-level O1 --resume weights/fastdarknet19_224/model_best.pth.tar --evaluate ./imagenet/
* Prec@1 68.686 Prec@5 88.118
```

## Base Recipe

* `Model`: 
  * Type: Darknet19
  * Activation: LeakyReLU (0.1)
* `Train`:
  * Epochs: 120
  * Hybrid train: True
  * Distributed train: True
  * GPUs: 4
  * `Data`:
    * `Dataset`: 
      * Type: ImageNet Train
    * `Transform`:
      * RandomResizedCrop: 224
      * RandomHorizontalFlip
      * RandAugment
    * `Sampler`:
      * Type: DistributedSampler
    * `Dataloader`:
      * Batch size: 256
      * Num workers: 4
  * `Criterion`: 
    * Type: LabelSmoothingLoss
    * Factor: 0.1
  * `Optimizer`: 
    * Type: SGD
    * LR: 1e-1
    * Weight decay: 
      * 1e-4 for Darknet19
      * 1e-5 for FastDarknet19
    * Momentum: 0.9
  * `Lr_Scheduler`:
    * Warmup: 5
    * MultiStep: [60, 90, 110]
* `Test`:
  * `Data`:
    * `Dataset`:
      * Type: ImageNet Val
    * `Transform`:
      * Resize: 256
      * CenterCrop: 224

## Linear FC vs. Conv FC

The implementation of the fc layer is as follows:

```python
        self.fc = nn.Sequential(
            conv_bn_act(1024, num_classes, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True,
                        act='leaky_relu'),
            nn.AdaptiveAvgPool2d((1, 1))
        )
```

The training results are as follows:

```text
* Prec@1 74.034 Prec@5 91.858
```

Another implementation is as follows:

```python
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes)
        )
```

The training results are as follows:

```text
* Prec@1 74.006 Prec@5 91.730
```