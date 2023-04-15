
# Darknet19

## Train

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py -b 256 --workers 4 --lr 0.1 --weight-decay 1e-4 --epochs 120 --opt-level O1 ./imagenet/
```

## Recipe

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
    * Weight decay: 1e-4
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