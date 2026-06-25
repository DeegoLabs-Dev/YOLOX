#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
        
        # Define yourself dataset path
        self.data_dir = "datasets/COCO"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.input_size = (512, 640)
        self.test_size = (512, 640)
        
        self.num_classes = 9

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 150
        self.data_num_workers = 4
        self.eval_interval = 5
        self.no_aug_epochs = 15
        
        self.save_history_ckpt = False
        
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        
        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
