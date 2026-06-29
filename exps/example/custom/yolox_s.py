#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/COCO"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.input_size = (512, 640)
        self.test_size = (512, 640)
        self.multiscale_range = 0
        
        self.num_classes = 1

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 10
        self.no_aug_epochs = 15
        
        self.save_history_ckpt = False
        self.print_interval = 100
        
        self.test_conf = 0.001
        self.nmsthre = 0.75

        self.mosaic_prob = 1.0
        self.mixup_prob = 0.0
        self.degrees = 30.0
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.shear = 2.0
