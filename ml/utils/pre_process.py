#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/28 17:42
# @File     : pre_process.py

"""
import torchvision


def normal_transform():
    normal = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return normal


def data_augment_transform():
    data_augment = torchvision.transforms.Compose([
        # torchvision.transforms.RandomCrop((21, 21)),
        # torchvision.transforms.Resize((28, 28)),
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
    ])
    return data_augment
