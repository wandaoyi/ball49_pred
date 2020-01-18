#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/01/15 22:41
# @Author   : WanDaoYi
# @FileName : config.py
# ============================================

from easydict import EasyDict as edict
import os


__C = edict()

cfg = __C

# common options 公共配置文件
__C.COMMON = edict()
# windows 获取文件绝对路径, 方便 windows 在黑窗口 运行项目
__C.COMMON.BASE_PATH = os.path.abspath(os.path.dirname(__file__))
# # 获取当前窗口的路径, 当用 Linux 的时候切用这个，不然会报错。(windows也可以用这个)
# __C.COMMON.BASE_PATH = os.getcwd()

# 训练集，验证集，测试集占的百分比
__C.COMMON.TRAIN_PERCENT = 0.9
__C.COMMON.VAL_PERCENT = 0.1


# 模型训练配置文件
__C.TRAIN = edict()

# 是否绘制 ROC 曲线，绘制为 True
__C.TRAIN.ROC_FLAG = True

# 数据路径
__C.TRAIN.DATA_PATH = os.path.join(__C.COMMON.BASE_PATH, "dataset/train_ori_data.txt")
# 将数据转为目标数据的路径
__C.TRAIN.TRAIN_DATA_INFO_PATH = os.path.join(__C.COMMON.BASE_PATH, "info/train_data.txt")
__C.TRAIN.VAL_DATA_INFO_PATH = os.path.join(__C.COMMON.BASE_PATH, "info/val_data.txt")

# 模型保存路径
__C.TRAIN.MODEL_SAVE_PATH = os.path.join(__C.COMMON.BASE_PATH, "models/model_")


# 模型预测配置文件
__C.TEST = edict()

__C.TEST.DATA_PATH = os.path.join(__C.COMMON.BASE_PATH, "dataset/predict_data.txt")
__C.TEST.TEST_DATA_INFO_PATH = os.path.join(__C.COMMON.BASE_PATH, "info/test_data.txt")

# 使用 acc 高的模型，当模型 acc 大于 0.5 时，用 True，否则用 False
__C.TEST.ACC_FLAG = False

# 模型路径
__C.TEST.MODEL_PATH = os.path.join(__C.COMMON.BASE_PATH, "models/model_acc=0.200000.m")
# __C.TEST.MODEL_PATH = os.path.join(__C.COMMON.BASE_PATH, "models/model_acc=0.785714.m")


