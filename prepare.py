#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/01/15 22:43
# @Author   : WanDaoYi
# @FileName : prepare.py
# ============================================


import numpy as np
import random
from config import cfg


class DataPrepare(object):

    def __init__(self):
        self.data_path = cfg.TRAIN.DATA_PATH
        self.test_path = cfg.TEST.DATA_PATH

        self.train_data_info_path = cfg.TRAIN.TRAIN_DATA_INFO_PATH
        self.val_data_info_path = cfg.TRAIN.VAL_DATA_INFO_PATH
        self.test_data_info_path = cfg.TEST.TEST_DATA_INFO_PATH

        self.train_percent = cfg.COMMON.TRAIN_PERCENT

        pass

    # 读取数据
    def read_data(self, data_path):
        """
        :param data_path: 读取文件的路径
        :return:
        """
        time_info = []
        concate_info = []
        # 去掉数据中的 空格符 制表符 .  + 等符合。
        with open(data_path, "r") as file:
            txt_info = file.readlines()
            for data in txt_info:
                data_info = data.strip()
                data_num = data_info.split("\t")
                time_num = data_num[0]
                time_info.append(time_num)
                num_info = data_num[-1]
                num_list = num_info.split(".")
                last_num = num_list[-1]
                split_num = last_num.split("+")
                concate_num = num_list[: -1] + split_num
                concate_info.append(concate_num)

                pass

        # 判断彩票周期是否倒序，如果倒序则要处理为顺序。后面设置 label 需要用到
        time_first = int(time_info[0])
        time_last = int(time_info[-1])
        if time_first > time_last:
            time_info.reverse()
            concate_info.reverse()
            pass
        # 将分开的数据整合
        data_info = np.c_[time_info, concate_info]
        return data_info

    # 数据划分方法
    def data_split(self, data_list, split_percent):
        """
        :param data_list: 要划分的list
        :param split_percent: 划分的百分比
        :return:
        """

        # 数据的长度
        data_len = len(data_list)
        # 划分的长度
        n_split = int(split_percent * data_len)

        i = 0
        n_split_index_list = []
        while True:
            random_num = random.randint(0, data_len)

            if random_num in n_split_index_list:
                continue

            n_split_index_list.append(random_num)
            i += 1
            if i == n_split:
                break
            pass

        n_split_data_list = []
        leave_data_list = []

        for index_number, value_info in enumerate(data_list):
            if index_number in n_split_index_list:
                n_split_data_list.append(value_info)
            else:
                leave_data_list.append(value_info)
                pass
            pass

        return n_split_data_list, leave_data_list

    def add_label(self, data_list):
        # 设置 label 值。
        # 将下一期开的特别码 的 单双，设置为 这期的label 值。双为 0，单为 1
        label_list = []
        for number in data_list:
            obj_number = int(number[-1])
            if obj_number % 2 == 0:
                label_list.append("0")
            else:
                label_list.append("1")
            pass

        obj_data = np.c_[data_list[: -1], label_list[1:]]

        return obj_data
        pass

    # 保存数据
    def data_save(self, data_path, data_list):
        """
        :param data_path: 保存路径
        :param data_list: 需要保存的 python 原生 list
        :return:
        """
        # 将目标数据进行保存，每个数值，以 . 号 隔开
        data_file = open(data_path, "w")

        for data in data_list:
            data_file.write(".".join([info for info in data]))
            data_file.write("\n")
            pass

        data_file.close()
        pass

    # 数据拼接保存
    def generate_data(self):
        train_info = self.read_data(self.data_path)
        test_info = self.read_data(self.test_path)

        # 下面的方法，可以将 list 中内容的类型之间转为 int 类型
        # data_info = np.array(data_info, dtype=int)

        train_val_data = self.add_label(train_info)

        # 转为python 原生的list，下面 write 方法需要原生的 list，numpy 的list无法保存
        train_val_data = train_val_data.tolist()
        test_data_list = test_info.tolist()

        train_data_list, val_data_list = self.data_split(train_val_data, self.train_percent)

        self.data_save(self.train_data_info_path, train_data_list)
        self.data_save(self.val_data_info_path, val_data_list)
        self.data_save(self.test_data_info_path, test_data_list)

        print("data prepare already!")
        pass


if __name__ == "__main__":

    demo = DataPrepare()
    demo.generate_data()
    pass


