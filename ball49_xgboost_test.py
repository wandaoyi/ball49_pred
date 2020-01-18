#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/01/15 22:45
# @Author   : WanDaoYi
# @FileName : ball49_xgboost_test.py
# ============================================

import numpy as np
import joblib
from config import cfg


class Ball49Test(object):

    def __init__(self):
        self.test_data_path = cfg.TEST.TEST_DATA_INFO_PATH

        self.data_time, self.data_info, self.true_label = self.read_data(self.test_data_path)

        self.acc_flag = cfg.TEST.ACC_FLAG
        self.model = joblib.load(cfg.TEST.MODEL_PATH)
        pass

    # 读取数据
    def read_data(self, data_path):
        with open(data_path, "r") as file:
            data_info = file.readlines()
            data_list = [data.strip().split(".") for data in data_info]

            data_list = np.array(data_list, dtype=int)
            print("predict_data_shape: {}".format(data_list.shape))
            # 期数
            data_time = data_list[:, : 1].T[0]
            # 开奖号码
            data_info = data_list[:, 1:]
            # 真实的特别码
            true_label = data_list[:, -1:].T[0]

            return data_time, data_info, true_label
        pass

    def predict_data(self):
        y_pred = self.model.predict(self.data_info)
        print("y_pred: {}".format(y_pred))

        pred_label = []
        # 如果是 acc > 0.5, 则直接使用预测值
        if self.acc_flag:
            for pred in y_pred:
                if pred == 1:
                    pred_label.append("单")
                else:
                    pred_label.append("双")
                pass
            pass
        # 如果 acc < 0.5，则将预测值反过来用。因为，刚好预测反了。
        else:
            for pred in y_pred:
                if pred == 1:
                    pred_label.append("双")
                else:
                    pred_label.append("单")
                pass
            pass

        true_label = []
        true_label_len = len(self.true_label)
        # print("true_label: {}".format(self.true_label[1:]))
        for index in range(1, true_label_len):
            label = self.true_label[index]
            label_str = str(label)
            if len(label_str) == 1:
                label_str = "0" + label_str
            if label % 2 == 1:
                true_label.append(label_str + "-单")
                pass
            else:
                true_label.append(label_str + "-双")
                pass

        true_label.append("坐等开奖收钱!...... ")

        time_number = self.data_time[1:]
        time_number = time_number.tolist()
        time_number.append(self.data_time[-1] + 1)

        for i in range(0, true_label_len):
            print("第 {} 期 预测结果为 ({}), 真实结果为 ({})".format(time_number[i],
                                                         pred_label[i],
                                                         true_label[i]))
            pass


if __name__ == "__main__":
    demo = Ball49Test()
    demo.predict_data()
    pass
