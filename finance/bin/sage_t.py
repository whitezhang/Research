#!/usr/bin/env python
# encoding: utf-8

import os
import static_config_t
import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn import preprocessing

class Sage():
    """
    保守策略：预测当日最低价
    """
    def __init__(self, static_config):
        self.static_config = static_config
        # 从数据中加载的df
        self.stock_data_df = None
        # 所有训练用的x,y
        self.x_matrix = None
        self.y_daily_return = None

        # 加载本地db数据
        self._load_stock_data2df()
        # 将db中的数据处理成训练数据格式
        self._preprocess_df_data()

    def _cal_daily_return(self, df):
        for idx in range(df.shape[0] - 1):
            day1_avg = df.iloc[idx]['close'] - df.iloc[idx]['open']
            day2_avg = df.iloc[idx + 1]['close'] - df.iloc[idx]['open']
            # 单日回报率
            daily_return_value = day2_avg - day1_avg
            df.loc[idx, 'daily_return_value'] = daily_return_value
        return df

    def _save_to_local_file_to_debug(self, fname, fname_extend, content_df):
        fname = fname + '-' + fname_extend
        content_df.to_csv(fname, index=0)

    def _load_stock_data2df(self):
        for root, dirs, files in os.walk(self.static_config.local_db_path):
            content_df = None
            for name in files:
                fname = os.path.join(root, name)
                content_df = pd.read_csv(fname)
                # 计算便于定义问题的特征（回报率等）
                content_df = self._cal_daily_return(content_df)
                # 存储中间计算的结果，便于debug分析
                content_df = content_df.fillna(content_df.mean()) # 均值填充缺失值
                self._save_to_local_file_to_debug(fname, 'add_value', content_df)

                # 把所有的数据都加载到df
                if self.stock_data_df == None:
                    self.stock_data_df = content_df
                else:
                    self.stock_data_df.merge(content_df)

                # 调试用，只读一个文件
                break

    def _preprocess_df_data(self):
        self.x_matrix = np.array(self.stock_data_df.drop(['day', 'daily_return_value'], axis=1))
        self.y_daily_return = np.array(self.stock_data_df['daily_return_value'])

    def _train_test_daily_return_value(self):
        """
        单日回报率预测
        """
        x_matrix = self.x_matrix
        y = self.y_daily_return

        # model
        x_train, x_test, y_train, y_test = \
                cross_validation.train_test_split(x_matrix, y, test_size=0.2)
        clf = LinearRegression(n_jobs=-1)
        clf.fit(x_train, y_train)
        confidence = clf.score(x_test, y_test)
        print(confidence)






