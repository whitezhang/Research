#!/usr/bin/env python
# encoding: utf-8

import os
import static_config
import pandas as pd
import numpy as np

class Sage():
    """
    保守策略：预测当日最低价
    """
    def __init__(self, static_config):
        self.static_config = static_config
        self.stock_data_json_list = []
        self.stock_data_df = None

        # 加载本地db数据
        self._load_stock_data2df()
        # 将db中的数据处理成训练数据格式
        self._preprocess_df_data()

    def _cal_daily_return(self, df):
        for idx in range(df.shape[0] - 1):
            day1_avg = df.iloc[idx]['close'] - df.iloc[idx]['open']
            day2_avg = df.iloc[idx + 1]['close'] - df.iloc[idx]['open']
            # 单日回报率
            df.loc[idx, 'daily_return_value'] = day2_avg - day1_avg
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
                self._save_to_local_file_to_debug(fname, 'add_value', content_df)

                # 把所有的数据都加载到df
                if self.stock_data_df == None:
                    self.stock_data_df = content_df
                else:
                    self.stock_data_df.merge(content_df)

                # 调试用，只读一个文件
                break

    def _preprocess_df_data(self):
        x = np.array(self.stock_data_df.drop(['day', 'daily_return_value'], axis=1))
        y_daily_return = np.array(self.stock_data_df['daily_return_value'])
