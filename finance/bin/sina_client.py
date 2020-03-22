#!/usr/bin/env python
# encoding: utf-8

import sys
import requests
import logging
import json
import os
import configparser
import pandas as pd

import demjson

import static_config_t
import item_def_t
import sage_t

logging.basicConfig(level=logging.INFO)

class SinaStockInterface():
    def __init__(self, static_config):
        # http://hq.sinajs.cn/list=sh601003,sh601001
        #恒生指：<script type="text/javascript" src="http://hq.sinajs.cn/list=int_hangseng" charset="gb2312"></script>
        #日经指数：<script type="text/javascript" src="http://hq.sinajs.cn/list=int_nikkei" charset="gb2312"></script>
        #台湾加权：<script type="text/javascript" src="http://hq.sinajs.cn/list=b_TWSE" charset="gb2312"></script>
        #新加坡：<script type="text/javascript" src="http://hq.sinajs.cn/list=b_FSSTI" charset="gb2312"></script>
        #self.hq_url = 'http://hq.sinajs.cn/list='
        #self.dji = 'http://hq.sinajs.cn/list=int_dji'
        #self.nasdaq = 'http://hq.sinajs.cn/list=int_nasdaq'
        #self.history_data = 'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sz000001&scale=5&ma=5&datalen=1023'
        self.history_format_url = 'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={scale}&ma={mean}&datalen=1023'

        self.static_config = static_config

    def query_histroy_stock_data(self, code, scale, mean):
        """
        return: unicode type
        """
        # API说明: scale: 股票编号、分钟间隔（5、15、30、60）、均值（5、10、15、20、25）、查询个数点（最大值242）
        # 返回值: {u'volume': u'8089748600', u'ma_price5': 3088.25, u'high': u'3092.272', u'low': u'3075.384', u'close': u'3086.430', u'open': u'3091.493', u'day': u'2020-01-13 10:30:00', u'ma_volume5': 5826078340}
        curl_url = self.history_format_url.format(code = code, scale = scale, mean = mean)
        response = requests.get(curl_url)
        status_code = response.status_code
        text = response.text
        if status_code != 200:
            logging.warnging('Fail to query history data, error code %d' % (status_code))
        return text

    def query_all_stock_code(self):
        code_name = 'sh000001'
        code_name_list = []
        code_name_list.append(code_name)
        return code_name_list

    def query_local_stock_code(self):
        stock_item_list = []
        with open(self.static_config.stock_path) as fin:
            for line in fin:
                # code, name, type
                line = line.strip().split(' ')
                code, name, typ = line
                stock_item = item_def_t.StockItem(code, name, typ)
                stock_item_list.append(stock_item)
        return stock_item_list

    def save_to_local_file_by_pd(self, fname, response_text):
        """
        只存放抓取的结果，y预测时调整，可能有多个
        """
        fname = os.path.join(self.static_config.local_db_path, fname)
        text_json = demjson.decode(response_text)
        df = pd.DataFrame.from_dict(text_json)
        df.to_csv(fname)

    def save_to_local_file(self, fname, context):
        fname = os.path.join(self.static_config.local_db_path, fname)
        with open(fname, 'w') as fout:
            fout.write(context)
            logging.info("Saving stock data -> %s ..." % (fname))

def init():
    static_config = static_config_t.StaticConfig()
    client = SinaStockInterface(static_config)

    if static_config.errno != 0:
        return None

    return client

def main():
    client = init()
    if client == None:
        return

    stock_item_list = client.query_local_stock_code()
    for stock_item in stock_item_list:
        code = stock_item.code
        name = stock_item.name
        typ = stock_item.typ

        response_text = client.query_histroy_stock_data(code, 60, 5)
        client.save_to_local_file(code, response_text)
        #text_json = demjson.decode(response_text)
        #print text_json

def test_download():
    client = init()
    if client == None:
        return

    stock_item_list = client.query_local_stock_code()
    for stock_item in stock_item_list:
        code = stock_item.code
        name = stock_item.name
        typ = stock_item.typ

        response_text = client.query_histroy_stock_data(code, 60, 5)
        client.save_to_local_file_by_pd(code, response_text)

def test_sage():
    static_config = static_config_t.StaticConfig()
    sage = sage_t.Sage(static_config)
    sage._preprocess_df_data()

if __name__ == '__main__':
    #main()
    #test_download()
    test_sage()


