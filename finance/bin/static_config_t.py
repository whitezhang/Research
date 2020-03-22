#!/usr/bin/env python
# encoding: utf-8

import configparser

class StaticConfig():
    def __init__(self):
        # 错误代码，0为正确
        self.errno = 0

        config = configparser.ConfigParser()
        config.read('./conf/black_hole.ini')
        if 'default' in config:
            if 'stock_path' in config['default']:
                self.stock_path = config['default']['stock_path']
            else:
                logging.critical("Error loading ./conf/black_hole.ini")
                self.errno = 1

            if 'local_db_path' in config['default']:
                self.local_db_path = config['default']['local_db_path']
                pass
            else:
                logging.critical("Error loading ./conf/black_hole.ini")
                self.errno = 1

