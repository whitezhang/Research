#!/usr/bin/env python
# encoding: utf-8

class FeatureSlots():
    '''
    '''

    def __init__(self, debug_info=False):
        self.debug_info = debug_info

    def hash_slots(self):
        pass

    def fit_transform(self, col_names, data, dttyp):
        for name in col_names:
            fea_values = data[:].astype(dttyp, copy=False)[name]
