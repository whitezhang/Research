#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import utils

class AddFeatures():
    '''
    '''

    def __init__(self, debug_info=False):
        self.debug_info = debug_info
        self.reversed_table = {}
        self.bit_mod = {16:0xFFFF, 48:0xFFFFFFFFFFFF, 64:0xFFFFFFFFFFFFFFFF}

    def _is_invalided_feature(self, x):
        '''
        usage: remove 0 values from feature list
        '''
        if x == 0:
            return True
        return False

    def hash_slot_str(self, original_input, bit_mod):
        '''
        hash str
        '''
        slot_val = 0
        for s in original_input:
            slot_val += ord(s) & bit_mod
        return slot_val % self.bit_mod[16]

    def hash_slot_int(self, original_input, bit_mod):
        '''
        hash int
        '''
        return original_input % self.bit_mod[48]

    def merge_kv_slot(self, key, value):
        '''
        merge key and value, and hash into 64 bits
        '''
        return key * self.bit_mod[48] + value % self.bit_mod[64]

    # @discarded
    def _fit_transform_no_discret(self, data, dttyp):
        data_slots = []
        col_names = dttyp.names
        for name in col_names:
            fea_values = data[:].astype(dttyp, copy=False)[name]
            name_slot = self.hash_slot_str(name, self.bit_mod16)
            fea_slots = [self.hash_slot_int(x, self.bit_mod48) for x in fea_values]
            for fea in fea_slots:
                data_slots.append(self._merge_kv_slot(name_slot, fea))
        return data_slots

    def _fit_transform_discret(self, data, dttyp, discret_intervals):
        '''
        input: single data stream
        type: str
        return: {k: v, ...}
        '''
        def pick_interval(val, intervals):
            L = len(intervals)
            for i in range(L):
                if val <= intervals[i]:
                    return i
            return L - 1

        data_slots = {}
        col_names = dttyp.names
        for name in col_names:
            fea_value = data[:].astype(dttyp, copy=False)[name]
            if self._is_invalided_feature(fea_value[0]):
                continue
            # discretization for intervals
            interval_idx = pick_interval(fea_value, discret_intervals[name])
            #data_slots.append(name + BaseC.FEATUREVALUE_DELIMITER + str(interval_idx))
            data_slots[name] = interval_idx

        return data_slots

    def _fit_transform_combine(self, data):
        from copy import deepcopy
        combined_features = deepcopy(data)
        keys = data.keys()
        for i in range(len(keys)-1):
            for j in range(i+1, len(keys)-1):
                k1 = keys[i]
                k2 = keys[j]
                v1 = combined_features[k1]
                v2 = combined_features[k2]
                k1, k2, v1, v2 = utils.sortString(k1, k2, v1, v2)
                tag = utils.mergeKeyValue(k1, k2, 'combine')
                combined_features[tag] = 1
        return combined_features

    def _fit_transform_hashing(self, data):
        hashed_features = []
        for k, v in data.items():
            hk = self.hash_slot_str(k, 16)
            vk = self.hash_slot_int(v, 48)
            hash_value = self.merge_kv_slot(hk, vk)
            hashed_features.append(hash_value)
        return hashed_features

    def fit_transform(self, data, dttyp, discret_intervals=None):
        if discret_intervals == None:
            # @discarded
            #return self._fit_transform_no_discret(data, dttyp)
            pass
        else:
            data_discreted = self._fit_transform_discret(data, dttyp, discret_intervals)
            data_combined = self._fit_transform_combine(data_discreted)
            data_hashed = self._fit_transform_hashing(data_combined)
            return data_hashed

