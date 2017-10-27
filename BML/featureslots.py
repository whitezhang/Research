#!/usr/bin/env python
# encoding: utf-8

import numpy as np

class FeatureSlots():
    '''
    '''

    def __init__(self, debug_info=False, delimiter='\001'):
        self.debug_info = debug_info
        self.delimiter = delimiter
        self.bit_mod64 = 0xFFFFFFFFFFFFFFFF
        self.bit_mod48 = 0xFFFFFFFFFFFF
        self.bit_mod16 = 0xFFFF

    def _hash_slot_str(self, original_input, bit_mod):
        slot_val = 0
        for s in original_input:
            slot_val += ord(s) & bit_mod
        return slot_val % bit_mod

    def _hash_slot_int(self, original_input, bit_mod):
        return original_input % bit_mod

    """
    def hash_slot(self, original_input, mod=16):
        bit_mod = None
        if mod == 64:
            bit_mod= 0xFFFFFFFFFFFFFFFF
        elif mod == 48:
            bit_mod= 0xFFFFFFFFFFFF
        elif mod == 16:
            bit_mod= 0xFFFF
        else:
            print 'Please select bit size: 16, 48 or 64'
            return None

        if type(original_input) == str or type(original_input) == np.string_:
            return self._hash_slot_str(original_input, bit_mod)
        elif type(original_input) == np.int64 or type(original_input) == np.int:
            return self._hash_slot_int(original_input, bit_mod)
    """

    def _merge_kv_slot(self, key, value):
        return key * self.bit_mod48 + value % self.bit_mod64

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
        return: order follows the input order, save time for the following computing(feature combination)
        '''
        def pick_interval(val, intervals):
            L = len(intervals)
            for i in range(L):
                if val <= intervals[i]:
                    return i
            return L - 1

        data_slots = []
        col_names = dttyp.names
        for name in col_names:
            fea_value = data[:].astype(dttyp, copy=False)[name]
            if fea_value[0] == 0:
                continue
            # discretization for intervals
            interval_idx = pick_interval(fea_value, discret_intervals[name])
            # mapping into slot
            name_slot = self._hash_slot_str(name, self.bit_mod16)
            interval_slot = self._hash_slot_int(interval_idx, self.bit_mod48)
            data_slots.append(self._merge_kv_slot(name_slot, interval_slot))
        return data_slots

    def fit_transform(self, data, dttyp, discret_intervals=None):
        if discret_intervals == None:
            return self._fit_transform_no_discret(data, dttyp)
        else:
            return self._fit_transform_discret(data, dttyp, discret_intervals)

    def remove_invalided_feature(self):
        pass

    def combine_features(self, data):
        '''
        combine the features which has been hashed
        '''
        from copy import deepcopy
        combined_features = deepcopy(data)
        for i in range(len(data)-1):
            for j in range(i+1, len(data)-1):
                combined_feature = self._hash_slot_int(data[i]+data[j], self.bit_mod64)
                combined_features.append(combined_feature)
        return combined_features
