#!/usr/bin/env python
# encoding: utf-8

import numpy as np

class FeatureSlots():
    '''
    '''

    def __init__(self, debug_info=False):
        self.debug_info = debug_info

    def _hash_slot_str(self, original_input, bit_mod):
        slot_val = 0
        for s in original_input:
            slot_val += ord(s) & bit_mod
        return slot_val % bit_mod

    def _hash_slot_int(self, original_input, bit_mod):
        slot_val = 0
        return original_input % bit_mod

    def hash_slot(self, original_input, mod=16):
        bit_mod = None
        if mod == 48:
            bit_mod= 0xFFFFFFFFFFFF
        elif mod == 16:
            bit_mod= 0xFFFF
        else:
            print 'Please select bit size: 16 or 48'
            return None

        if type(original_input) == str or type(original_input) == np.string_:
            return self._hash_slot_str(original_input, bit_mod)
        elif type(original_input) == np.int64 or type(original_input) == np.int:
            return self._hash_slot_int(original_input, bit_mod)

    def _merge_kv_slot(self, key, value):
        return key * 0xFFFFFFFFFFFF + value

    def _fit_transform_no_discret(self, data, dttyp):
        data_slots = []
        col_names = dttyp.names
        for name in col_names:
            fea_values = data[:].astype(dttyp, copy=False)[name]
            name_slot = self.hash_slot(name, 16)
            fea_slots = [self.hash_slot(x, 48) for x in fea_values]
            for fea in fea_slots:
                data_slots.append(self._merge_kv_slot(name_slot, fea))
        return data_slots

    def _fit_transform_discret(self, data, dttyp, discret_intervals):
        def pick_from_intervals(val, intervals):
            L = len(intervals)
            for i in range(L):
                if val <= intervals[i]:
                    return i
            return L - 1

        data_slots = []
        col_names = dttyp.names
        for name in col_names:
            fea_value = data[:].astype(dttyp, copy=False)[name]
            name_slot = self.hash_slot(name, 16)
            interval_idx = pick_from_intervals(fea_value, discret_intervals[name])
            interval_slot = self.hash_slot(interval_idx, 48)
            data_slots.append(self._merge_kv_slot(name_slot, interval_slot))
        return data_slots

    def fit_transform(self, data, dttyp, discret_intervals=None):
        if discret_intervals == None:
            return self._fit_transform_no_discret(data, dttyp)
        else:
            return self._fit_transform_discret(data, dttyp, discret_intervals)
