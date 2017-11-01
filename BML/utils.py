from datetime import datetime
import operator

class BaseInfo():
    COMBINE_DELIMITER= '$'
    DISCRET_DELIMITER = '^'
    FEATUREVALUE_DELIMITER = ':'

def printf(msg):
    strtowrite = "[{}] {}".format(datetime.now(), msg)
    print(strtowrite)

def sortDictByValue(x, desc=True):
    sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=desc)
    return sorted_x

def sortDictByKey(x, desc=True):
    sorted_x = sorted(x.items(), key=operator.itemgetter(0),reverse=desc)
    return sorted_x

def mergeKeyValue(k, v, mode):
    if mode == 'combine':
        return k + BaseInfo.COMBINE_DELIMITER + v
    elif mode == 'discret':
        return k + BaseInfo.COMBINE_DELIMITER + v
    elif mode == 'faturevalue':
        return l + BaseInfo.FEATUREVALUE_DELIMITER + v
    else:
        raise Exception('No function foud', mode)

def sortString(k1, k2, v1, v2):
    if k1 > k2:
        return k1, k2, v1, v2
    else:
        return k2, k1, v2, v1


