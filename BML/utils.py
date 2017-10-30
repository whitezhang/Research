from datetime import datetime
import operator

class BaseInfo():
    COMBINE_DELIMITER= '$'
    DISCRET_DELIMITER = '^'
    FEATUREVALUE_DELIMITER = ':'

def printf(msg):
    strtowrite = "[{}] {}".format(datetime.now(), msg)
    print(strtowrite)

def sortDictByValue(x, desc):
    sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=desc)
    return sorted_x

def sortDictByKey(x, desc):
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


