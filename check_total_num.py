import cv2
import os
import mysql.connector
import concatenate_gray as ct
from datetime import datetime

loc_collection = '/home/mhbrt/Desktop/Wind/Multiscale/Auto_Collection_Gray/Square/'

_lst_path_font = [os.path.join(loc_collection, path)
                  for path in os.listdir(loc_collection)]

_dict_path_char = {}
for x in range(len(_lst_path_font)):
    for char_name in os.listdir(_lst_path_font[x]):
        _lst_temp = [os.path.join(_lst_path_font[x]+'/'+char_name, path)
                     for path in os.listdir(_lst_path_font[x]+'/'+char_name)]
        if _dict_path_char.get(char_name) is None:
            _dict_path_char[char_name] = []
        for loc in _lst_temp:
            # Modify here to append only png extension format
            _dict_path_char[char_name].append(loc)

total = 0
for key in _dict_path_char:
    count = 0
    for char in _dict_path_char[key]:
        split_char = char.split('/')
        checking = split_char[10].split('.')
        if checking[1] != 'png':
            continue
        count += 1
    total += count
    print(key, count)
print('total= ', total)