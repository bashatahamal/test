import cv2
import os
import mysql.connector
import concatenate_gray as ct
from datetime import datetime

loc_collection = '/home/mhbrt/Desktop/Wind/Multiscale/Auto_Collection/Raw_Cut/'

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

try:
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd='',
        database='collection'
    )
    db_cursor = db.cursor()
    start = True
    db_stat = 'Connected.'
except mysql.connector.errors.ProgrammingError:
    print('Failed connecting to database')
    start = False
    db_stat = 'Not connected'
except mysql.connector.errors.InterfaceError:
    print('Connection refused database is offline')
    start = False
    db_stat = 'Not connected'
table = 'dataset_auto_gray'
char_list_nameonly = [
    'Alif‬', 'Bā’', 'Tā’', 'Ṡā’‬', 'Jīm', 'h_Ḥā’‬', 'Khā’‬',
    'Dāl‬', 'Żāl‬', 'Rā’‬', 'zai‬', 'sīn‬', 'syīn‬', 's_ṣād',
    'd_ḍād', 't_ṭā’‬', 'z_ẓȧ’‬', '‘ain', 'gain‬', 'fā’‬', 'qāf‬',
    'kāf‬', 'lām‬', 'mīm‬', 'nūn‬', 'hā’‬', 'wāw‬', 'yā’‬'
]

count = 1
for key in _dict_path_char:
    for char in _dict_path_char[key]:
        split_char = char.split('/')
        checking = split_char[10].split('.')
        if checking[1] != 'png':
            continue
        font_type = split_char[8]
        char_name = split_char[9]
        marker = split_char[10].split('_')
        marker_type = marker[0]
        QS = 'auto'
        image_location = char
        print(image_location)
        store_folder_ori = '/home/mhbrt/Desktop/Wind/Multiscale/Auto_Collection_Gray/No_Margin/' \
                            + font_type + '/' + char_name
        if not os.path.exists(store_folder_ori):
            os.makedirs(store_folder_ori)
        # store_folder_sqr = './Collection/Square/' \
        #                    + font_type + '/' + char_name
        store_folder_sqr = '/home/mhbrt/Desktop/Wind/Multiscale/Auto_Collection_Gray/Square/' \
                            + font_type + '/' + char_name
        if not os.path.exists(store_folder_sqr):
            os.makedirs(store_folder_sqr)

        img_ori, img_sqr, y1, y2, x1, x2 = ct.make_it_square(image_location)
        now = datetime.now()
        dt_string = now.strftime("%y%m%d%H%M%S")

        output_name_ori = store_folder_ori + '/' + marker_type \
                + '_' + dt_string + '.png'
        output_name_sqr = store_folder_sqr + '/' + marker_type \
            + '_' + dt_string + '.png'

        cv2.imwrite(output_name_ori, img_ori)
        cv2.imwrite(output_name_sqr, img_sqr)

        sql_query = "INSERT INTO " + table + " (font_type, char_name,\
                    marker_type, image_location, QS, dataset_type, ID)\
                    VALUES (%s, %s, %s, %s, %s ,%s, NULL)"
        sql_values = (font_type, char_name, marker_type,
                      output_name_ori, QS, 'No_Margin')
        db_cursor.execute(sql_query, sql_values)
        db.commit()
        sql_query = "INSERT INTO " + table + " (font_type, char_name,\
                    marker_type, image_location, QS, dataset_type, ID)\
                    VALUES (%s, %s, %s, %s, %s ,%s, NULL)"
        sql_values = (font_type, char_name, marker_type,
                      output_name_sqr, QS, 'Square')
        db_cursor.execute(sql_query, sql_values)
        db.commit()
        count += 1
# print(count)
