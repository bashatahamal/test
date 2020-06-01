import cv2
import os
import numpy as np


def load_image_path(loc_collection, shape):
    char_list_nameonly = [
        'Alif‬', 'Bā’', 'Tā’', 'Ṡā’‬', 'Jīm', 'h_Ḥā’‬', 'Khā’‬',
        'Dāl‬', 'Żāl‬', 'Rā’‬', 'zai‬', 'sīn‬', 'syīn‬', 's_ṣād',
        'd_ḍād', 't_ṭā’‬', 'z_ẓȧ’‬', '‘ain', 'gain‬', 'fā’‬', 'qāf‬',
        'kāf‬', 'lām‬', 'mīm‬', 'nūn‬', 'wāw‬', 'hā’‬', 'yā’‬'
    ]
    _lst_path_font = [os.path.join(loc_collection + shape, path)
                      for path in os.listdir(loc_collection + shape)]
    _lst_path_char = []

    _dict_path_char = {}
    for x in range(len(_lst_path_font)):
        for char_name in os.listdir(_lst_path_font[x]):
            _lst_temp = [os.path.join(_lst_path_font[x]+'/'+char_name, path)
                         for path in os.listdir(_lst_path_font[x]+'/'+char_name)]
            for char_num in range(len(char_list_nameonly)):
                if char_name == char_list_nameonly[char_num]:
                    if _dict_path_char.get(char_num) is None:
                        _dict_path_char[char_num] = []
                    for loc in _lst_temp:
                        split_loc = loc.split('.')
                        if split_loc[1] != 'png':
                            continue
                        # Modify here to append only png extension format
                        _dict_path_char[char_num].append(loc)

    # Check non image file
    all_images = True
    not_image = []
    for x in _dict_path_char:
        for y in _dict_path_char[x]:
            split = y.split('.')
            if split[1] != 'png':
                not_image.append(y)
                all_images = False
    if all_images:
        print('All files are .png')
    else:
        print(not_image)
    
    return _dict_path_char, all_images

loc_collection = 'Auto_Collection_Gray/'
#shape = 'Square'
shape = 'No_Margin'
path, img = load_image_path(loc_collection, shape)
size = []
id_d = []
count = 0
for x in path:
    for img in path[x]:
        image = cv2.imread(img)
        h, w, _ = image.shape
        if h > 200 or w > 200:
            size.append((h,w))
            id_d.append(count)
        count += 1
            
print(size)
print(id_d)
