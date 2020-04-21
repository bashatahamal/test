import os
import subprocess
import sys
import glob
import match_oop
import cv2
import concatenate as ct
from datetime import datetime
import mysql.connector
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--pdf", required=True,
	            help="PDF File Location")
args = vars(ap.parse_args())
pdf_location = args["pdf"]

# Get font type
slash = pdf_location.split('/')
for k in slash:
    file_name = k.split('_')
    if len(file_name) > 1:
        for name in file_name:
            font_name = name.split('.')
            if len(font_name) > 1:
                font_type = font_name[0]
def runCommand(cmd, timeout=None, window=None):
	p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output = ''
	for line in p.stdout:
		line = line.decode(errors='replace' if (sys.version_info) < (3, 5) else 'backslashreplace').rstrip()
		output += line
		# print(line)
	retval = p.wait(timeout)
	return (retval, output)


store_folder = '/home/mhbrt/Desktop/Wind/Quran Font/Automated/' + font_type
if not os.path.exists(store_folder):
    os.makedirs(store_folder)
pdftoppm = 'pdftoppm "' +pdf_location+ '" "' + store_folder +'/'+font_type+ '" -png -rx 300 -ry 300'

print('converting...')
_,ret = runCommand(pdftoppm)
# print(ret)
print('Done')


count = 1
page_cnt = 1
char_dict = {}
page_count = {}
for imagePath in sorted(glob.glob(store_folder + "/*.png")):
    char_page_count = 0
    print(imagePath)
    original_image = cv2.imread(imagePath)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    image_proc = match_oop.ImageProcessing(original_image=original_image.copy())
    image_proc.horizontal_projection(image.copy())
    image_proc.detect_horizontal_line(
        image=original_image.copy(),
        pixel_limit_ste=3,
        pixel_limit_ets=50,
        view=False
    )
    image_proc.crop_image(input_image=original_image.copy(),
                          h_point=image_proc.start_point_h,
                          view=False)
    # print(image_proc.bag_of_h_crop)
    ordinat_1= 1
    for x in image_proc.bag_of_h_crop:
        h_image = image_proc.bag_of_h_crop[x]
        gray = cv2.cvtColor(h_image, cv2.COLOR_BGR2GRAY)
        # temp_image = cv2.adaptiveThreshold(gray, 255,
        #                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                    cv2.THRESH_BINARY, 11, 2)
        _, temp_image = cv2.threshold(gray, 127, 255,
                                      cv2.THRESH_BINARY)
        image_proc.vertical_projection(temp_image)
        image_proc.detect_vertical_line(
            image=temp_image.copy(),
            pixel_limit_ste=5,
            view=False
        )
        ordinat_2 = len(image_proc.start_point_v)-1
        for x in range(int(len(image_proc.start_point_v)/2)):
            char_dict[count] = original_image[
                image_proc.start_point_h[ordinat_1-1]:
                image_proc.start_point_h[ordinat_1],
                image_proc.start_point_v[ordinat_2-1]:
                image_proc.start_point_v[ordinat_2]
            ]  # [(y1,y2),(x1,x2)]
            ordinat_2 -= 2
            count += 1
            char_page_count += 1
        ordinat_1 += 2
    page_count[page_cnt] = char_page_count
    page_cnt += 1

# print(char_dict)
# print(page_count)
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
table = 'dataset_auto'
char_list_nameonly = [
    'Alif‬', 'Bā’', 'Tā’', 'Ṡā’‬', 'Jīm', 'h_Ḥā’‬', 'Khā’‬',
    'Dāl‬', 'Żāl‬', 'Rā’‬', 'zai‬', 'sīn‬', 'syīn‬', 's_ṣād',
    'd_ḍād', 't_ṭā’‬', 'z_ẓȧ’‬', '‘ain', 'gain‬', 'fā’‬', 'qāf‬',
    'kāf‬', 'lām‬', 'mīm‬', 'nūn‬', 'hā’‬', 'wāw‬', 'yā’‬'
]

def char_109(count):
    up = 7
    if count <= up:
        char_name = char_list_nameonly[0]
        if count <= 4:
            marker_type = 'isolated'
        else:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[1]
        i = down + 1
        b = i + 1 
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 6
    if down < count <= up:
        char_name = char_list_nameonly[2]
        i = down + 2
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[3]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[4]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[5]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[6]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 2
    if down < count <= up:
        char_name = char_list_nameonly[7]
        i = down + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= up:
            marker_type = 'end'
    down = up
    up = down + 2
    if down < count <= up:
        char_name = char_list_nameonly[8]
        i = down + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= up:
            marker_type = 'end'
    down = up
    up = down + 2
    if down < count <= up:
        char_name = char_list_nameonly[9]
        i = down + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= up:
            marker_type = 'end'
    down = up
    up = down + 2
    if down < count <= up:
        char_name = char_list_nameonly[10]
        i = down + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[11]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[12]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[13]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[14]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[15]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[16]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[17]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[18]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[19]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[20]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[21]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 6
    if down < count <= up:
        char_name = char_list_nameonly[22]
        i = down + 2
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[23]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[24]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[25]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    down = up
    up = down + 2
    if down < count <= up:
        char_name = char_list_nameonly[26]
        i = down + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= up:
            marker_type = 'end'
    down = up
    up = down + 4
    if down < count <= up:
        char_name = char_list_nameonly[27]
        i = down + 1
        b = i + 1
        m = b + 1
        if count <= i:
            marker_type = 'isolated'
        if i < count <= b:
            marker_type = 'begin'
        if b < count <= m:
            marker_type = 'middle'
        if m < count <= up:
            marker_type = 'end'
    
    return char_name, marker_type

count = 1
for x in char_dict:
    if len(char_dict) == 109:
        char_name, marker_type = char_109(count)
    # cv2.imshow('test', char_dict[x])
    print(char_name, marker_type)
    # cv2.waitKey(0)

    # img_ori, img_sqr, y1, y2, x1, x2 = ct.make_it_square(char_dict[x], True)
    img_ori, img_sqr, y1, y2, x1, x2 = ct.just_projection(char_dict[x], True)
    now = datetime.now()
    dt_string = now.strftime("%y%m%d%H%M%S")

    store_folder_ori = '/home/mhbrt/Desktop/Wind/Multiscale/Auto_Collection/No_Margin/' \
                        + font_type + '/' + char_name
    if not os.path.exists(store_folder_ori):
        os.makedirs(store_folder_ori)
    store_folder_sqr = '/home/mhbrt/Desktop/Wind/Multiscale/Auto_Collection/Square/' \
                        + font_type + '/' + char_name
    if not os.path.exists(store_folder_sqr):
        os.makedirs(store_folder_sqr)

    output_name_ori = store_folder_ori + '/' + marker_type \
        + '_' + dt_string +'_'+str(count) + '.png'
    output_name_sqr = store_folder_sqr + '/' + marker_type \
        + '_' + dt_string +'_'+str(count) + '.png'
    cv2.imwrite(output_name_ori, img_ori, [cv2.IMWRITE_PNG_BILEVEL])
    cv2.imwrite(output_name_sqr, img_sqr, [cv2.IMWRITE_PNG_BILEVEL])
    # cv2.imshow('square', img_sqr)
    # cv2.imshow('ori', img_ori)
    # cv2.waitKey(0)

    QS = 'auto'
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