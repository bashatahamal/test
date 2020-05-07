import PySimpleGUIQt as sg
import mysql.connector
import concatenate as ct
import os
# import base64
import cv2
from datetime import datetime
# from PIL import Image
# import io

char_list_symbol = [
    'Alif          ‫ا‬',
    'Bā’           ‫ب‬',
    'Tā’           ‫ت‬',
    'Ṡā’           ‫ث‬',
    'Jīm           ‫ج‬',
    'h_Ḥā’         ‫ح‬',
    'Khā’          ‫خ‬',
    'Dāl           ‫د‬',
    'Żāl           ‫ذ‬',
    'Rā’           ‫ر‬',
    'zai           ‫ز‬',
    'sīn           ‫س‬',
    'syīn          ‫ش‬',
    's_ṣād         ‫ص‬',
    'd_ḍād         ‫ض‬',
    't_ṭā’         ‫ط‬',
    'z_ẓȧ’         ‫ظ‬',
    '‘ain          ‫ع‬',
    'gain          ‫غ‬',
    'fā’           ‫ف‬',
    'qāf           ‫ق‬',
    'kāf           ‫ك‬',
    'lām           ‫ل‬',
    'mīm           ‫م‬',
    'nūn           ‫ن‬',
    'wāw          ‫و‬',
    'hā’            ‫هـ‬',
    'yā’            ‫ي‬'
]
char_list_nameonly = [
    'Alif‬', 'Bā’', 'Tā’', 'Ṡā’‬', 'Jīm', 'h_Ḥā’‬', 'Khā’‬',
    'Dāl‬', 'Żāl‬', 'Rā’‬', 'zai‬', 'sīn‬', 'syīn‬', 's_ṣād',
    'd_ḍād', 't_ṭā’‬', 'z_ẓȧ’‬', '‘ain', 'gain‬', 'fā’‬', 'qāf‬',
    'kāf‬', 'lām‬', 'mīm‬', 'nūn‬', 'wāw‬', 'hā’‬', 'yā’‬'
]
# marker_list_full = [
#     'Fathah',
#     'Kasrah',
#     'Dhammah',
#     'Fathahtain',
#     'Kasrahtain',
#     'Dhammahtain',
#     'Tasdid:',
#     '   ^Fathah',
#     '   ^Kasrah',
#     '   ^Dhammah',
#     'Sukun',
# ]
# marker_list_full_1 = [

# 	'Fathah',
# 	'Kasrah',
# 	'Dhammah',
# 	'Fathahtain',
# 	'Kasrahtain',
# 	'Dhammahtain',
# 	'Tasdid_Fathah',
# 	'Tasdid_Kasrah',
# 	'Tasdid_Dhammah',
# 	'Sukun',
# ]


marker_type_full = ['Isolated', 'Begin', 'Middle', 'End']
marker_type_half = ['Isolated', 'End']
list_type = []
empty_image = 'empty_image.png'
table = 'dataset_auto'

# def get_base64_str_from_file(filepath):
#     with open(filepath, "rb") as f:
#         bytes_content = f.read()  # bytes
#         bytes_64 = base64.b64encode(bytes_content)
#     # return bytes_64.decode('utf-8') # bytes--->str  (remove `b`)
#     return bytes_content


def refresh_sql_result_from_DB():
    global sql_result
    global font_type
    global QS
    global char_name
    global marker_type
    global image_location
    global status_bm
    # No font type
    if values['_MARKERTYPE_'] != [] and values['_FONT_'] == '' \
            and values['_LISTBOX_'] != []:
        status_bm = status_lb + '   '\
                    + values['_MARKERTYPE_'][0]
        window.Element('_STATUS_').Update(status_bm)

        for x in range(len(char_list_symbol)):
            if values['_LISTBOX_'][0] == char_list_symbol[x]:
                # print(char_name)
                db_cursor = db.cursor()
                font_type = values['_FONT_']
                QS = values['_QS_']
                char_name = char_list_nameonly[x]
                marker_type = values['_MARKERTYPE_'][0]
                image_location = values['_INPUT_']
                sql_query = "SELECT * FROM " + table + " WHERE marker_type='" \
                            + marker_type + "' AND dataset_type='No_Margin'\
                            AND char_name='" + char_name + "'"
                # print(sql_query)
                db_cursor.execute(sql_query)
                sql_result = db_cursor.fetchall()

    # Select font type
    if values['_MARKERTYPE_'] != [] and values['_FONT_'] != '' \
            and values['_LISTBOX_'] != []:
        status_bm = status_lb + '   '\
                    + values['_MARKERTYPE_'][0] + ' - '\
                    + values['_FONT_']
        window.Element('_STATUS_').Update(status_bm)

        for x in range(len(char_list_symbol)):
            if values['_LISTBOX_'][0] == char_list_symbol[x]:
                # print(char_name)
                db_cursor = db.cursor()
                font_type = values['_FONT_']
                QS = values['_QS_']
                char_name = char_list_nameonly[x]
                marker_type = values['_MARKERTYPE_'][0]
                image_location = values['_INPUT_']
                sql_query = "SELECT * FROM " + table + " WHERE font_type='"\
                            + font_type + "' AND marker_type='" + marker_type\
                            + "' AND dataset_type='No_Margin' AND char_name='"\
                            + char_name + "'"
                # print(sql_query)
                db_cursor.execute(sql_query)
                sql_result = db_cursor.fetchall()
                # print(sql_result)

    # return sql_result, font_type, QS, char_name, marker_type, image_location


try:
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd='',
        database='collection'
    )
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

layout = [
    [sg.Text('Font:'), sg.Text('QS:')],
    [sg.Input(do_not_clear=True, enable_events=False, focus=False,
     key='_FONT_'), sg.Input(do_not_clear=True, enable_events=False,
     focus=False, key='_QS_')],
    [sg.Button('Refresh'), sg.Exit()],
    [sg.Listbox(values=char_list_symbol, size=(15, 5.5), key='_LISTBOX_',
                enable_events=True, select_mode='single', auto_size_text=True),
     sg.Listbox(values=[], size=(8, 5.5), key='_MARKERTYPE_',
                enable_events=True, default_values=None,
                select_mode='single'),
     sg.Listbox(values=list_type, size=(20, 5.5), key='_LISTTYPE_',
                enable_events=True, default_values=None,
                select_mode='single')],
    # [sg.Combo(values=char_list, default_value='Alif', key='_COMBOBOX_',
    #           enable_events=True,)],
    # [sg.Input(key='_ADDTYPEINPUT_', enable_events=True,
    #  background_color='brown', size=(10, 0.7), disabled=True),
    #  sg.FileBrowse(file_types=(("image", "*png"), ('img', "*jpg")),
    #  initial_folder=None, enable_events=True, button_text='Add Type',
    #  size=(8, 0.8), key='_ADDTYPE_', disabled=True),
    [sg.Input(key='_INPUT_', enable_events=True,
     background_color='white'),
     sg.FileBrowse(file_types=(("image", "*png"), ('img', "*jpg")),
     initial_folder=None, enable_events=True, size=(7, 0.8))],
    [sg.Button('Delete', size=(7, 1), disabled=True, key='_DELBUTTON_',
     button_color=('gray', 'gray')),
     sg.Checkbox('', key='_LOCKDEL_', enable_events=True),
     sg.Text('', justification='left'),
     sg.Image(key='_IMAGE_', filename=empty_image, visible=True,
     pad=((0, 0), (0, 0))),
     sg.Image(key='_IMAGE2_', filename=empty_image, visible=True,
     pad=((0, 0), (0, 0))),
    #  sg.Image(key='_IMAGE3_', filename=empty_image, visible=True,
    #  pad=((0, 0), (0, 0))),
     sg.Text('', justification='right'),
     sg.Checkbox('skip', key='_SKIPPROCESS_', enable_events=True),
     sg.Button('Add', size=(7, 1), key='_ADDBUTTON_',
     button_color=('white', 'green'))],
    # [sg.Image(key='_IMAGE1_', filename=empty_image, visible=True,
    #  pad=((15, 0), (0, 10))),
    #  sg.Image(key='_IMAGE3_', filename=empty_image, visible=True,
    #  pad=((20, 0), (0, 10)))],
    [sg.Text('')],
    [sg.Text('', key='_STATUS_', justification='left')],
    [sg.Text('Total char:', key='_TOTALCHAR_', justification='left'),
     sg.Text(db_stat, justification='right', text_color='black', key='_DB_')],
    [sg.Text('Last action', key='_LASTACT_', justification='center',
     text_color='blue')]
]

# sg.theme('DarkBrown1')
window = sg.Window('Alternative items', layout)

last_act = ''
while True:
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
    # Run __TIMEOUT__ event every 50 ms
    # print(db.is_connected())
    event, values = window.Read()
    if start:
        if not db.is_connected():
            window.Element('_DB_').Update('Offline', text_color='blue')
    else:
        window.Element('_DB_').Update(value='Offline', text_color='blue')
    # print(event, values)
    select_done = False
    if values['_MARKERTYPE_'] != []:
        marker_type_last_value = values['_MARKERTYPE_'][0]
    else:
        marker_type_last_value = None
        set_index = 10

    if event is None or event == 'Exit':
        break

    if event == '_LOCKDEL_' and values['_LOCKDEL_']:
        window.Element('_DELBUTTON_').Update(disabled=False,
                                             button_color=('white', 'green'))
    if event == '_DELBUTTON_':
        window.Element('_LOCKDEL_').Update(False)
    if event == '_ADDBUTTON_':
        window.Element('_SKIPPROCESS_').Update(False)
    if not values['_LOCKDEL_']:
        window.Element('_DELBUTTON_').Update(disabled=True,
                                             button_color=('gray', 'gray'))

    sql_result = []
    font_type = []
    QS = []
    char_name = []
    marker_type = []
    image_location = []
    delete = False
    # Alif
    # x = 0
    status = ''
    if event == '_LISTBOX_' and (
        values['_LISTBOX_'][0] == char_list_symbol[0]
        or values['_LISTBOX_'][0] == char_list_symbol[7]
        or values['_LISTBOX_'][0] == char_list_symbol[8]
        or values['_LISTBOX_'][0] == char_list_symbol[9]
        or values['_LISTBOX_'][0] == char_list_symbol[10]
        or values['_LISTBOX_'][0] == char_list_symbol[25]

    ):
        # view_half_radio()
        # hide_full_radio()
        for x in range(len(marker_type_half)):
            if marker_type_last_value == marker_type_half[x]:
                set_index = x
                break
            elif marker_type_last_value == 'Begin' \
                    or marker_type_last_value == 'Middle':
                set_index = 0
                break
        window.Element('_MARKERTYPE_').Update(
            values=marker_type_half, set_to_index=set_index
        )

        status_lb = status + values['_LISTBOX_'][0]
        window.Element('_STATUS_').Update(status_lb)
        # print(marker_list_full)

    elif event == '_LISTBOX_':
        # view_full_radio()
        # hide_half_radio()
        for x in range(len(marker_type_full)):
            if marker_type_last_value == marker_type_full[x]:
                set_index = x
                break
        window.Element('_MARKERTYPE_').Update(
            values=marker_type_full, set_to_index=set_index
        )
        status_lb = status + values['_LISTBOX_'][0]
        window.Element('_STATUS_').Update(status_lb)

    # Updating No font type & Select font type sql_result
    # sql_result, font_type, QS, char_name, marker_type, image_location \
    #     = refresh_sql_result_from_DB()
    refresh_sql_result_from_DB()

    # View Image
    if sql_result != [] and list_type == []:
        list_type = []
        list_id = []
        count = 0
        for res in sql_result:
            count += 1
            list_type.append(res[0] + '_' + str(count))
            list_id.append(res[6])
        window.Element('_LISTTYPE_').Update(values=list_type)

    elif sql_result != [] and list_type != [] and (
            event == 'Refresh' or event == '_ADDBUTTON_'
            or event == '_DELBUTTON_' or event == '_INPUT_'):
        list_type = []
        list_id = []
        count = 0
        for res in sql_result:
            count += 1
            list_type.append(res[0] + '_' + str(count))
            list_id.append(res[6])
        window.Element('_LISTTYPE_').Update(values=list_type)

    elif event == 'Refresh' or event == '_MARKERTYPE_' or delete \
            or event == '_DELBUTTON_':
        list_type = []
        list_id = []
        count = 0
        for res in sql_result:
            count += 1
            list_type.append(res[0] + '_' + str(count))
            list_id.append(res[6])
        window.Element('_LISTTYPE_').Update(values=list_type)

    elif sql_result == []:
        list_type = []
        window.Element('_LISTTYPE_').Update(values=list_type)

    # Showing image
    if event == '_LISTTYPE_' and values['_LISTTYPE_'] != []:
        res_id = 100
        for x in range(len(list_id)):
            if values['_LISTTYPE_'][0] == list_type[x]:
                res_id = x
                break
        sql_query = "SELECT * FROM " + table + " WHERE ID='" \
                    + str(list_id[res_id]) + "'"
        # print(list_id)
        # print(sql_query)
        db_cursor.execute(sql_query)
        sql_result_img = db_cursor.fetchall()
        # print(sql_result_img)
        view_image_loc = sql_result_img[0][3]
        selected_QS = sql_result_img[0][4]
        view_image = cv2.imread(view_image_loc)
        v_height, v_width, _ = view_image.shape 
        # print(view_image_loc)
        window.Element('_IMAGE_').Update(filename=view_image_loc)
        key = view_image_loc.split('/')
        key[7] = 'Square'
        view_image_loc = '/'.join(key)
        # print(view_image_loc)
        window.Element('_IMAGE2_').Update(filename=view_image_loc)
        key = view_image_loc.split('/')
        # key[7] = 'No_Margin (rgb)'
        # view_image_loc = '/'.join(key)
        # # print(view_image_loc)
        # window.Element('_IMAGE3_').Update(filename=view_image_loc)
        selected_QS = status_bm + '@' + selected_QS + '_size=' \
                      + str(v_width) + 'x' + str(v_height)
        window.Element('_STATUS_').Update(selected_QS)
    elif values['_LISTTYPE_'] == []:
        window.Element('_IMAGE_').Update(filename=empty_image)
        window.Element('_IMAGE2_').Update(filename=empty_image)
        # window.Element('_IMAGE3_').Update(filename=empty_image)

    # ADDING DATASET
    if values['_MARKERTYPE_'] != [] and values['_LISTBOX_'] != [] \
            and values['_FONT_'] != '' and values['_QS_'] != '' \
            and event == '_ADDBUTTON_':
        if font_type != '' and QS != '' and image_location != '':
            # Check folder
            # store_folder_ori = './Collection/No_Margin/' \
            #                    + font_type + '/' + char_name
            store_folder_ori = '/home/mhbrt/Desktop/Wind/Multiscale/Auto_Collection/No_Margin/' \
                               + font_type + '/' + char_name
            if not os.path.exists(store_folder_ori):
                os.makedirs(store_folder_ori)
            # store_folder_sqr = './Collection/Square/' \
            #                    + font_type + '/' + char_name
            store_folder_sqr = '/home/mhbrt/Desktop/Wind/Multiscale/Auto_Collection/Square/' \
                               + font_type + '/' + char_name
            if not os.path.exists(store_folder_sqr):
                os.makedirs(store_folder_sqr)
            # store_folder_rgb = './Collection/No_Margin (rgb)/' \
            #                    + font_type + '/' + char_name
            # if not os.path.exists(store_folder_rgb):
            #     os.makedirs(store_folder_rgb)
            store_folder_rc = '/home/mhbrt/Desktop/Wind/Multiscale/Auto_Collection/Raw_Cut/' \
                              + font_type + '/' + char_name
            if not os.path.exists(store_folder_rc):
                os.makedirs(store_folder_rc)

            # cnt = len(list_type) + 1
            # print(cnt)
            # best format has to be *.png
            if values['_SKIPPROCESS_']:
                img_ori, img_sqr, y1, y2, x1, x2 = ct.just_projection(image_location)
            else:
                img_ori, img_sqr, y1, y2, x1, x2 = ct.make_it_square(image_location)
            # ct.make_it_square(image_location, store_folder + '/'
            #                   + marker_type + '_' + str(cnt) + '.png')
            pixel_limit = 32
            if y2 - y1 < pixel_limit or x2 - x1 < pixel_limit:
                sg.Popup(title='Warning!', custom_text='Character is below limit',
                         keep_on_top=True)
            img_rc = cv2.imread(image_location)
            img_rgb = img_rc[y1:y2, x1:x2]
            # Write image
            now = datetime.now()
            dt_string = now.strftime("%y%m%d%H%M%S")

            output_name_ori = store_folder_ori + '/' + marker_type \
                + '_' + dt_string + '.png'
            output_name_sqr = store_folder_sqr + '/' + marker_type \
                + '_' + dt_string + '.png'
            # output_name_rgb = store_folder_rgb + '/' + marker_type \
            #     + '_' + dt_string + '.png'
            output_name_rc = store_folder_rc + '/' + marker_type \
                + '_' + dt_string + '.png'
            # print(output_name_ori)
            cv2.imwrite(output_name_ori, img_ori, [cv2.IMWRITE_PNG_BILEVEL])
            cv2.imwrite(output_name_sqr, img_sqr, [cv2.IMWRITE_PNG_BILEVEL])
            # cv2.imwrite(output_name_rgb, img_rgb, [cv2.IMWRITE_PNG_COMPRESSION])
            cv2.imwrite(output_name_rc, img_rc, [cv2.IMWRITE_PNG_COMPRESSION])

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

            # sql_query = "INSERT INTO " + table + " (font_type, char_name,\
            #             marker_type, image_location, QS, dataset_type, ID)\
            #             VALUES (%s, %s, %s, %s, %s ,%s, NULL)"
            # sql_values = (font_type, char_name, marker_type,
            #               output_name_rgb, QS, 'No_Margin (rgb)')
            # db_cursor.execute(sql_query, sql_values)
            # db.commit()

            window.Element('_INPUT_').Update('')
            last_act = ''
            last_act = 'Add ' + marker_type + '_' + dt_string \
                       + ' to database'
            print(last_act)

    # Delete
    up = False
    if values['_LOCKDEL_']:
        up = True
    if not values['_LOCKDEL_'] and up:
        delete = True
        up = False

    if event == '_DELBUTTON_':
        last_act = ''
        res_id = 100
        for x in range(len(list_id)):
            if values['_LISTTYPE_'][0] == list_type[x]:
                res_id = x
                break
        sql_query = "SELECT * FROM " + table + " WHERE ID='" \
                    + str(list_id[res_id]) + "'"
        db_cursor.execute(sql_query)
        sql_result_del = db_cursor.fetchall()
        # print(sql_result)
        delete_image_loc = sql_result_del[0][3]
        sql_query = "DELETE FROM " + table + " WHERE ID='" \
                    + str(list_id[res_id]) + "'"
        db_cursor.execute(sql_query)
        db.commit()
        os.remove(delete_image_loc)
        last_act = last_act + '\n' + 'Delete id {} and {}'.format(
            str(list_id[res_id]), delete_image_loc)
        print('Delete id {} and {}'.format(str(list_id[res_id]),
                                           delete_image_loc))

        sql_query = "DELETE FROM " + table + " WHERE ID='" \
                    + str(list_id[res_id] + 1) + "'"
        db_cursor.execute(sql_query)
        db.commit()
        key = delete_image_loc.split('/')
        key[7] = 'Square'
        delete_image_loc = '/'.join(key)
        os.remove(delete_image_loc)
        last_act = last_act + '\n' + 'Delete id {} and {}'.format(
            str(list_id[res_id] + 1), delete_image_loc)
        print('Delete id {} and {}'.format(str(list_id[res_id] + 1),
                                           delete_image_loc))

        # sql_query = "DELETE FROM " + table + " WHERE ID='" \
        #             + str(list_id[res_id] + 2) + "'"
        # db_cursor.execute(sql_query)
        # db.commit()
        # key = delete_image_loc.split('/')
        # key[7] = 'No_Margin (rgb)'
        # delete_image_loc = '/'.join(key)
        # os.remove(delete_image_loc)
        # last_act = last_act + '\n' + 'Delete id {} and {}'.format(
        #     str(list_id[res_id] + 2), delete_image_loc)
        # print('Delete id {} and {}'.format(str(list_id[res_id] + 2),
        #                                    delete_image_loc))
        key = delete_image_loc.split('/')
        key[7] = 'Raw_Cut'
        delete_image_loc = '/'.join(key)
        os.remove(delete_image_loc)

        refresh_sql_result_from_DB()
        # print(sql_result)
        list_type = []
        list_id = []
        count = 0
        for res in sql_result:
            count += 1
            list_type.append(res[0] + '_' + str(count))
            list_id.append(res[6])
        window.Element('_LISTTYPE_').Update(values=list_type)

    sql_query = "SELECT `ID` FROM " + table
    db_cursor.execute(sql_query)
    sql_result_id = db_cursor.fetchall()
    total_char = 'Total char: ' + str(int(len(sql_result_id)/2))
    window.Element('_TOTALCHAR_').Update(total_char)
    window.Element('_LASTACT_').Update(last_act)


window.Close()
