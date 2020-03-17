import PySimpleGUIQt as sg
import mysql.connector
import concatenate as ct
import os
import base64
# from PIL import Image
# import io

char_list_symbol= [
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
    'd_ḍād', 't_ṭā’‬','z_ẓȧ’‬', '‘ain', 'gain‬', 'fā’‬', 'qāf‬',
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


def get_base64_str_from_file(filepath):
    with open(filepath, "rb") as f:
        bytes_content = f.read()  # bytes
        bytes_64 = base64.b64encode(bytes_content)
    # return bytes_64.decode('utf-8') # bytes--->str  (remove `b`)
    return bytes_content


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
     key='_FONT_'), sg.Input(do_not_clear=True, enable_events=False, focus=False,
     key='_QS_')],
    [sg.Button('Refresh'), sg.Exit()],
    [sg.Listbox(values=char_list_symbol, size=(15, 5.5), key='_LISTBOX_',
                enable_events=True, select_mode='single', auto_size_text=True),
     sg.Listbox(values=[], size=(12, 5.5), key='_MARKERTYPE_',
                enable_events=True, default_values=None,
                select_mode='single'),
     sg.Listbox(values=list_type, size=(11, 5.5), key='_LISTTYPE_',
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
     pad=((0, 60), (0, 0))),
     sg.Text('', justification='right'),
     sg.Button('Add', size=(7, 1), key='_ADDBUTTON_',
     button_color=('white', 'green'))],
    [sg.Image(key='_IMAGE1_', filename=empty_image, visible=True,
     pad=((15, 0), (0, 10))),
     sg.Image(key='_IMAGE2_', filename=empty_image, visible=True,
     pad=((26.4, 25), (0, 10))),
     sg.Image(key='_IMAGE3_', filename=empty_image, visible=True,
     pad=((20, 0), (0, 10)))],
    [sg.Text('', key='_STATUS_'), sg.Text(db_stat, justification='right',
     text_color='black', key='_DB_')]
    
]

# sg.theme('DarkBrown1')
window = sg.Window('Alternative items', layout)


while True:
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
    # Run __TIMEOUT__ event every 50 ms
    # print(db.is_connected())
    event, values = window.Read()
    if start:
        if not db.is_connected():
            window.Element('_DB_').Update('Offline', text_color='blue')
    else:
        window.Element('_DB_').Update(value='Offline', text_color='blue')
    print(event, values)
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
    if event == '_DELBUTTON_' :
        window.Element('_LOCKDEL_').Update(False)
    if not values['_LOCKDEL_']:
        window.Element('_DELBUTTON_').Update(disabled=True,
                                             button_color=('gray', 'gray'))

    sql_result = []
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
        status_lb= status + values['_LISTBOX_'][0]
        window.Element('_STATUS_').Update(status_lb)

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
                sql_query = "SELECT * FROM dataset WHERE marker_type='" \
                            + marker_type + "' AND char_name='" + char_name \
                            + "'"
                # print(sql_query)
                db_cursor.execute(sql_query)
                sql_result = db_cursor.fetchall()
                print('IM READING DATABASE')

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
                sql_query = "SELECT * FROM dataset WHERE font_type='"\
                            + font_type + "' AND marker_type='" + marker_type\
                            + "' AND char_name='" + char_name + "'"
                # print(sql_query)
                db_cursor.execute(sql_query)
                sql_result = db_cursor.fetchall()
                # print(sql_result)
                print('IM READING DATABASE')

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
        print('IM UPDATING LISTTYPE')
    elif sql_result != [] and list_type != [] and (
            event == 'Refresh' or event == '_ADDBUTTON_' or event == '_DELBUTTON_'
            or event == '_INPUT_'):
        list_type = []
        list_id = []
        count = 0
        for res in sql_result:
            count += 1
            list_type.append(res[0] + '_' + str(count))
            list_id.append(res[6])
        window.Element('_LISTTYPE_').Update(values=list_type)
        print('IM UPDATING LISTTYPE')
    elif event == 'Refresh' or event == '_MARKERTYPE_' or delete:
        list_type = []
        list_id = []
        count = 0
        for res in sql_result:
            count += 1
            list_type.append(res[0] + '_' + str(count))
            list_id.append(res[6])
        window.Element('_LISTTYPE_').Update(values=list_type)
        print('IM UPDATING LISTTYPE')
    elif sql_result == []:
        list_type = []
        window.Element('_LISTTYPE_').Update(values=list_type)
        print('IM UPDATING LISTTYPE')

    if event == '_LISTTYPE_' and values['_LISTTYPE_'] != []:
        res_id = 100
        for x in range(len(list_id)):
            if values['_LISTTYPE_'][0] == list_type[x]:
                res_id = x
                break
        sql_query = "SELECT * FROM dataset WHERE ID='" \
                    + str(list_id[res_id]) + "'"
        # print(list_id)
        # print(sql_query)
        db_cursor.execute(sql_query)
        sql_result = db_cursor.fetchall()
        print(sql_result)
        view_image_loc = sql_result[0][4]
        print(view_image_loc) 
        window.Element('_IMAGE_').Update(filename=view_image_loc)
    elif values['_LISTTYPE_'] == []:
        window.Element('_IMAGE_').Update(filename=empty_image)

    # ADDING DATASET
    if values['_MARKERTYPE_'] != [] and values['_LISTBOX_'] != [] \
            and values['_FONT_'] != '' and values['_QS_'] != '' \
            and event == '_ADDBUTTON_':
        if font_type != '' and QS != '' and image_location != '':
            store_folder = './collection/' + font_type + '/' + char_name
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)

            cnt = len(list_type) + 1
            # best format has to be *.png
            ct.make_it_square(image_location, store_folder + '/'
                              + marker_type + '_' + str(cnt) + '.png')

            image_location = store_folder + '/' + marker_type \
                + '_' + str(cnt) + '.png'
            sql_query = "INSERT INTO dataset (font_type, char_name,\
                        marker_type, image_location, QS, ID)\
                        VALUES (%s, %s, %s, %s, %s, NULL)"
            sql_values = (font_type, char_name, marker_type,
                          image_location, QS)
            db_cursor.execute(sql_query, sql_values)
            db.commit()

            window.Element('_INPUT_').Update('')
            print('adding ' + marker_type + '_' + str(cnt) + ' to database')

    # Delete
    up = False
    if values['_LOCKDEL_']:
        up = True
    if not values['_LOCKDEL_'] and up:
        delete = True
        up = False

    if event == '_DELBUTTON_':
        res_id = 100
        for x in range(len(list_id)):
            if values['_LISTTYPE_'][0] == list_type[x]:
                res_id = x
                break
        sql_query = "SELECT * FROM dataset WHERE ID='" \
                    + str(list_id[res_id]) + "'"
        db_cursor.execute(sql_query)
        sql_result = db_cursor.fetchall()
        print(sql_result)
        delete_image_loc = sql_result[0][4]
        sql_query = "DELETE FROM dataset WHERE ID='" \
                    + str(list_id[res_id]) + "'"
        db_cursor.execute(sql_query)
        db.commit()
        os.remove(delete_image_loc)
        print('Delete id {} and {}'.format(str(list_id[res_id]),
                                           delete_image_loc))
    #     for x in range(len(char_list_symbol)):
    #         if values['_LISTBOX_'][0] == char_list_symbol[x]:
    #             # print(char_name)
    #             db_cursor = db.cursor()
    #             font_type = values['_FONT_']
    #             QS = values['_QS_']
    #             char_name = char_list_nameonly[x]
    #             marker_type = values['_MARKERTYPE_'][0]
    #             marker_name = values['_MARKERNAME_'][0]
    #             image_location = values['_INPUT_']
    #             sql_query = "SELECT * FROM dataset WHERE font_type='"\
    #                         + font_type + "' AND marker_type='" + marker_type\
    #                         + "' AND char_name='" + char_name \
    #                         + "' AND marker_name='" + marker_name + "'"
    #             # print(sql_query)
    #             db_cursor.execute(sql_query)
    #             sql_result = db_cursor.fetchall()
    #             # print(myresult)

    #             if sql_result == []:
    #                 for x in range(1, 11):
    #                     window.Element('_IMAGE' + str(x) + '_').Update(
    #                         visible=False
    #                     )
    #                 window.Element('_ADDTYPE_').Update(disabled=True)
    #                 window.Element('_ADDTYPEINPUT_').Update(disabled=True)
                    
    #                 if font_type != '' and QS != '' and image_location != '':
    #                     store_folder = './collection/' + font_type + '/'\
    #                                     + marker_name
    #                     if not os.path.exists(store_folder):
    #                         os.makedirs(store_folder)
                        
    #                     ct.make_it_square(image_location, store_folder + '/'
    #                                       + marker_type + '.png')

    #                     image_location = store_folder + '/'+ marker_type + '.png'
    #                     sql_query = "INSERT INTO dataset (font_type, char_name,\
    #                              marker_type, marker_name, image_location, QS, ID)\
    #                              VALUES (%s, %s, %s, %s, %s, %s, NULL)"
    #                     sql_values = (font_type, char_name, marker_type,
    #                                 marker_name, image_location, QS)
    #                     db_cursor.execute(sql_query, sql_values)
    #                     db.commit()

    #                     window.Element('_INPUT_').Update('')
    #                     print('adding ' + marker_name + ' to database')
    #             else:
    #                 print(sql_result)
    #                 for x in range(1, 11):
    #                     window.Element('_IMAGE' + str(x) + '_').Update(
    #                         visible=False
    #                     )
    #                 window.Element('_ADDTYPE_').Update(disabled=False)
    #                 window.Element('_ADDTYPEINPUT_').Update(disabled=False)
    #                 if event == '_ADDTYPEINPUT_' and QS != '':
                        
    #                     store_folder = './collection/' + font_type + '/'\
    #                                    + marker_name
    #                     # if not os.path.exists(store_folder):
    #                     #     os.makedirs(store_folder)
    #                     image_location = values['_ADDTYPEINPUT_']

    #                     ct.make_it_square(image_location, store_folder + '/'
    #                                       + marker_type + '_'
    #                                       + str(len(sql_result) + 1) + '.png')

    #                     image_location = store_folder + '/' + marker_type \
    #                         + '_' + str(len(sql_result) + 1) + '.png'
    #                     sql_query = "INSERT INTO dataset (font_type, char_name,\
    #                              marker_type, marker_name, image_location,\
    #                              QS, ID)\
    #                              VALUES (%s, %s, %s, %s, %s, %s, NULL)"
    #                     sql_values = (font_type, char_name, marker_type,
    #                                   marker_name, image_location, QS)
    #                     db_cursor.execute(sql_query, sql_values)
    #                     db.commit()
    #                     # window.Element('_ADDTYPEINPUT_').Update('')
    #                     # window.Element('_ADDTYPEINPUT_').Update(disabled=True)
    #                     print('adding new type ' + marker_name + ' to database')

    #                 for x in range(len(sql_result)):
    #                     img_data = get_base64_str_from_file(sql_result[x][4])
    #                     window.Element('_IMAGE' + str(x+1) + '_').Update(
    #                         data=img_data, visible=True
    #                     )





window.Close()
