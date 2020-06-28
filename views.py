from flask import Response
import subprocess
import pickle
from flask import jsonify, make_response
import time
from Multiscale import app
from flask import render_template, request, redirect, url_for, send_from_directory
import os
import shutil
import cv2
import sys
from datetime import datetime
sys.path.insert(1, '/home/mhbrt/Desktop/Wind/Multiscale/')
import complete_flow as flow
import write_html

# model_name = '/home/mhbrt/Desktop/Wind/Multiscale/Colab/best_model_DenseNet_DD.pkl'
# model = pickle.load(open(model_name, 'rb'))
from tensorflow.keras.models import model_from_json
json_file = open('/home/mhbrt/Desktop/Wind/Multiscale/Colab/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/home/mhbrt/Desktop/Wind/Multiscale/Colab/model.h5")
print('_LOAD MODEL DONE_')
# model = ''

# Global Variable
from_sketch_button = False
list_image_files = []
global_count = 0
setting = {}
req = {}
temp_object = []
imagelist_template_matching_result = []
imagelist_template_scale_visualize = []
imagelist_visualize_white_block = []
final_image_result = 0
normal_processing_result = 0
crop_ratio_processing_result = 0
imagelist_horizontal_line_by_eight_conn = 0
listof_imagelist_template_matching_result = []
listof_imagelist_template_scale_visualize = []
listof_imagelist_visualize_white_block = []
listof_final_image_result = []
listof_normal_processing_result = []
listof_crop_ratio_processing_result = []
listof_imagelist_horizontal_line_by_eight_conn = []
listof_prediction_result = []
listof_image_v_checking = []
listof_char_recog = []
listof_input_image = []
listof_max_id = []
listof_detected_and_segmented = []
processed_font = []

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
def index():
    # return render_template('public/index.html')
    return redirect('/sketch')


@app.route('/stream')
def yieldd():
    def inner():
        proc = subprocess.Popen(
            # call something with a lot of output so we can see it
            ["python", "-u", app.root_path + "/count_timer.py"],
            stdout=subprocess.PIPE,
            # universal_newlines=True
        )

        for line in iter(proc.stdout.readline, b''):
        # print(line.decode("utf-8"))
            # yield line.decode("utf-8").rstrip() + '<br/>\n'
            yield line.decode("utf-8").rstrip() + '$'

    # text/html is required for most browsers to show th$
    return Response(inner(), mimetype='text/event-stream')
    # return Response(inner(), mimetype='text/html')

import flask
@app.route('/processing_page')
def get_page():
    # if from_sketch_button:
    # return flask.send_file('templates/public/page.html')
    return render_template('public/page.html')


def detected_and_segmented_image(save_state, bw_image):
    bw_copy = bw_image.copy()
    for x in save_state:
        if len(save_state[x]) < 2:
            continue
        cv2.rectangle(bw_copy, (save_state[x][3][0], save_state[x][3][1]),
                      (save_state[x][3][2], save_state[x][3][3]),
                      (0, 0, 0), 1)
        cv2.rectangle(bw_copy, save_state[x][6][0], save_state[x][6][1],
                      (200, 150, 0), 1)
    
    return bw_copy

@app.route('/processing')
def processing():
    global from_sketch_button
    req['A message from python'] = 'Initialiation'
    if from_sketch_button:
        # resetting all global variable
        from_sketch_button = False
        def runner():
            global list_image_files
            global setting
            global req
            # global temp_object
            global listof_imagelist_template_matching_result
            global listof_imagelist_template_scale_visualize
            global listof_imagelist_visualize_white_block
            global listof_final_image_result
            global listof_normal_processing_result
            global listof_crop_ratio_processing_result
            global listof_imagelist_horizontal_line_by_eight_conn
            global listof_prediction_result
            global listof_image_v_checking
            global listof_char_recog
            global listof_input_image
            global listof_max_id
            global listof_detected_and_segmented
            global processed_font
            global model
            global global_count
            font_folder = ['AlKareem', 'AlQalam', 'KFGQPC', 'LPMQ', 'PDMS',
                        'amiri', 'meQuran', 'norehidayat', 'norehira', 'norehuda']
            global processed_font
            for global_count in range(len(list_image_files)):
                
                font_list = 0
                loc_path = 0
                numfiles = 0
                imagePath = 0
                markerPath = 0
                img = 0
                gray = 0
                font_object = 0
                imagelist_horizontal_line_by_eight_conn = []
                imagelist_template_matching_result= []
                imagelist_template_scale_visualize = []
                imagelist_visualize_white_block = []
                temp_object = []
                check_image_size = 0
                final_image_result = []
                pred_result = []
                image_v_checking = []
                char_recog = []
                input_image = []
                max_id = ''
                ds_image = []
                processed_font = []
                save_state = 0
                normal_processing_result = []
                crop_ratio_processing_result = []
                bw_method = 0   # 0 = Otshu's threshold
                                # 1 = Simple threshold
                                # 2 = Adaptive mean of neighbourhood area
                                # 3 = Adaptive weighted sum of neighb area 
                

                numfiles = len(list_image_files)
                imagePath = list_image_files[global_count]
                markerPath = app.config['MARKER_ROOT']
                img = cv2.imread(imagePath)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if bw_method == 0:
                # Otsu threshold
                    print('using otsu threshold')
                    _, bw_image = cv2.threshold(gray, 0, 255,
                                                cv2.THRESH_BINARY
                                                + cv2.THRESH_OTSU)
                if bw_method == 1:
                # Simple threshold
                    print('using simple threshold')
                    _, bw_image = cv2.threshold(gray, 127, 255,
                                                cv2.THRESH_BINARY)
                if bw_method == 2:
                # Adaptive threshold value is the mean of neighbourhood area
                    print('using adaptive mean threshold')
                    bw_image = cv2.adaptiveThreshold(gray, 255,
                                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
                if bw_method == 3:
                # Adaptive threshold value is the weighted sum of neighbourhood
                # values where weights are a gaussian window
                    print('using adaptive sum threshold')
                    bw_image = cv2.adaptiveThreshold(gray, 255,
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 11, 2)
                input_image = [img, gray, bw_image]
                font_list, loc_path = flow.font_list(
                    imagePath=imagePath, image=gray, setting=setting, markerPath=markerPath)
                local_count = -1
                for font_object in font_list:
                    local_count += 1
                    if setting[font_folder[local_count]][2] == 'true':
                        print('skipping ', font_folder[local_count])
                        continue
                    else:
                        processed_font.append(font_folder[local_count])
                    yield str(global_count+1)+'_'+font_folder[local_count]+'_'+ str(numfiles) + '$'
                    skip_marker = []
                    for sm in setting[font_folder[local_count]][3]:
                        split = sm.split(' ')
                        marker = ''
                        for x in range(len(split)):
                            if x < len(split)-1:
                                marker += split[x].lower() + '_'
                            else:
                                marker += split[x].lower()
                        skip_marker.append(marker)
                    font_object.run(bw_method=bw_method, skip_marker=skip_marker)
                    imagelist_template_scale_visualize.append(font_object.imagelist_visualize)
                    imagelist_visualize_white_block.append(font_object.imagelist_visualize_white_blok)
                    temp_object.append(font_object.get_object_result())
                    imagelist_template_matching_result.append(font_object.display_marker_result(img, skip_marker))
                # print(temp_object)
                yield str(global_count+1) + '_Doing BIG BLOK_' + str(numfiles) + '$'
                max_id = flow.most_marker(temp_object)
                if max_id is None:
                    yield str(global_count+1) + '_empty_' + str(numfiles) + '$'
                    listof_imagelist_template_matching_result.append(imagelist_template_matching_result)
                    listof_imagelist_template_scale_visualize.append(imagelist_template_scale_visualize)
                    listof_imagelist_visualize_white_block.append(imagelist_visualize_white_block)
                    listof_final_image_result.append(final_image_result)
                    listof_prediction_result.append(pred_result)
                    listof_image_v_checking.append(image_v_checking)
                    listof_char_recog.append(char_recog)
                    listof_input_image.append(input_image)
                    listof_max_id.append([])
                    listof_detected_and_segmented.append([])
                    listof_normal_processing_result.append(normal_processing_result)
                    listof_crop_ratio_processing_result.append(crop_ratio_processing_result)
                    listof_imagelist_horizontal_line_by_eight_conn.append(imagelist_horizontal_line_by_eight_conn)
                else:
                    yield str(global_count+1) + '_Image Processing_' + str(numfiles) + '$'
                    save_state, normal_processing_result, crop_ratio_processing_result,\
                    imagelist_horizontal_line_by_eight_conn, image_v_checking\
                         = flow.define_normal_or_crop_processing(
                            imagePath, temp_object, max_id, font_object, font_list, bw_method
                         )
                    yield  str(global_count+1) + '_Recognition_' + str(numfiles) + '$'
                    final_image_result, pred_result, char_recog = flow.character_recognition(
                        save_state, imagePath, model
                    )
                    yield  str(global_count+1) + '_Saving Result_' + str(numfiles) + '$'
                    ds_image = detected_and_segmented_image(save_state, bw_image)
                    listof_imagelist_template_matching_result.append(imagelist_template_matching_result)
                    listof_imagelist_template_scale_visualize.append(imagelist_template_scale_visualize)
                    listof_imagelist_visualize_white_block.append(imagelist_visualize_white_block)
                    listof_final_image_result.append(final_image_result)
                    listof_prediction_result.append(pred_result)
                    listof_image_v_checking.append(image_v_checking)
                    listof_char_recog.append(char_recog)
                    listof_input_image.append(input_image)
                    listof_max_id.append(processed_font[max_id])
                    listof_detected_and_segmented.append(ds_image)
                    listof_normal_processing_result.append(normal_processing_result)
                    listof_crop_ratio_processing_result.append(crop_ratio_processing_result)
                    listof_imagelist_horizontal_line_by_eight_conn.append(imagelist_horizontal_line_by_eight_conn)
                    if global_count+1 == numfiles:
                        yield str(global_count+1) + '_DONE!_' + str(numfiles) + '$'
            print('processed font: ', processed_font)
        time.sleep(2)  #wait for class init done
        return Response(runner(), mimetype='text/event-stream')
        # return render_template('public/sketch_.html', next='/number1', req=req)

    # return render_template('public/sketch.html')
    return redirect('/processing_result')

def save_image_to_disk():
    store_folder_mr = app.root_path + '/static/img/result/matching_result'
    store_folder_sv = app.root_path + '/static/img/result/scale_visualize'
    store_folder_wb = app.root_path + '/static/img/result/white_block'
    store_folder_final = app.root_path + '/static/img/result/final_result'
    store_folder_np = app.root_path + '/static/img/result/normal_processing'
    store_folder_cp = app.root_path + '/static/img/result/crop_processing'
    store_folder_ec = app.root_path + '/static/img/result/eight_conn'
    store_folder_ii = app.root_path + '/static/img/result/input_image'
    store_folder_vc = app.root_path + '/static/img/result/v_checking'
    store_folder_cr = app.root_path + '/static/img/result/char_recog'
    store_folder_ds = app.root_path + '/static/img/result/detected_and_segmented'

    if os.path.exists(store_folder_mr):
        shutil.rmtree(store_folder_mr)
        os.makedirs(store_folder_mr)
    else:
        os.makedirs(store_folder_mr)
    if os.path.exists(store_folder_sv):
        shutil.rmtree(store_folder_sv)
        os.makedirs(store_folder_sv)
    else:
        os.makedirs(store_folder_sv)
    if os.path.exists(store_folder_wb):
        shutil.rmtree(store_folder_wb)
        os.makedirs(store_folder_wb)
    else:
        os.makedirs(store_folder_wb)
    if os.path.exists(store_folder_final):
        shutil.rmtree(store_folder_final)
        os.makedirs(store_folder_final)
    else:
        os.makedirs(store_folder_final)
    if os.path.exists(store_folder_np):
        shutil.rmtree(store_folder_np)
        os.makedirs(store_folder_np)
    else:
        os.makedirs(store_folder_np)
    if os.path.exists(store_folder_cp):
        shutil.rmtree(store_folder_cp)
        os.makedirs(store_folder_cp)
    else:
        os.makedirs(store_folder_cp)
    if os.path.exists(store_folder_ec):
        shutil.rmtree(store_folder_ec)
        os.makedirs(store_folder_ec)
    else:
        os.makedirs(store_folder_ec)
    if os.path.exists(store_folder_ii):
        shutil.rmtree(store_folder_ii)
        os.makedirs(store_folder_ii)
    else:
        os.makedirs(store_folder_ii)
    if os.path.exists(store_folder_vc):
        shutil.rmtree(store_folder_vc)
        os.makedirs(store_folder_vc)
    else:
        os.makedirs(store_folder_vc)
    if os.path.exists(store_folder_cr):
        shutil.rmtree(store_folder_cr)
        os.makedirs(store_folder_cr)
    else:
        os.makedirs(store_folder_cr)
    if os.path.exists(store_folder_ds):
        shutil.rmtree(store_folder_ds)
        os.makedirs(store_folder_ds)
    else:
        os.makedirs(store_folder_ds)

    pathof_imagelist_template_matching_result = []
    count1 = 0
    for list_image in listof_imagelist_template_matching_result:
        count2 = 0
        path_image = []
        for image in list_image:
            file = store_folder_mr+'/'+str(count1)+'_'+str(count2)+'.png'
            if image != []:
                cv2.imwrite(file, image)
                path_image.append(file[35:])
            count2 += 1
        pathof_imagelist_template_matching_result.append(path_image)
        count1 += 1
    
    
    pathof_imagelist_template_scale_visualize = []
    count1 = 0
    for list_image in listof_imagelist_template_scale_visualize:
        count2 = 0
        path_image = []
        for font in list_image:
            for image in font:
                file = store_folder_sv+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
        pathof_imagelist_template_scale_visualize.append(path_image)
        count1 += 1

    pathof_imagelist_visualize_white_block = []
    count1 = 0
    for list_image in listof_imagelist_visualize_white_block:
        count2 = 0
        path_image = []
        for font in list_image:
            for marker in font:
                for image in marker:
                    if image != []:
                        file = store_folder_wb+'/'+str(count1)+'_'+str(count2)+'.png'
                        if image != []:
                            cv2.imwrite(file, image)
                            path_image.append(file[35:])
                        count2 += 1
        pathof_imagelist_visualize_white_block.append(path_image)
        count1 += 1

    pathof_imagelist_horizontal_line_by_eight_conn= []
    count1 = 0
    for list_image in listof_imagelist_horizontal_line_by_eight_conn:
        count2 = 0
        path_image = []
        for image in list_image:
            file = store_folder_ec+'/'+str(count1)+'_'+str(count2)+'.png'
            if image != []:
                cv2.imwrite(file, image)
                path_image.append(file[35:])
            count2 += 1
        pathof_imagelist_horizontal_line_by_eight_conn.append(path_image)
        count1 += 1


    pathof_normal_processing_result = []
    count1 = 0
    for list_image in listof_normal_processing_result:
        pathof_normal = []
        if list_image != []:
            count2 = 0
            one = []
            two = []
            three = []
            four = []
            five = []
            six = []
            zero = []
            path_image = []
            image = list_image[6]
            file = store_folder_np+'/'+str(count1)+'_'+str(count2)+'.png'
            cv2.imwrite(file, image)
            path_image.append(file[35:])
            count2 += 1
            six.append(path_image)
            path_image = []
            for image in list_image[3]:
                file = store_folder_np+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            three.append(path_image)
            path_image = []
            for image in list_image[4]:
                file = store_folder_np+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            four.append(path_image)
            path_image = []
            for image in list_image[5]:
                file = store_folder_np+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            five.append(path_image)
            path_image = []
            for image in list_image[1]:
                file = store_folder_np+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            one.append(path_image)
            path_image = []
            for result in list_image[0]:
                for image in result:
                    file = store_folder_np+'/'+str(count1)+'_'+str(count2)+'.png'
                    if image != []:
                        cv2.imwrite(file, image)
                        path_image.append(file[35:])
                    count2 += 1
            zero.append(path_image)
            path_image = []
            for image in list_image[2]:
                file = store_folder_np+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            two.append(path_image)
            pathof_normal.append(zero)
            pathof_normal.append(one)
            pathof_normal.append(two)
            pathof_normal.append(three)
            pathof_normal.append(four)
            pathof_normal.append(five)
            pathof_normal.append(six)
            pathof_normal_processing_result.append(pathof_normal)
            count1 += 1
        else:
            pathof_normal_processing_result.append([])

    pathof_crop_ratio_processing_result = []
    count1 = 0
    for list_image in listof_crop_ratio_processing_result:
        pathof_crop = []
        if list_image != []:
            count2 = 0
            one = []
            two = []
            three = []
            four = []
            zero = []
            path_image = []
            for image in list_image[3]:
                file = store_folder_cp+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            three.append(path_image)
            path_image = []
            for image in list_image[4]:
                file = store_folder_cp+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            four.append(path_image)
            path_image = []
            for image in list_image[1]:
                file = store_folder_cp+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            one.append(path_image)
            path_image = []
            for image in list_image[0]:
                file = store_folder_cp+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            zero.append(path_image)
            path_image = []
            for image in list_image[2]:
                file = store_folder_cp+'/'+str(count1)+'_'+str(count2)+'.png'
                if image != []:
                    cv2.imwrite(file, image)
                    path_image.append(file[35:])
                count2 += 1
            two.append(path_image)
            pathof_crop.append(zero)
            pathof_crop.append(one)
            pathof_crop.append(two)
            pathof_crop.append(three)
            pathof_crop.append(four)
            pathof_crop_ratio_processing_result.append(pathof_crop)
            count1 += 1
        else:
            pathof_crop_ratio_processing_result.append([])
    
    pathof_imagelist_input_image = []
    count1 = 0
    for list_image in listof_input_image:
        count2 = 0
        path_image = []
        for image in list_image:
            file = store_folder_ii+'/'+str(count1)+'_'+str(count2)+'.png'
            if image != []:
                cv2.imwrite(file, image)
                path_image.append(file[35:])
            count2 += 1
        pathof_imagelist_input_image.append(path_image)
        count1 += 1
    
    pathof_imagelist_v_checking = []
    count1 = 0
    for list_image in listof_image_v_checking:
        count2 = 0
        path_image = []
        for image in list_image:
            file = store_folder_vc+'/'+str(count1)+'_'+str(count2)+'.png'
            if image != []:
                cv2.imwrite(file, image)
                path_image.append(file[35:])
            count2 += 1
        pathof_imagelist_v_checking.append(path_image)
        count1 += 1

    pathof_imagelist_char_recog = []
    count1 = 0
    for list_image in listof_char_recog:
        count2 = 0
        path_image = []
        for image in list_image:
            file = store_folder_cr+'/'+str(count1)+'_'+str(count2)+'.png'
            if image != []:
                cv2.imwrite(file, image)
                path_image.append(file[35:])
            count2 += 1
        pathof_imagelist_char_recog.append(path_image)
        count1 += 1
    
    
    
    pathof_final_image_result = []
    count1 = 0
    for image in listof_final_image_result:
        if image != []:
            file = store_folder_final+'/'+str(count1)+'.png'
            cv2.imwrite(file, image)
            pathof_final_image_result.append(file[35:])
        else:
            pathof_final_image_result.append([])
        count1 += 1
    
    pathof_detected_and_segmented_image = []
    count1 = 0
    for image in listof_detected_and_segmented:
        if image != []:
            file = store_folder_ds+'/'+str(count1)+'.png'
            cv2.imwrite(file, image)
            pathof_detected_and_segmented_image.append(file[35:])
        else:
            pathof_detected_and_segmented_image.append([])
        count1 += 1

    return pathof_crop_ratio_processing_result, pathof_imagelist_horizontal_line_by_eight_conn,\
        pathof_imagelist_template_matching_result, pathof_imagelist_template_scale_visualize, \
            pathof_imagelist_visualize_white_block, pathof_normal_processing_result, \
                pathof_final_image_result, pathof_imagelist_input_image, \
                    pathof_imagelist_v_checking, pathof_imagelist_char_recog, \
                        pathof_detected_and_segmented_image


print('processed font: ', processed_font)
@app.route("/processing_result")
def processing_result():
    global from_sketch_button
    global list_image_files
    global setting
    global req
    global temp_object
    global listof_imagelist_template_matching_result
    global listof_imagelist_template_scale_visualize
    global listof_imagelist_visualize_white_block
    global listof_final_image_result
    global listof_normal_processing_result
    global listof_crop_ratio_processing_result
    global listof_imagelist_horizontal_line_by_eight_conn
    global listof_prediction_result
    global listof_image_v_checking
    global listof_char_recog
    global listof_input_image
    global listof_max_id
    global listof_detected_and_segmented
    global processed_font
    global model
    global global_count

    print("READY TO DO SOMETHING")
    shutil.rmtree(app.config["IMAGE_UPLOADS"])
    os.makedirs(app.config["IMAGE_UPLOADS"])

    pathof_crop_ratio_processing_result, pathof_imagelist_horizontal_line_by_eight_conn,\
        pathof_imagelist_template_matching_result, pathof_imagelist_template_scale_visualize, \
            pathof_imagelist_visualize_white_block, pathof_normal_processing_result, \
                pathof_final_image_result, pathof_imagelist_input_image, \
                    pathof_imagelist_v_checking, pathof_imagelist_char_recog, \
                        pathof_detected_and_segmented_image = save_image_to_disk()
    
    dumpPath = [pathof_crop_ratio_processing_result, pathof_imagelist_horizontal_line_by_eight_conn,
                pathof_imagelist_template_matching_result, pathof_imagelist_template_scale_visualize, 
                pathof_imagelist_visualize_white_block, pathof_normal_processing_result, 
                pathof_final_image_result, listof_prediction_result, pathof_imagelist_input_image,
                pathof_imagelist_v_checking, pathof_imagelist_char_recog, listof_max_id,
                pathof_detected_and_segmented_image]

    filename = "/home/mhbrt/Desktop/Wind/Multiscale/static/dumpPath.pkl"
    pickle.dump(dumpPath, open(filename, 'wb'))

    # Clearing global variable
    from_sketch_button = False
    list_image_files = []
    global_count = 0
    setting = {}
    req = {}
    temp_object = []
    imagelist_template_matching_result = []
    imagelist_template_scale_visualize = []
    imagelist_visualize_white_block = []
    final_image_result = 0
    normal_processing_result = 0
    crop_ratio_processing_result = 0
    imagelist_horizontal_line_by_eight_conn = 0
    listof_imagelist_template_matching_result = []
    listof_imagelist_template_scale_visualize = []
    listof_imagelist_visualize_white_block = []
    listof_final_image_result = []
    listof_normal_processing_result = []
    listof_crop_ratio_processing_result = []
    listof_imagelist_horizontal_line_by_eight_conn = []
    listof_prediction_result = []
    listof_image_v_checking = []
    listof_char_recog = []
    listof_input_image = []
    listof_max_id = []
    listof_detected_and_segmented = []

    print('processed font: ', processed_font)
    write_html.display_result(filename, processed_font)
    processed_font = []
    # return 'OK'
    return render_template('public/test_result.html', marker_type=app.config['MARKER_TYPE'])

@app.route("/guestbook/create-entry", methods=["POST"])
def create_entry():

    req = request.get_json()

    print(req)
    print(request.url)

    # res = make_response(jsonify({"message": "OK"}), 200)
    res = make_response(jsonify(req), 200)

    return res
    # return render_template('public/jinja.html')


@app.route('/jinja', methods=["GET", "POST"])
def jinja():

    req = request.get_json()
    # print(app.config["MARKER"])
    print('hhh', req)

    test_list = ['2323', 'fdfd', '23123']
    if request.method == "POST":
        req = request.form
        # username = request.form.get("username")
        # email = request.form.get("email")
        # password = request.form.get("password")

        # # Alternatively

        # username = request.form["username"]
        # email = request.form["email"]
        # password = request.form["password"]

        print(req)
        print(request.url)

        return redirect(request.url)

    return render_template('public/jinja.html', test_list=test_list)


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            image.save(os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename))

            print("Image saved")

            return redirect(request.url)

    return render_template("public/upload_image.html")


@app.route('/dataset')
def dataset():
    font_folder = ['AlKareem', 'AlQalam', 'KFGQPC', 'LPMQ', 'PDMS',
                   'amiri', 'meQuran', 'norehidayat', 'norehira', 'norehuda']
    print('ds', app.config['MARKER_FOLDER'][font_folder[1]])
    print(app.root_path)
    return render_template('public/dataset.html', marker_folder=app.config['MARKER_FOLDER'], marker_type=app.config['MARKER_TYPE'])

@app.route('/marker/<path:filename>')
def base_static(filename):
    return send_from_directory(app.root_path + '/marker/', filename)

def check_image_size(path, threshold):
    image = cv2.imread(path)
    h, w, _ = image.shape
    if h > threshold or w > threshold:
        if h >= w:
            scale = threshold/h
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
        else:
            scale = threshold/w
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
        cv2.imwrite(path, image)

# res = ''
@app.route('/sketch', methods=["GET", "POST"])
def sketch():
    global from_sketch_button
    global setting
    global list_image_files
    # global res
    if request.method == "POST":
        if request.is_json:
            req = {}
            res = request.get_json()
            print(res)

            if res == 'start':
                from_sketch_button = True
                response = make_response(jsonify('processing'), 200)
                return response
            setting = res
            if type(res) == type([]):
                print(len(res))
                now = datetime.now()
                dt_string = now.strftime("%y%m%d%H%M%S")
                thefile = open('/home/mhbrt/Desktop/Wind/Multiscale/templates/Saved Configuration_'+dt_string+'.dcfg', 'w')
                thefile.write(str(res).replace("'", '"'))
                response = make_response(jsonify(res), 200)
                return response
            # print('____IM DOING SOMENTHING___')
            # imagePath = '/home/mhbrt/Desktop/Wind/Multiscale/temp/0v4.jpg'
            # markerPath = '/home/mhbrt/Desktop/Wind/Multiscale/marker'
            # img = cv2.imread(imagePath)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # # font_list = mess.font(imagePath=imagePath, image=gray, setting=res)
            # font_list = flow.font_list(
            #     imagePath=imagePath, image=gray, setting=res, markerPath=markerPath)
            # temp_object = []
            # # print(font_list)
            # # print(font_list[1].marker_location)
            # for font_object in font_list:
            #     font_object.run()
            #     temp_object.append(font_object.get_object_result())
            # print(temp_object)
            # flow.big_blok(temp_object, imagePath,
            #               font_object, model, font_list)
            if type(res) == type({}):
                response = make_response(jsonify('ok'), 200)
                return response
            # return redirect(request.url)

        if request.files:
            # files = request.files.getlist("files")
            # print(files)

            image = request.files["image"]
            print(image)
            saved_path = os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename)
            list_image_files.append(saved_path)
            image.save(saved_path)
            check_image_size(saved_path, 1200)
            print("Image saved")
            print(list_image_files)
            res = make_response(jsonify(saved_path), 200)
            # return render_template('public/sketch.html', marker_type=app.config['MARKER_TYPE'])
            # return redirect(request.url)
            return res

    print('outside if')

    return render_template('public/sketch.html', marker_type=app.config['MARKER_TYPE'])




req = {}
# from_sketch_button = True


@app.route('/sketch_')
def sketch_():
    global from_sketch_button
    global req

    # from_sketch_button = True
    # for x in range(3):
    req['A message from python'] = 'Initialiation'
    if from_sketch_button:
        # resetting all global variable
        from_sketch_button = True
        req = {}
        req['A message from python'] = 'Doing the number1'
        print('button________________')
        # res = make_response(jsonify(req), 200)
        # time.sleep(6)
        print(request.url)
        return render_template('public/sketch_.html', next='/number1', req=req)

    # return render_template('public/sketch.html')
    return redirect('/sketch')


@app.route('/number1')
def do_something():
    print(from_sketch_button)
    time.sleep(5)
    req['A message from python'] = 'number 1 done and prepare for number 2'
    return render_template('public/sketch_.html', next='/number2', req=req)


@app.route('/number2')
def do_something_again():
    print(from_sketch_button)
    time.sleep(4)
    req['A message from python'] = 'number 2 done and prepare for number 3'
    return render_template('public/sketch_.html', next='/number3', req=req)


@app.route('/number3')
def do_something_and_again():
    print(from_sketch_button)
    time.sleep(4)
    req['A message from python'] = 'number 3 done and prepare for number 4'
    return render_template('public/sketch_.html', next='/number4', req=req)


@app.route('/number4')
def do_something_and_again_and_again_and_again():
    print(from_sketch_button)
    time.sleep(4)
    req['A message from python'] = 'number 4 done and prepare for number 5'
    return render_template('public/sketch_.html', next='/result', req=req)


@app.route('/result')
def do_something_and_again_final():
    print(from_sketch_button)
    time.sleep(3)
    req['A message from python'] = 'number 4 done and back to sketch'
    # return render_template('public/sketch_.html', next='/sketch', req=req)
    return render_template('public/sketch.html', marker_type=app.config['MARKER_TYPE'])

@app.route('/training_result')
def training_result():
    return render_template('public/training_result.html')

@app.route('/temp_result')
def temp_result():
    global list_image_files
    list_image_files = []
    return render_template('public/test_result.html', marker_type=app.config['MARKER_TYPE'])

# @app.route('/test')
# def test():
#     test = 'THIS IS JINJA'
#     return render_template('public/test.html', test=test)

# @app.route('/sketch_')
# def sketch_():
#     global from_sketch_button
#     global first
#     global second
#     global end
#     global l
#     global req
#     from_sketch_button = True
#     # for x in range(3):
#     req['A message from python'] = 'Initialiation'
#     if from_sketch_button:
#         from_sketch_button = False
#         first = True
#         req['A message from python'] = 'This is the message'
#         # req[x]=x
#         print('button________________')
#         res = make_response(jsonify(req), 200)
#         # time.sleep(1)
#         print(request.url)
#         # return render_template('public/sketch_.html', req=req)
#         return res
#         # return redirect(request.url)
#     if first:
#         first = False
#         second = True
#         print('first________________________________')
#         req['A message from python'] = 'This is comes from the first section'
#         req[l] = l
#         res = make_response(jsonify(req), 200)
#         time.sleep(1)
#         # return render_template('public/sketch_.html', req=req)
#         return res
#     if second:
#         second = False
#         end = True
#         print('second_________________')
#         req['A message from python'] = 'This is comes from the second section'
#         req[l] = l
#         time.sleep(1)
#         res = make_response(jsonify(req), 200)
#         # return render_template('public/sketch_.html', req=req)
#         return res
#     # if end:
#     #     end = False
#     #     if l > 0:
#     #         l -= 1
#     #         first = True
#     #         time.sleep(1)
#     #         return render_template('public/sketch_.html', req=req)
#     return render_template('public/sketch_.html', req=req)
