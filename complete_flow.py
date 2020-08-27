#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mess
import match_prepare as match
import matplotlib.pyplot as plt
import numpy as np
import imutils
import glob
import cv2
import copy
import pickle
from scipy.signal import find_peaks, peak_prominences
from matplotlib.figure import Figure


# #### horizontal projection

# In[2]:


def horizontal_projection(image_h):
    image = image_h.copy()
    image[image < 127] = 1
    image[image >= 127] = 0
    h_projection = np.sum(image, axis=1)

    return h_projection


def detect_horizontal_line_up_down(h_projection):
    # Detect line horizontal
    start_point = []
    for x in range(len(h_projection)):
        if h_projection[x] > 0:
            start_point.append(x)
            break

    for x in range(len(h_projection))[::-1]:
        if h_projection[x] > 0:
            start_point.append(x)
            break

    if len(start_point) % 2 != 0:
        if h_projection[len(h_projection) - 1] > 0:
            start_point.append(len(h_projection) - 1)

    return start_point


def detect_horizontal_line(h_projection, pixel_limit_ste, pixel_limit_ets):
    # Detect line horizontal
    up_flag = 0
    down_flag = 0
    # pixel_limit = 5
    start_to_end = 0
    end_to_start = pixel_limit_ets + 1
    start_point = []
    for x in range(len(h_projection)):
        if h_projection[x] > 0 and up_flag == 1:
            start_to_end += 1

        if h_projection[x] == 0 and up_flag == 1:
            # print(start_to_end)
            start_point.append(x)
            # print(start_point)
            if start_to_end < pixel_limit_ste:
                del(start_point[len(start_point) - 1])
                # print('delete ste')
                down_flag = 0
                up_flag = 1
            else:
                down_flag = 1
                up_flag = 0
                start_to_end = 0

        if h_projection[x] == 0 and down_flag == 1:
            end_to_start += 1

        if h_projection[x] > 0 and up_flag == 0:
            start_point.append(x)
            # print(start_point)
            if end_to_start < pixel_limit_ets:
                del(start_point[len(start_point)-1])
                del(start_point[len(start_point)-1])
            up_flag = 1
            down_flag = 0
            end_to_start = 0

    if len(start_point) % 2 != 0:
        if h_projection[len(h_projection) - 1] > 0:
            start_point.append(len(h_projection) - 1)

    return start_point


# #### used by image processing stage

# In[3]:


def get_ul_coordinat(coordinat, height, ht_overide=0):
    if ht_overide > 0:
        height_tanwin = ht_overide
    else:
        height_tanwin = coordinat[3] - coordinat[1]
    # upper
    if coordinat[1]-height_tanwin > 0:
        upper = [coordinat[0], coordinat[1]-height_tanwin,
                 coordinat[2], coordinat[3]-height_tanwin]
    else:
        upper = [coordinat[0], 0,
                 coordinat[2], coordinat[3]-height_tanwin]
    # lower
    if coordinat[3]+height_tanwin < height:
        lower = [coordinat[0], coordinat[1]+height_tanwin,
                 coordinat[2], coordinat[3]+height_tanwin]
    else:
        lower = [coordinat[0], coordinat[1]+height_tanwin,
                 coordinat[2], height-1]

    return upper, lower


def upper_or_lower(bw_img, upper, lower):
    upper_count = 0
    for x in range(upper[0], upper[2]):
        for y in range(upper[1], upper[3]):
            if bw_img[y, x] < 1:
                upper_count += 1

    lower_count = 0
    for x in range(lower[0], lower[2]):
        for y in range(lower[1], lower[3]):
            if bw_img[y, x] < 1:
                lower_count += 1

    return upper_count, lower_count


def get_lr_coordinat(coordinat, width):
    width_tanwin = coordinat[2] - coordinat[0]
    if coordinat[0] - width_tanwin > 0:
        left = [coordinat[0]-width_tanwin, coordinat[1],
                coordinat[2]-width_tanwin, coordinat[3]]
    else:
        left = [0, coordinat[1],
                coordinat[2]-width_tanwin, coordinat[3]]
    if coordinat[2] + width_tanwin < width:
        right = [coordinat[0]+width_tanwin, coordinat[1],
                 coordinat[2]+width_tanwin, coordinat[3]]
    else:
        right = [coordinat[0]+width_tanwin, coordinat[1],
                 width-1, coordinat[3]]

    return left, right


def black_pixel_count(bw_img, lower):
    lower_count = 0
    for x in range(lower[0], lower[2]):
        for y in range(lower[1], lower[3]):
            if bw_img[y, x] < 1:
                lower_count += 1

    return lower_count


def get_marker_name(key):
    part = key.split('_')
    name = []
    for x in range(len(part)):
        if x == 0:
            continue
        if x == len(part) - 1:
            name.append(part[x])
        else:
            name.append(part[x] + '_')
    name = ''.join(name)

    return name


def region_tanwin(coordinat, image, font_list, view=True):
    saved_tanwin_height = coordinat[3] - coordinat[1]
    saved_tanwin_width = coordinat[2] - coordinat[0]
    font_object = font_list[0]
    h, w, = image.shape

    marker_only_count = black_pixel_count(image, coordinat)

    con_pack = font_object.eight_connectivity(image.copy(), coordinat,
                                              left=False, right=False)
    image_process = image.copy()
    image_process[:] = 255
    for region in con_pack:
        for val in con_pack[region]:
            image_process[val] = 0
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)
    # cv2.destroyAllWindows()
    font_object.horizontal_projection(image_process)
    h_image = font_object.detect_horizontal_line(image.copy(), int(
        1/2*saved_tanwin_width), int(1/2*saved_tanwin_width))
    start_point_h = font_object.start_point_h
    font_object.vertical_projection(image_process)
    h_image = font_object.detect_vertical_line(image.copy(), 0)
    start_point_v = font_object.start_point_v

    coordinat_candidate = [start_point_v[0], start_point_h[0],
                           start_point_v[1], start_point_h[1]]
    cc_count = black_pixel_count(image, coordinat_candidate)

    if marker_only_count < cc_count < 2 * marker_only_count:
        print('marker only count', marker_only_count)
        print('cc count', cc_count)
        print('region tanwin modified')
        coordinat = coordinat_candidate

    cv2.rectangle(image_process, (coordinat[0], coordinat[1]),
                  (coordinat[2], coordinat[3]), (0, 255, 0), 2)
    if view:
        cv2.imshow('coordinat tanwin', image_process)
        cv2.waitKey(0)

    upper, lower = get_ul_coordinat(coordinat, h)
    # print(upper)
    # print(lower)
    upper_count, lower_count = upper_or_lower(image, upper, lower)

    if upper_count < lower_count:
        cv2.rectangle(image_process, (lower[0], lower[1]),
                      (lower[2], lower[3]), (100, 150, 0), 2)
        region = lower
    elif upper_count > lower_count:
        cv2.rectangle(image_process, (upper[0], upper[1]),
                      (upper[2], upper[3]), (100, 150, 0), 2)
        region = upper
    else:
        print('enlarge')
        while(upper_count == lower_count):
            upper, _ = get_ul_coordinat(upper, h)
            _, lower = get_ul_coordinat(lower, h)
            upper_count, lower_count = upper_or_lower(image, upper, lower)
            if upper_count < lower_count:
                cv2.rectangle(image_process, (lower[0], lower[1]),
                              (lower[2], lower[3]), (100, 150, 0), 2)
                region = lower
            elif upper_count > lower_count:
                cv2.rectangle(image_process, (upper[0], upper[1]),
                              (upper[2], upper[3]), (100, 150, 0), 2)
                region = upper
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)

    return region


def raw_baseline(image_b, font_object):
    #     image_b = bw_image[oneline_coordinat[0]:oneline_coordinat[1], :]
    font_object.horizontal_projection(image_b)
    font_object.base_line(image_b)
    oneline_baseline = []
    oneline_baseline.append(font_object.base_start)
    oneline_baseline.append(font_object.base_end)
    if oneline_baseline[1] < oneline_baseline[0]:
        temp = oneline_baseline[0]
        oneline_baseline[0] = oneline_baseline[1]
        oneline_baseline[1] = temp
    oneline_image = font_object.one_line_image
    # cv2.imshow('line', oneline_image)
    # print('>')

    return oneline_baseline


# #### eight_conn_by_seed_tanwin

# In[6]:


def eight_conn_by_seed_tanwin(coordinat, img, font_list, view=True):
    original_coordinat = coordinat
    saved_tanwin_height = coordinat[3] - coordinat[1]
    font_object = font_list[0]
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     coordinat = [ 98, 625, 109, 640]
    # mid_seed = coordinat[1] + int((coordinat[3]-coordinat[1])/2)
    # seed = [0, mid_seed, w, mid_seed+1]
    # Otsu threshold
    # ret_img, image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY
    #                                 + cv2.THRESH_OTSU)
    # Simple threshold
    # ret_img, image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Adaptive threshold value is the mean of neighbourhood area
    # image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               cv2.THRESH_BINARY, 11, 2)

    # Adaptive threshold value is the weighted sum of neighbourhood
    # values where weights are a gaussian window
    image = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

    marker_only_count = black_pixel_count(image, coordinat)

    con_pack = font_object.eight_connectivity(image.copy(), coordinat,
                                              left=False, right=False)
    image_process = image.copy()
    image_process[:] = 255
    for region in con_pack:
        for val in con_pack[region]:
            image_process[val] = 0
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)
    # cv2.destroyAllWindows()
    font_object.horizontal_projection(image_process)
    h_image = font_object.detect_horizontal_line(image.copy(), 0, 0)
    start_point_h = font_object.start_point_h
    font_object.vertical_projection(image_process)
    h_image = font_object.detect_vertical_line(image.copy(), 0)
    start_point_v = font_object.start_point_v

    coordinat_candidate = [start_point_v[0], start_point_h[0],
                           start_point_v[1], start_point_h[1]]
    cc_count = black_pixel_count(image, coordinat_candidate)

    if cc_count < 2 * marker_only_count:
        coordinat = coordinat_candidate

    cv2.rectangle(image_process, (coordinat[0], coordinat[1]),
                  (coordinat[2], coordinat[3]), (0, 255, 0), 2)
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)

    upper, lower = get_ul_coordinat(coordinat, h)
    # print(upper)
    # print(lower)
    upper_count, lower_count = upper_or_lower(image, upper, lower)

    if upper_count < lower_count:
        print('lower')
        left, right = get_lr_coordinat(lower, w)
        con_pack = font_object.eight_connectivity(image.copy(), lower,
                                                  left=False, right=False)
        for region in con_pack:
            for val in con_pack[region]:
                image_process[val] = 0
        cv2.rectangle(image_process, (lower[0], lower[1]),
                      (lower[2], lower[3]), (100, 150, 0), 2)
    elif upper_count > lower_count:
        print('upper')
        left, right = get_lr_coordinat(upper, w)
        con_pack = font_object.eight_connectivity(image.copy(), upper,
                                                  left=False, right=False)
        for region in con_pack:
            for val in con_pack[region]:
                image_process[val] = 0
        cv2.rectangle(image_process, (upper[0], upper[1]),
                      (upper[2], upper[3]), (100, 150, 0), 2)
    else:
        print('enlarge')
        while(upper_count == lower_count):
            upper, _ = get_ul_coordinat(upper, h)
            _, lower = get_ul_coordinat(lower, h)
            upper_count, lower_count = upper_or_lower(image, upper, lower)
            if upper_count < lower_count:
                left, right = get_lr_coordinat(lower, w)
                con_pack = font_object.eight_connectivity(image.copy(), lower,
                                                          left=False, right=False)
                for region in con_pack:
                    for val in con_pack[region]:
                        image_process[val] = 0
                cv2.rectangle(image_process, (lower[0], lower[1]),
                              (lower[2], lower[3]), (100, 150, 0), 2)
            elif upper_count > lower_count:
                left, right = get_lr_coordinat(upper, w)
                con_pack = font_object.eight_connectivity(image.copy(), upper,
                                                          left=False, right=False)
                for region in con_pack:
                    for val in con_pack[region]:
                        image_process[val] = 0
                cv2.rectangle(image_process, (upper[0], upper[1]),
                              (upper[2], upper[3]), (100, 150, 0), 2)

    left_count, _ = upper_or_lower(image, left, right)
    while(left_count < marker_only_count):
        left, _ = get_lr_coordinat(left, w)
        left_count, _ = upper_or_lower(image, left, right)
        if left[0] < 1 and left_count < 2:
            left = original_coordinat
            break
    cv2.rectangle(image_process, (left[0], left[1]),
                  (left[2], left[3]), (100, 150, 0), 2)
    con_pack = font_object.eight_connectivity(image.copy(), left,
                                              left=False, right=False)
    max_left_region = 0
    for region in con_pack:
        #         print(con_pack[region])
        #         print(region)
        if len(con_pack[region]) > max_left_region:
            max_left_region = len(con_pack[region])
            # print(max_left_region)
        for val in con_pack[region]:
            #             print(region)
            #             print(val)
            image_process[val] = 0
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image_process_after_left = image.copy()
    image_process_after_left[:] = 255
    for region in con_pack:
        for val in con_pack[region]:
            image_process_after_left[val] = 0
    font_object.horizontal_projection(image_process_after_left)
    h_image_al = font_object.detect_horizontal_line(image.copy(), 0, 0)
    start_point_h_al = font_object.start_point_h
    coordinat_al = [0, start_point_h_al[0], w, start_point_h_al[1]]

    cv2.rectangle(image_process_after_left, (coordinat_al[0], coordinat_al[1]),
                  (coordinat_al[2], coordinat_al[3]), (0, 255, 0), 2)
    if view:
        cv2.imshow('d_al', image_process_after_left)
        cv2.waitKey(0)

    left = [left[0], start_point_h_al[0], left[2], start_point_h_al[1]]
    cv2.rectangle(image_process, (left[0], left[1]),
                  (left[2], left[3]), (100, 150, 0), 2)

    upper, lower = get_ul_coordinat(left, h, saved_tanwin_height)
#     cv2.rectangle(image_process, (lower[0], lower[1]),
#                           (lower[2], lower[3]), (100, 150,0), 2)
#     cv2.rectangle(image_process, (upper[0], upper[1]),
#                           (upper[2], upper[3]), (100, 150,0), 2)
    con_pack = font_object.eight_connectivity(image.copy(), upper,
                                              left=False, right=False)
#     max_left_region = 30
    for region in con_pack:
        if len(con_pack[region]) > max_left_region:
            continue
        for val in con_pack[region]:
            image_process[val] = 0
    con_pack = font_object.eight_connectivity(image.copy(), lower,
                                              left=False, right=False)
    for region in con_pack:
        if len(con_pack[region]) > max_left_region:
            continue
        for val in con_pack[region]:
            image_process[val] = 0
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)

############################
#     left_count_mod = black_pixel_count(image, left)
    left_count_mod = 100
    left, _ = get_lr_coordinat(left, w)
    left_count = black_pixel_count(image, left)
    while(left_count < 2):
        left, _ = get_lr_coordinat(left, w)
        left_count = black_pixel_count(image, left)
        if left[0] < 1:
            break
    cv2.rectangle(image_process, (left[0], left[1]),
                  (left[2], left[3]), (100, 150, 0), 2)
    con_pack = font_object.eight_connectivity(image.copy(), left,
                                              left=False, right=False)
    for region in con_pack:
        if len(con_pack[region]) > 2 * left_count_mod:
            continue
        for val in con_pack[region]:
            image_process[val] = 0
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)

    left, _ = get_lr_coordinat(left, w)
    left_count = black_pixel_count(image, left)
    while(left_count < 2):
        left, _ = get_lr_coordinat(left, w)
        left_count = black_pixel_count(image, left)
        if left[0] < 1:
            break
    cv2.rectangle(image_process, (left[0], left[1]),
                  (left[2], left[3]), (100, 150, 0), 2)
    con_pack = font_object.eight_connectivity(image.copy(), left,
                                              left=False, right=False)
    for region in con_pack:
        if len(con_pack[region]) > 2 * left_count_mod:
            continue
        for val in con_pack[region]:
            image_process[val] = 0
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)
##########################

    con_pack = font_object.eight_connectivity(image.copy(), right,
                                              left=False, right=False)
    for region in con_pack:
        for val in con_pack[region]:
            image_process[val] = 0
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)
#     cv2.destroyAllWindows()

    font_object.horizontal_projection(image_process)
    al_height = start_point_h_al[1]-start_point_h_al[0]
    print('al_height:', al_height)
    h_image = font_object.detect_horizontal_line(image.copy(), al_height, 5)
    start_point_h = font_object.start_point_h
    if view:
        cv2.imshow('line', h_image)
        cv2.waitKey(0)
#     cv2.destroyAllWindows()

    return start_point_h, image_process


# #### eight_conn_by_seed

# In[7]:


def eight_conn_by_seed(coordinat, img, font_list, view=True):
    original_coordinat = coordinat
    saved_starting_height = coordinat[3] - coordinat[1]
    saved_starting_width = coordinat[2] - coordinat[0]
    font_object = font_list[0]
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     coordinat = [ 98, 625, 109, 640]
    # mid_seed = coordinat[1] + int((coordinat[3]-coordinat[1])/2)
    # seed = [0, mid_seed, w, mid_seed+1]
    # Otsu threshold
    # ret_img, image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY
    #                                 + cv2.THRESH_OTSU)
    # Simple threshold
    # ret_img, image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Adaptive threshold value is the mean of neighbourhood area
    # image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               cv2.THRESH_BINARY, 11, 2)

    # Adaptive threshold value is the weighted sum of neighbourhood
    # values where weights are a gaussian window
    image = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

    marker_only_count = black_pixel_count(image, coordinat)

    con_pack = font_object.eight_connectivity(image.copy(), coordinat,
                                              left=False, right=False)
    max_y_start = 0
    min_y_start = 10000
    for region in con_pack:
        for val in con_pack[region]:
            if val[0] > max_y_start:
                max_y_start = val[0]
            if val[0] < min_y_start:
                min_y_start = val[0]

    starting_height = coordinat[3] - coordinat[1]
    sch_a = max_y_start - coordinat[1]
    sch_b = coordinat[3] - min_y_start

    if sch_a > sch_b:
        starting_conpack_height = sch_a
    else:
        starting_conpack_height = sch_b

    image_process = image.copy()
    image_process[:] = 255
    if starting_conpack_height < 2 * starting_height:
        for region in con_pack:
            for val in con_pack[region]:
                image_process[val] = 0
        if view:
            cv2.imshow('d', image_process)
            cv2.waitKey(0)
    # cv2.destroyAllWindows()
        font_object.horizontal_projection(image_process)
        h_image = font_object.detect_horizontal_line(
            image.copy(), starting_height, 0)
        start_point_h = font_object.start_point_h
        font_object.vertical_projection(image_process)
        h_image = font_object.detect_vertical_line(
            image.copy(), saved_starting_width)
        start_point_v = font_object.start_point_v

        if len(start_point_v) > 1 and len(start_point_h) > 1:
            coordinat_candidate = [start_point_v[0], start_point_h[0],
                                   start_point_v[1], start_point_h[1]]
            cc_count = black_pixel_count(image, coordinat_candidate)

            if cc_count < 2 * marker_only_count:
                print('replace coordinat')
                coordinat = coordinat_candidate

    cv2.rectangle(image_process, (coordinat[0], coordinat[1]),
                  (coordinat[2], coordinat[3]), (0, 255, 0), 2)
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)

    coordinat, right = get_lr_coordinat(coordinat, w)
    left = coordinat
    left_count = black_pixel_count(image, coordinat)
    print('leftcount:', left_count)
    left_thresh = int(1/2*marker_only_count)
    while(left_count < left_thresh):
        coordinat, _ = get_lr_coordinat(coordinat, w)
        left_count = black_pixel_count(image, coordinat)
        print('lt:', left_thresh)
        print('leftcount:', left_count)
        left = coordinat
        if coordinat[0] < 1 and left_count < left_thresh:
            left = original_coordinat
            break

    cv2.rectangle(image_process, (left[0], left[1]),
                  (left[2], left[3]), (100, 150, 0), 2)
    con_pack = font_object.eight_connectivity(image.copy(), left,
                                              left=False, right=False)
    max_left_region = 0
    for region in con_pack:
        if len(con_pack[region]) > max_left_region:
            max_left_region = len(con_pack[region])
#             print(max_left_region)
        for val in con_pack[region]:
            image_process[val] = 0
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image_process_after_left = image.copy()
    image_process_after_left[:] = 255
    for region in con_pack:
        for val in con_pack[region]:
            image_process_after_left[val] = 0
    font_object.horizontal_projection(image_process_after_left)
    h_image_al = font_object.detect_horizontal_line(image.copy(), 0, 0)
    start_point_h_al = font_object.start_point_h
    coordinat_al = [0, start_point_h_al[0], w, start_point_h_al[1]]

    cv2.rectangle(image_process_after_left, (coordinat_al[0], coordinat_al[1]),
                  (coordinat_al[2], coordinat_al[3]), (0, 255, 0), 2)
    if view:
        cv2.imshow('d_al', image_process_after_left)
        cv2.waitKey(0)

    left = [left[0], start_point_h_al[0], left[2], start_point_h_al[1]]
    cv2.rectangle(image_process, (left[0], left[1]),
                  (left[2], left[3]), (100, 150, 0), 2)

    upper, lower = get_ul_coordinat(left, h, saved_starting_height)
#     cv2.rectangle(image_process, (lower[0], lower[1]),
#                           (lower[2], lower[3]), (100, 150,0), 2)
#     cv2.rectangle(image_process, (upper[0], upper[1]),
#                           (upper[2], upper[3]), (100, 150,0), 2)
    con_pack = font_object.eight_connectivity(image.copy(), upper,
                                              left=False, right=False)
    max_left_region = 20
    for region in con_pack:
        if len(con_pack[region]) > max_left_region:
            continue
        for val in con_pack[region]:
            image_process[val] = 0
    con_pack = font_object.eight_connectivity(image.copy(), lower,
                                              left=False, right=False)
    for region in con_pack:
        if len(con_pack[region]) > max_left_region:
            continue
        for val in con_pack[region]:
            image_process[val] = 0
    if view:
        cv2.imshow('d', image_process)
        cv2.waitKey(0)

#     con_pack = font_object.eight_connectivity(image.copy(), right,
#                                               left=False, right=False)
#     for region in con_pack:
#         for val in con_pack[region]:
#             image_process[val] = 0
#     cv2.imshow('d', image_process)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

####

    font_object.horizontal_projection(image_process)
    al_height = start_point_h_al[1]-start_point_h_al[0]
    print('al_height:', al_height)
    h_image = font_object.detect_horizontal_line(image.copy(), al_height, 5)
    start_point_h = font_object.start_point_h
    if view:
        cv2.imshow('line', h_image)
        cv2.waitKey(0)
#     cv2.destroyAllWindows()

    return start_point_h, image_process


# #### image_processing_blok

# In[8]:


def normal_image_processing_blok(imagePath, object_result, bw_method, list_start_point_h):
    original_image = cv2.imread(imagePath)
    height, width, _ = original_image.shape
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # template = cv2.Canny(gray, 50, 200)
    if bw_method == 0:
        # Otsu threshold
        ret_img, image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY
                                       + cv2.THRESH_OTSU)
    if bw_method == 1:
        # Simple threshold
        ret_img, image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if bw_method == 2:
        # Adaptive threshold value is the mean of neighbourhood area
        image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    if bw_method == 3:
        # Adaptive threshold value is the weighted sum of neighbourhood
        # values where weights are a gaussian window
        image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    # cv2.imshow('otsu', image1)
    # cv2.imshow('simple', image2)
    # cv2.imshow('adapt mean', image3)
    # cv2.imshow('start', image)
    # cv2.waitKey(0)
    # image = cv2.bitwise_not(image)
    # kernel = np.ones((1,1), np.uint8)
    # dilation = cv2.dilate(final_img.copy(),kernel,iterations = 1)
    # kernel = np.ones((2,2), np.uint8)
    # image = cv2.erode(image,kernel,iterations = 1)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # image = cv2.bitwise_not(image)
    # closing = cv2.morphologyEx(final_img.copy(), cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('morph', image)
    # print('morph')
    # cv2.waitKey(0)

    input_image = match.ImageProcessing(original_image=original_image.copy())
    # input_image.horizontal_projection(image.copy())  # adaptive binaryimage
    # horizontal_image = input_image.detect_horizontal_line(
    #     image=original_image.copy(),
    #     pixel_limit_ste=5,  # Start to end
    #     pixel_limit_ets=5   # End to start
    # )  # Got self.start_point_h
    # horizontal_image = original_image.copy()
    # for x in range(len(list_start_point_h)):
    #     if x % 2 == 0:     # Start_point
    #         cv2.line(horizontal_image, (0, list_start_point_h[x]),
    #                  (width, list_start_point_h[x]), (0, 0, 255), 2)
    #         # print(x)
    #     else:         # End_point
    #         cv2.line(horizontal_image, (0, list_start_point_h[x]),
    #                  (width, list_start_point_h[x]), (100, 100, 255), 2)
    # cv2.imshow('from main', input_image.original_image)
    # cv2.imshow('h_image', horizontal_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # bag_h_original = input_image.start_point_h.copy()
    bag_h_original = list_start_point_h
    input_image.crop_image(h_point=list_start_point_h,
                           input_image=original_image.copy())  # crop ori
    horizontal_image = input_image.horizontal_image.copy()
#     marker_height_list = []
#     font_list = mess.font(imagePath=imagePath, image=gray)
#     for font_object in font_list:
#         for location in font_object.get_marker_location():
#             temp = cv2.imread(location)
#             h, _, _ = temp.shape
#             marker_height_list.append(h)
#     print(marker_height_list)
    # Block font processing
    count = 0
    save_state = {}
    imagelist_bag_of_h_with_baseline = []
    imagelist_image_final_body = []
    imagelist_image_final_marker = []
    imagelist_perchar_marker = []
    imagelist_final_word_img = []
    imagelist_final_segmented_char = []
    for image in input_image.bag_of_h_crop:
        # Get original cropped one line binary image
        temp_image_ori = input_image.bag_of_h_crop[image]
        h, _, _ = temp_image_ori.shape
        # Scaled image by height ratio
#         scaled_one_line_img_size = 1.3 * max(marker_height_list)
#         if h > scaled_one_line_img_size:
#             scale = scaled_one_line_img_size / h
#             temp_image_ori = imutils.resize(temp_image_ori,
#                                             height=int(h * scale))
#         else:
#             scale = 1
#         if scale != 1:
#             print('Scalling image to ' + str(scale))
#         scale = 0.9285714285714286
#         temp_image_ori = imutils.resize(temp_image_ori,
#                                             height=int(h * scale))
        scale = 1
        gray = cv2.cvtColor(temp_image_ori, cv2.COLOR_BGR2GRAY)
        if bw_method == 0:
            # Otsu threshold
            ret_img, temp_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY
                                        + cv2.THRESH_OTSU)
        if bw_method == 1:
            # Simple threshold
            ret_img, temp_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        if bw_method == 2:
            # Adaptive threshold value is the mean of neighbourhood area
            temp_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        if bw_method == 3:
            # Adaptive threshold value is the weighted sum of neighbourhood
            # values where weights are a gaussian window
            temp_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        # cv2.imshow('line', temp_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # temp_image = temp_image_ori
        # Calculate base line processing from self.h_projection
        input_image.horizontal_projection(temp_image.copy())
        input_image.base_line(one_line_image=temp_image_ori)
        oneline_baseline = []
        oneline_baseline.append(input_image.base_start)
        oneline_baseline.append(input_image.base_end)
        if oneline_baseline[1] < oneline_baseline[0]:
            temp = oneline_baseline[0]
            oneline_baseline[0] = oneline_baseline[1]
            oneline_baseline[1] = temp
        imagelist_bag_of_h_with_baseline.append(input_image.one_line_image)
#         cv2.imshow('Base start =' + str(input_image.base_start)
#                    + ' end =' + str(input_image.base_end),
#                    input_image.one_line_image)
#         print('>')
#         cv2.waitKey(0)
#         cv2.destroyWindow('Base start =' + str(input_image.base_start)
#                           + ' end =' + str(input_image.base_end))

        # Font_Processing
#         font_list = font(imagePath=imagePath, image=gray)
#         max_font_value = 0
#         font_type = 0
#         numstep = 20
        # Looking for font type by the greatest value
#         for font_object in font_list:
#             font_object.run(numstep=numstep)
#             for value in font_object.get_object_result().values():
#                 # print(value)
#                 if type(value) == float:
#                     if value > max_font_value:
#                         max_font_value = value
#                         font_type = font_object
        not_empty = False
        for data in object_result:
            if isinstance(object_result[data], type(np.array([]))):
                not_empty = True
                break

        if not_empty:
            print('into eight connectivity')
            input_image.eight_connectivity(
                temp_image.copy(), oneline_baseline
            )
            conn_pack_sorted = copy.deepcopy(
                input_image.conn_pack_sorted
            )
            conn_pack_minus_body = copy.deepcopy(
                input_image.conn_pack_minus_body
            )
            imagelist_image_final_body.append(input_image.image_final_sorted)
            imagelist_image_final_body.append(input_image.image_v_line)
            imagelist_image_final_marker.append(input_image.image_final_marker)
            # font_type.display_marker_result(input_image=temp_image_ori)
        else:
            object_result = False
            print('Not a valuable result found check the numstep!')
            continue
            # cv2.waitKey(0)

        # Crop next word marker wether it's inside or beside
        crop_words = {}
        if object_result:
            # Grouping marker by its v_projection
            input_image.grouping_marker()
            group_marker_by_wall = copy.deepcopy(
                input_image.group_marker_by_wall
            )
#             print('bw:', group_marker_by_wall)
#             print('bw1:', conn_pack_sorted)
#             print('bw2:', conn_pack_minus_body)
#             print(object_result)
#             cv2.waitKey(0)
            for data in object_result:
                if isinstance(object_result[data], type(np.array([]))):
                    temp_x = object_result[data]
                    part = data.split('_')
                    name = []
                    for x in range(len(part)):
                        if x == 0:
                            continue
                        if x == len(part) - 1:
                            name.append(part[x])
                        else:
                            name.append(part[x] + '_')
                    name = ''.join(name)
                    # crop_words['ordinat_' + name]=temp_x
                    for arr in range(len(temp_x)):
                        y1 = (temp_x)[arr][1]
                        y2 = (temp_x)[arr][3]
                        y_mid = (temp_x)[arr][1] + int((y2-y1)/2)
                        if bag_h_original[image] <= y1 <= bag_h_original[image+1] \
                                or bag_h_original[image] <= y2 <= bag_h_original[image+1]\
                            or bag_h_original[image] <= y_mid <= bag_h_original[image+1]:
                            # print('processing:', name)
                            pass
                        else:
                            # print('continue:', name)
                            continue
                        x2 = (temp_x)[arr][2]  # x2 is on the right
                        x1 = (temp_x)[arr][0]  # x1 is on the left
                        width_x = x2-x1
                        mid_x = x1 + round((x2 - x1)/2)  # x in the middle
#                         print('ordinat ' + data + '={}'.format(x))
                        # marker_width = (temp_x[arr][2]) - x
                        wall_count = -1
                        for wall in group_marker_by_wall:
                            wall_count += 1
                            if wall[0] <= mid_x <= wall[1]:
                                break
#                         cv2.waitKey(0)
                        wall = group_marker_by_wall.keys()
                        wall = list(wall)
                        if wall_count > len(wall)-1 or wall_count < 0:
                            continue
                        ####
                        # print(group_marker_by_wall, wall_count)
                        ####
                        # print('wall min 1', len(wall)-1)
                        # print('wall count ', wall_count)
                        found_in_wall = False
                        for region in group_marker_by_wall[
                                wall[wall_count]]:
                            if found_in_wall:
                                break
                            region_yx = conn_pack_minus_body[
                                region]
                            for y_x in region_yx:
                                # if there is pixel marker that lower than the - x than it is inside
                                if y_x[1] < x1 - 4/5 * width_x:
                                    print('add inside wall')
                                    crop_words['final_inside_' + name
                                               + '_' + str(arr)] \
                                        = wall[wall_count]
                                    crop_words['ordinat_' + name
                                               + '_' + str(arr)] \
                                        = temp_x[arr]
                                    found_in_wall = True
                                    break
                        if not found_in_wall:
                            if wall_count > 0:
                                next_wall = wall[wall_count - 1]
                                found_next_wall = False
                                if group_marker_by_wall[next_wall] != []:
                                    print('add next wall')
                                    crop_words['final_beside_'
                                               + name + '_' + str(arr)] \
                                        = next_wall
                                    crop_words['ordinat_' + name
                                               + '_' + str(arr)] \
                                        = temp_x[arr]
                                    found_next_wall = True
                                if not found_next_wall and wall_count > 1:
                                    beside_next_wall = wall[wall_count - 2]
                                    if group_marker_by_wall[
                                            beside_next_wall] != []:
                                        print('add beside next wall')
                                        crop_words['final_beside_'
                                                   + name + '_'
                                                   + str(arr)] \
                                            = beside_next_wall
                                        crop_words['ordinat_' + name
                                                   + '_' + str(arr)] \
                                            = temp_x[arr]

#             font_type.display_marker_result(input_image=temp_image_ori)

        # Looking for final segmented character
        # print(crop_words_final)
        # for key in crop_words_final:

        print('CROP WORDS = ', crop_words)
        for key in crop_words:
            name = key.split('_')
            if name[0] == 'final':
                save_state[count] = []
                count += 1
                # x_value = crop_words_final[key]
                x_value = crop_words[key]
                # print(x_value)
                join = []
                for x in range(len(name)):
                    if x == 0:
                        continue
                    if x == 1:
                        continue
                    if x == len(name) - 1:
                        join.append(name[x])
                        # print(name[x])
                    else:
                        join.append(name[x] + '_')
                        # print(name[x])
                join = ''.join(join)
                save_state[count-1].append(join)
                print('join = {}'.format(join))

                # List available for final segmented char
                final_segmented_char = temp_image.copy()
                final_segmented_char[:] = 255
                if name[1] == 'beside':
                    # final_img = temp_image.copy()[:, x_value[0]:x_value[1]]
                    final_img = input_image.image_join.copy()[
                        :, x_value[0]:x_value[1]]
                    w_height, w_width = final_img.shape
                    # cv2.imshow(join, final_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    final_segmented_char, pass_x1 = input_image.find_final_processed_char(
                        x_value, oneline_baseline
                    )
                    if final_segmented_char == 'continue':
                        print(
                            '>> from main to continue next word candidate'
                        )
                        continue
                    else:
                        save_state[count-1].append(scale)
                        save_state[count -
                                   1].append((bag_h_original[image], bag_h_original[image+1]))
                        save_state[count-1].append(
                            crop_words['ordinat_' + join]
                        )
                        save_state[count-1].append(final_segmented_char)
                        save_state[count-1].append(pass_x1)
                        save_state[count-1].append([(input_image.fsc_coordinate[0], bag_h_original[image]),
                                                    (input_image.fsc_coordinate[1], bag_h_original[image+1])])

                if name[1] == 'inside':
                    x1_ordinat = crop_words['ordinat_' + join][0]
                    # x1_ordinat = crop_words_final['ordinat_' + join][0]
                    # Cut before the detected char marker
#                     print('x1_ordinat = {}'.format(x1_ordinat))
#                     cv2.waitKey(0)
                    # final_img = temp_image.copy()[:, x_value[0]:x1_ordinat]
                    final_img = input_image.image_join.copy()[
                        :, x_value[0]:x1_ordinat]
                    w_height, w_width = final_img.shape
#                     cv2.imshow('inside', final_img)
                    final_wall = (x_value[0], x1_ordinat)
                    final_segmented_char, pass_x1 = input_image.find_final_processed_char(
                        final_wall, oneline_baseline
                    )
                    if final_segmented_char == 'continue':
                        print(
                            '>> from main to continue next word candidate'
                        )
                        continue
                    else:
                        save_state[count-1].append(scale)
                        save_state[count -
                                   1].append((bag_h_original[image], bag_h_original[image+1]))
                        save_state[count-1].append(
                            crop_words['ordinat_' + join]
                        )
                        save_state[count-1].append(final_segmented_char)
                        save_state[count-1].append(pass_x1)
                        save_state[count-1].append([(input_image.fsc_coordinate[0], bag_h_original[image]),
                                                    (input_image.fsc_coordinate[1], bag_h_original[image+1])])

                imagelist_perchar_marker.append(
                    input_image.imagelist_perchar_marker)
                # print('_________________________________', imagelist_perchar_marker)
                imagelist_final_word_img.append(final_img)
                # print('sd', imagelist_final_word_img)
                imagelist_final_segmented_char.append(final_segmented_char)
                # imagelist_final_segmented_char = []
                # print('l', imagelist_final_segmented_char)
    return save_state, imagelist_perchar_marker, imagelist_final_word_img, imagelist_final_segmented_char,\
        imagelist_bag_of_h_with_baseline, imagelist_image_final_body, imagelist_image_final_marker,\
        horizontal_image


def get_number_of_lines_by_result(temp_object, max_id, h):
    y_marker = []
    # get y marker coordinat
    for key in temp_object[max_id].keys():
        if type(temp_object[max_id][key]) == type(np.array([])):
            split = key.split('_')
            name = get_marker_name(key)
            if split[1] == 'tanwin':
                for c in temp_object[max_id][key]:
                    upper, lower = get_ul_coordinat(c, h)
                    y_marker.append((upper[1], lower[3]))
            else:
                for c in temp_object[max_id][key]:
                    y_marker.append((c[1], c[3]))
    # append overlapped
    y_dict = {}
    for y in y_marker:
        y_dict[y] = [y]
        for y_ in y_marker:
            if y == y_:
                continue
            if y[0] <= y_[0] <= y[1] or y[0] <= y_[1] <= y[1]:
                y_dict[y].append(y_)
    # append all overlapped and mark the same index
    temp_dict = copy.deepcopy(y_dict)
    what_same = {}
    for key in y_dict:
        found = False
        what_same[key] = [key]
        for val1 in y_dict[key]:
            if found:
                break
            for lst in y_dict.values():
                if val1 == lst[0]:
                    continue
                if val1 in lst:
                    what_same[key].append(lst[0])
                    for y in lst:
                        if y not in y_dict[key]:
                            temp_dict[key].append(y)
                    found = True
    # sort all overlapped
    longest = {}
    for key in what_same:
        l = 0
        for val in what_same[key]:
            if len(temp_dict[val]) > l:
                l = len(temp_dict[val])
                long = val
        longest[key] = val
    # check overlapped on every longest overlapped result
    latest = set(longest.values())
    temp_latest = copy.deepcopy(latest)
    last = {}
    for x in latest:
        last[x] = [x]
        for y in latest:
            if x == y:
                continue
            if x[0] <= y[0] <= x[1] or x[0] <= y[1] <= x[1]:
                last[x].append(y)
    # Filter out the same coordinat
    temp_last = copy.deepcopy(last)
    for key in last:
        temp_last[key] = False
    for key in last:
        if temp_last[key]:
            continue
        for val in last[key]:
            for key2 in last:
                if key == key2:
                    continue
                for lst in last[key2]:
                    if val == lst:
                        temp_last[key2] = True
                        break
    # count the number of lines
    line_count = 0
    for line in temp_last.values():
        if not line:
            line_count += 1
    print('number of lines: ', line_count)

    return line_count


# ## Image Processing Stage

# In[9]:


def most_frequent(List):
    return max(set(List), key=List.count)


# In[21]:


# imagePath = './temp/v1.jpg'
# img = cv2.imread(imagePath)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def font_list(imagePath, image, setting, markerPath):
    font_list, loc_path = mess.font(imagePath=imagePath, image=image,
                                    setting=setting, markerPath=markerPath)

    return font_list, loc_path

# In[22]:


# def big_blok(temp_object, imagePath, font_object, model, font_list):
def most_marker(temp_object):
    # Get the most marker
    count = -1
    temp_marker_count = {}
    for obj in temp_object:
        count += 1
        marker_count = 0
        for value in obj.values():
            if type(value) == type(np.array([])):
                marker_count += len(value)
        temp_marker_count[count] = marker_count

    max_count = 0
    max_id = None
    for x in temp_marker_count:
        if temp_marker_count[x] > max_count:
            max_count = temp_marker_count[x]
            max_id = x

    return max_id


def define_normal_or_crop_processing(imagePath, temp_object, max_id, font_object, font_list, bw_method):
    # In[23]:
    # Get the horizontal line image by using eight connectivity
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bw_method == 0:
        # Otsu threshold
        _, bw_image = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY
                                    + cv2.THRESH_OTSU)
    if bw_method == 1:
        # Simple threshold
        _, bw_image = cv2.threshold(gray, 127, 255,
                                    cv2.THRESH_BINARY)
    if bw_method == 2:
        # Adaptive threshold value is the mean of neighbourhood area
        bw_image = cv2.adaptiveThreshold(gray, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    if bw_method == 3:
        # Adaptive threshold value is the weighted sum of neighbourhood
        # values where weights are a gaussian window
        bw_image = cv2.adaptiveThreshold(gray, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    height, width = gray.shape
    # list_start_point_h = []
    imagelist_horizontal_line_by_eight_conn = []
    # continue_flag = False
    # for key in temp_object[max_id].keys():
    #     if type(temp_object[max_id][key]) == type(np.array([])):
    #         split = key.split('_')
    #         name = get_marker_name(key)
    #         if split[1] == 'tanwin':
    #             print(name)
    #             for c in temp_object[max_id][key]:
    #                 start_point_h, image_process = eight_conn_by_seed_tanwin(
    #                     c, img, font_list, False)
    #                 imagelist_horizontal_line_by_eight_conn.append(image_process)
    #                 list_start_point_h.append(start_point_h)
    #                 # print(list_start_point_h)
    #         else:
    #             print(name)
    #             for c in temp_object[max_id][key]:
    #                 start_point_h, image_process = eight_conn_by_seed(
    #                     c, img, font_list, False)
    #                 imagelist_horizontal_line_by_eight_conn.append(image_process)
    #                 list_start_point_h.append(start_point_h)
    #                 # print(list_start_point_h)

    max_height = 0
    max_width = 0
    bp_max = 0
    other_than_tanwin = False
    for key in temp_object[max_id].keys():
        if type(temp_object[max_id][key]) == type(np.array([])):
            split = key.split('_')
            name = get_marker_name(key)
            if split[1] == 'tanwin':
                # print(name)
                for c in temp_object[max_id][key]:
                    temp_h = c[3] - c[1]
                    temp_w = c[2] - c[0]
                    bp = black_pixel_count(bw_image, c)
                    if temp_h > max_height:
                        max_height = temp_h
                    if temp_w > max_width:
                        max_width = temp_w
                    if bp > bp_max:
                        bp_max = bp
            else:
                # print(name)
                other_than_tanwin = True
                for c in temp_object[max_id][key]:
                    temp_h = c[3] - c[1]
                    temp_w = c[2] - c[0]
                    bp = black_pixel_count(bw_image, c)
                    if temp_h > max_height:
                        max_height = temp_h
                    if temp_w > max_width:
                        max_width = temp_w
                    if bp > bp_max:
                        bp_max = bp

    if not other_than_tanwin:
        print('max height times x')
        max_height = max_height * 1
        # max_height = max_width * 1
    # max_height = 50
    print('line height: ', max_height)
    print('black pixel: ', bp_max)
    cv2.waitKey(0)

    image1 = gray
    body_v_proj = horizontal_projection(image1)
    x = body_v_proj*-1
    if max(body_v_proj) > bp_max:
        peaks, _ = find_peaks(x, distance=max_height, prominence=bp_max*(2/3))
        # peaks, _ = find_peaks(x, distance=max_height, prominence=0)
        # peaks, _ = find_peaks(x, distance=max_height, prominence=0,  threshold=[None, -0])
    else:
        print('no prominence')
        peaks, _ = find_peaks(x, distance=max_height, plateau_size=8, threshold=[None, -0])
    prominences = peak_prominences(x, peaks)[0]
    contour_heights = x[peaks] - prominences

    # plt.figure(1)
    fig = Figure()
    ax = fig.add_subplot(121)
    ax.imshow(image1)
    bx = fig.add_subplot(122)
    bx.plot(body_v_proj, np.arange(0, len(body_v_proj), 1))
    bx.plot(x[peaks], peaks, "x")
    bx.hlines(y=peaks, xmax=contour_heights*-1, xmin=x[peaks])
    fig.savefig('temp_h.png')
    # plt.subplot(121), plt.imshow(image1)
    # plt.subplot(122), plt.plot(
    #     body_v_proj, np.arange(0, len(body_v_proj), 1)
    # )
    # plt.plot(x[peaks], peaks, "x")
    # plt.hlines(y=peaks, xmax=contour_heights*-1, xmin=x[peaks])
    # plt.plot(np.zeros_like(x), "--", color="gray")
    # plt.savefig('temp_h.png')
    # plt.close(1)
    h_plot_img = cv2.imread('temp_h.png')
    # plt.show()
    h_point = list(peaks)
    start_end = detect_horizontal_line_up_down(body_v_proj)
    h_point.insert(0, start_end[0])
    h_point.insert(len(h_point), start_end[1])

    list_of_h_point = []
    list_h_list = []
    for x in range(len(h_point)):
        if x == len(h_point)-1:
            break
        list_h_list.append([h_point[x], h_point[x+1]])
        list_of_h_point.append(h_point[x])
        list_of_h_point.append(h_point[x+1])

    list_start_point_h = list_h_list
    list_for_mpclass = list_of_h_point
    gray_copy = gray.copy()
    height, width = gray_copy.shape
    normal_processing = []
    image_v_checking = []
    imagelist_horizontal_line_by_eight_conn.append(h_plot_img)
    max_v_width = 0
    # image_v_checking.append(h_plot_img)
    for y_y in list_start_point_h:
        image_vo = gray[y_y[0]:y_y[1], :]
        h_image_empty = True
        for x in range(0, width):
            if not h_image_empty:
                break
            for y in range(y_y[0], y_y[1]):
                if bw_image[y, x] < 1:
                    h_image_empty = False
                    break
        if h_image_empty:
            print('_Horizontal Image is Empty_')
            continue
        image_v = image_vo.copy()
        font_object.vertical_projection(image_v)
        font_object.detect_vertical_line(image_v.copy(), 5)
        start_point_v = font_object.start_point_v
        max_v = max(np.diff(start_point_v))
        if max_v > max_v_width:
            max_v_width = max_v
        # print('start point v:', start_point_v)
        for x in range(len(start_point_v)):
            if x % 2 == 0:
                print('line ', x, ' height :',
                      start_point_v[x+1] - start_point_v[x], 'pixel')
                cv2.line(image_v, (start_point_v[x], 0),
                         (start_point_v[x], height), (0, 0, 0), 2)
            else:
                cv2.line(image_v, (start_point_v[x], 0),
                         (start_point_v[x], height), (100, 100, 100), 2)
        image_v_checking.append(image_v)
        # cv2.imshow('line', image_v)
        # print('>')
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        v_point_thresh = 8
        if len(start_point_v) >= v_point_thresh \
                or (len(start_point_v) < v_point_thresh and max_v_width < max_width):  # if vpoint is smaller than largest w char
            # Go to the normal match
            normal_processing.append(True)
        else:
            # Just crop the next char by ratio
            normal_processing.append(False)
            print(len(start_point_v), 'is not enough')
    # cv2.destroyAllWindows()

    # In[25]:
    print('normal processing:', normal_processing)
    normal_processing = most_frequent(normal_processing)
    print(normal_processing)

    # In[24]:

    # img = cv2.imread(imagePath)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_copy = gray.copy()
    # height, width = gray_copy.shape
    # normal_processing = []
    # image_v_checking = []
    # for y_y in list_start_point_h:
    #     image_vo = gray[y_y[0]:y_y[1], :]
    #     image_v = image_vo.copy()
    #     font_object.vertical_projection(image_v)
    #     font_object.detect_vertical_line(image_v.copy(), 5)
    #     start_point_v = font_object.start_point_v
    #     # print('start point v:', start_point_v)
    #     for x in range(len(start_point_v)):
    #         if x % 2 == 0:
    #             cv2.line(image_v, (start_point_v[x], 0),
    #                      (start_point_v[x], height), (0, 0, 0), 2)
    #         else:
    #             cv2.line(image_v, (start_point_v[x], 0),
    #                      (start_point_v[x], height), (100, 100, 100), 2)
    #     image_v_checking.append(image_v)
    #     # cv2.imshow('line', image_v)
    #     # print('>')
    #     # cv2.waitKey(0)

    #     if len(start_point_v) > 8:
    #         # Go to the normal match
    #         normal_processing.append(True)
    #     else:
    #         # Just crop the next char by ratio
    #         normal_processing.append(False)
    #         print(len(start_point_v), 'is not enough')
    # # cv2.destroyAllWindows()

    # # In[25]:
    # print('normal processing:', normal_processing)
    # normal_processing = most_frequent(normal_processing)
    # print(normal_processing)

    # Adding horizontal checking point
    horizontal_line_is_good = True  # bypass horizontal checking
    # h_projection = font_object.horizontal_projection(gray_copy)
    # check_start_point_h = detect_horizontal_line(h_projection, 3, 3)
    # line_count = get_number_of_lines_by_result(temp_object, max_id, height)
    # print('number of h:', int(len(check_start_point_h)/2))
    # if int(len(check_start_point_h)/2) < line_count:
    #     horizontal_line_is_good = False
    # else:
    #     horizontal_line_is_good = True
    # In[26]:

    temp_gray_copy = []
    temp_image_process = []
    temp_sub_image = []
    temp_check_image = []
    temp_final_img = []
    save_state = {}
    normal_processing_result = []
    crop_ratio_processing_result = []
    if normal_processing and horizontal_line_is_good:
        print('\n___NORMAL PROCESSING___\n')
        #     pass
        # print(temp_object[max_id])
        save_state, imagelist_perchar_marker, imagelist_final_word_img, imagelist_final_segmented_char,\
            imagelist_bag_of_h_with_baseline, imagelist_image_final_body, imagelist_image_final_marker,\
            horizontal_image = normal_image_processing_blok(
                imagePath, temp_object[max_id], bw_method, list_for_mpclass)

        normal_processing_result = [imagelist_perchar_marker,
                                    imagelist_final_word_img,
                                    imagelist_final_segmented_char,
                                    imagelist_bag_of_h_with_baseline,
                                    imagelist_image_final_body,
                                    imagelist_image_final_marker,
                                    horizontal_image]
        normal_processing_result[2] = []
        temp = []
        for x in save_state:
            if len(save_state[x]) < 2:
                continue
            temp.append(copy.deepcopy(save_state[x][4]))
        normal_processing_result[2] = temp

    else:
        print('\n___CROP PROCESSING___\n')
        arr_count = -1
        for key in temp_object[max_id].keys():
            gray_copy = gray.copy()
            bw = bw_image.copy()
            if type(temp_object[max_id][key]) == type(np.array([])):
                split = key.split('_')
                name = get_marker_name(key)
                if split[1] == 'tanwin':
                    print(name)
                    for c in temp_object[max_id][key]:
                        y1_c = c[1]
                        arr_count += 1
                        c_mid = c[1] + int((c[3]-c[1])/2)
                        for y_point in list_start_point_h:
                            if y_point[0] <= c_mid <= y_point[1]:
                                selected_line = y_point
                                break
                        # oneline_coordinat = list_start_point_h[arr_count]
                        oneline_coordinat = selected_line
                        oneline_bw_image = bw_image[oneline_coordinat[0]:
                                                    oneline_coordinat[1], :]
                        cv2.rectangle(gray_copy,
                                      (0, oneline_coordinat[0]),
                                      (width, oneline_coordinat[1]),
                                      (0, 255, 0), 2)
                        cv2.rectangle(gray_copy,
                                      (c[0], c[1]),
                                      (c[2], c[3]),
                                      (0, 255, 0), 2)
                        # cv2.imshow('check', gray_copy)
                        # cv2.waitKey(0)
                        marker_only_count = black_pixel_count(bw_image, c)
                        skip_this_marker = False
                        # get up or down by tanwin position (c)
                        c = region_tanwin(c, bw_image, font_list, False)
                        next_c, _ = get_lr_coordinat(c, width)
                        next_c_count = black_pixel_count(bw_image, next_c)
                        while(next_c_count < marker_only_count):
                            next_c, _ = get_lr_coordinat(next_c, width)
                            next_c_count = black_pixel_count(bw_image, next_c)
                            if next_c[0] < 1 and next_c_count < marker_only_count:
                                skip_this_marker = True
                                break
                        if skip_this_marker:  # Skip process if next c candidate is empty
                            continue
                        # modified y coordinat
                        if next_c[1] > y1_c:
                            mod_c, _ = get_ul_coordinat(next_c, height)
                            next_c = [next_c[0], mod_c[1],
                                      next_c[2], next_c[3]]
                        else:
                            _, mod_c = get_ul_coordinat(next_c, height)
                            next_c = [next_c[0], next_c[1],
                                      next_c[2], mod_c[3]]
                        cv2.rectangle(gray_copy,
                                      (next_c[0], next_c[1]),
                                      (next_c[2], next_c[3]),
                                      (200, 150, 0), 2)
                        temp_height = c[3] - c[1]
                        crop_by = int(1/4 * temp_height)
                        crop_image = bw[next_c[1]+crop_by:next_c[3]  # -crop_by
                                        , next_c[0]:next_c[2]]
                        h_crop, w_crop = crop_image.shape
                        check_ci = black_pixel_count(
                            crop_image, [0, 0, w_crop, h_crop])
                        if check_ci < 1:
                            print('crop image is empty')
                            continue
                        one_base = raw_baseline(crop_image.copy(), font_object)
                        cv2.rectangle(gray_copy,
                                      (0, c[1] + one_base[0]),
                                      (width, c[1] + one_base[1]),
                                      (1000, 150, 0), 2)
    #                     crop_image = cv2.morphologyEx(crop_image,
    #                                                   cv2.MORPH_OPEN, kernel)
    #                     crop_image = cv2.erode(crop_image,kernel,iterations = 1)
                        # cv2.imshow('ff', crop_image)
                        # cv2.waitKey(0)
                        base = [0, one_base[0],
                                w_crop-1, one_base[1]]
                        # print(crop_image.shape, base)
                        con_pack = font_object.eight_connectivity(crop_image, base,
                                                                  left=False, right=False)
                        image_process = crop_image.copy()
                        image_process[:] = 255
                        for region in con_pack:
                            for val in con_pack[region]:
                                image_process[val] = 0
                        # cv2.imshow('d', image_process)
                        # cv2.waitKey(0)
                        sub_image = cv2.subtract(image_process, crop_image)
                        sub_image = cv2.bitwise_not(sub_image)
                        # final_c = [int(1/2*w_next), 0, w_crop, h_crop]
                        final_c = [0, 0, w_crop, h_crop]
                        check_img = sub_image[final_c[1]:final_c[3],
                                              final_c[0]:final_c[2]]
                        final_img = image_process[final_c[1]:final_c[3],
                                                  final_c[0]:final_c[2]]
                        # cv2.imshow('tanwin', check_img)
                        dot = font_object.dot_checker(check_img)
                        # print(dot)
                        # cv2.waitKey(0)
                        if dot:
                            final_img = cv2.bitwise_and(check_img, final_img)
                            # final_img = cv2.add(check_img, final_img)
                        # cv2.imshow('final', final_img)
                        # cv2.waitKey(0)

                        save_state[arr_count] = []
                        save_state[arr_count].append(name)
                        save_state[arr_count].append(1)  # scale
                        save_state[arr_count].append(
                            (next_c[1]+crop_by, next_c[3]))  # y_origin
                        save_state[arr_count].append(c)  # marker_coordinat
                        save_state[arr_count].append(final_img)
                        save_state[arr_count].append(next_c[0])
                        save_state[arr_count].append([(next_c[0], next_c[1]+crop_by),
                                                      (next_c[2], next_c[3])])

                        temp_gray_copy.append(gray_copy)
                        temp_image_process.append(image_process)
                        temp_sub_image.append(sub_image)
                        temp_check_image.append(check_img)
                        temp_final_img.append(final_img)

    #                     raw_baseline(oneline_coordinat, bw_image)
    #                     temp_height = c[3] - c[1]
    #                     crop_by = int(1/4 * temp_height)
    #                     crop_image = bw_image[c[1]+crop_by:c[3]#-crop_by
    #                                           , c[0]:c[2]]
    #                     one_base = raw_baseline(crop_image)
    #                     cv2.rectangle(gray_copy,
    #                                   (0, c[1] + one_base[0]),
    #                                   (width, c[1] + one_base[1]),
    #                                   (1000, 150,0), 2)

    #                     base = [0, one_base[0],
    #                             width, one_base[1]]
    #                     con_pack = font_object.eight_connectivity(oneline_bw_image, base,
    #                                               left=False, right=False)
    #                     image_process = oneline_bw_image.copy()
    #                     image_process[:] = 255
    #                     for region in con_pack:
    #                         for val in con_pack[region]:
    #                             image_process[val] = 0
    #                     cv2.imshow('d', image_process)
    #                     cv2.waitKey(0)

    #                     font_object.modified_eight_connectivity(oneline_bw_image, one_base)
    #                     font_object.grouping_marker()

    #                     cv2.imshow('image final marker', font_object.image_final_marker)
    #                     cv2.imshow('image final body', font_object.image_final_sorted)
    #                     cv2.waitKey(0)
                else:
                    print(name)
                    for c in temp_object[max_id][key]:
                        arr_count += 1
                        c_mid = c[1] + int((c[3]-c[1])/2)
                        for y_point in list_start_point_h:
                            if y_point[0] <= c_mid <= y_point[1]:
                                selected_line = y_point
                                break
                        # oneline_coordinat = list_start_point_h[arr_count]
                        oneline_coordinat = selected_line
                        oneline_bw_image = bw_image[oneline_coordinat[0]:
                                                    oneline_coordinat[1], :]
                        cv2.rectangle(gray_copy,
                                      (0, oneline_coordinat[0]),
                                      (width, oneline_coordinat[1]),
                                      (0, 255, 0), 2)
                        cv2.rectangle(gray_copy,
                                      (c[0], c[1]),
                                      (c[2], c[3]),
                                      (0, 255, 0), 2)
                        # cv2.imshow('check', gray_copy)
                        # cv2.waitKey(0)
                        marker_only_count = black_pixel_count(bw_image, c)
                        skip_this_marker = False
                        next_c, _ = get_lr_coordinat(c, width)
                        next_c_count = black_pixel_count(bw_image, next_c)
                        while(next_c_count < int(1/2*marker_only_count)):
                            next_c, _ = get_lr_coordinat(next_c, width)
                            next_c_count = black_pixel_count(bw_image, next_c)
                            if next_c[0] < 1 and next_c_count < int(1/2*marker_only_count):
                                skip_this_marker = True
                                break
                        if skip_this_marker:  # Skip process if next c candidate is empty
                            continue
                        mod_c, _ = get_lr_coordinat(next_c, width)
                        w_next = mod_c[2] - mod_c[0]
                        next_c = [next_c[0] - int(1/2*w_next), next_c[1],
                                  next_c[2], next_c[3]]
                        cv2.rectangle(gray_copy,
                                      (next_c[0], next_c[1]),
                                      (next_c[2], next_c[3]),
                                      (200, 150, 0), 2)
                        # cv2.imshow('check', gray_copy)
                        # cv2.waitKey(0)
                        temp_height = c[3] - c[1]
                        crop_by = int(1/4 * temp_height)
                        crop_image = bw[next_c[1]+crop_by:next_c[3]  # -crop_by
                                        , next_c[0]:next_c[2]]
                        h_crop, w_crop = crop_image.shape
                        check_ci = black_pixel_count(
                            crop_image, [0, 0, w_crop, h_crop])
                        if check_ci < 1:
                            print('crop image is empty')
                            continue
                        one_base = raw_baseline(crop_image.copy(), font_object)
                        cv2.rectangle(gray_copy,
                                      (0, c[1] + one_base[0]),
                                      (width, c[1] + one_base[1]),
                                      (1000, 150, 0), 2)
    #                     crop_image = cv2.morphologyEx(crop_image,
    #                                                   cv2.MORPH_OPEN, kernel)
    #                     crop_image = cv2.erode(crop_image,kernel,iterations = 1)
                        # cv2.imshow('ff', crop_image)
                        # cv2.waitKey(0)
                        base = [0, one_base[0],
                                w_crop-1, one_base[1]]
                        # print(base)
                        con_pack = font_object.eight_connectivity(crop_image, base,
                                                                  left=False, right=False)
                        image_process = crop_image.copy()
                        image_process[:] = 255
                        for region in con_pack:
                            for val in con_pack[region]:
                                image_process[val] = 0
                        # cv2.imshow('d', image_process)
                        # cv2.waitKey(0)
                        sub_image = cv2.subtract(image_process, crop_image)
                        sub_image = cv2.bitwise_not(sub_image)
                        final_c = [int(1/2*w_next), 0, w_crop, h_crop]
                        check_img = sub_image[final_c[1]:final_c[3],
                                              final_c[0]:final_c[2]]
                        final_img = image_process[final_c[1]:final_c[3],
                                                  final_c[0]:final_c[2]]
                        # cv2.imshow('mim', check_img)
                        dot = font_object.dot_checker(check_img)
                        # print(dot)
                        # cv2.waitKey(0)
                        if dot:
                            print('dot')
                            final_img = cv2.bitwise_and(final_img, check_img)
                            # final_img = cv2.add(final_img, check_img)
                        # cv2.imshow('final', final_img)
                        # cv2.waitKey(0)

                        save_state[arr_count] = []
                        save_state[arr_count].append(name)
                        save_state[arr_count].append(1)  # scale
                        save_state[arr_count].append(
                            (next_c[1]+crop_by, next_c[3]))  # y_origin
                        save_state[arr_count].append(c)  # marker_coordinat
                        save_state[arr_count].append(final_img)
                        save_state[arr_count].append(
                            next_c[0] + int(1/2*w_next))
                        save_state[arr_count].append([(next_c[0] + int(1/2*w_next),
                                                       next_c[1]+crop_by),
                                                      (next_c[2], next_c[3])])

                        temp_gray_copy.append(gray_copy)
                        temp_image_process.append(image_process)
                        temp_sub_image.append(sub_image)
                        temp_check_image.append(check_img)
                        temp_final_img.append(final_img)

        # temp_image_process = eight conn result on baseline
        # check image = sub_image cutted
        crop_ratio_processing_result = [temp_gray_copy,
                                        temp_image_process,
                                        temp_sub_image,
                                        temp_check_image,
                                        temp_final_img]
    #                     raw_baseline(oneline_coordinat, bw_image)
    #                     temp_height = c[3] - c[1]
    #                     crop_by = int(1/4 * temp_height)
    #                     crop_image = bw_image[c[1]+crop_by:c[3]#-crop_by
    #                                           , c[0]:c[2]]
    #                     one_base = raw_baseline(crop_image)
    #                     cv2.rectangle(gray_copy,
    #                                   (0, c[1] + one_base[0]),
    #                                   (width, c[1] + one_base[1]),
    #                                   (1000, 150,0), 2)

    #                     base = [0, one_base[0],
    #                             width, one_base[1]]
    #                     con_pack = font_object.eight_connectivity(oneline_bw_image, base,
    #                                               left=False, right=False)
    #                     image_process = oneline_bw_image.copy()
    #                     image_process[:] = 255
    #                     for region in con_pack:
    #                         for val in con_pack[region]:
    #                             image_process[val] = 0
    #                     cv2.imshow('d', image_process)
    #                     cv2.waitKey(0)

    #                     font_object.modified_eight_connectivity(oneline_bw_image, one_base)
    #                     font_object.grouping_marker()

    #                     cv2.imshow('image final marker', font_object.image_final_marker)
    #                     cv2.imshow('image final body', font_object.image_final_sorted)
    #                     cv2.waitKey(0)

    return save_state, normal_processing_result, crop_ratio_processing_result, imagelist_horizontal_line_by_eight_conn, image_v_checking


def draw_bounding_box(img, coordinat, label, color, font_scale=1, font=cv2.FONT_HERSHEY_PLAIN):
    x1 = coordinat[0]
    y1 = coordinat[1]
    x2 = coordinat[2]
    y2 = coordinat[3]
    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color=color,
        thickness=2
    )
    ((label_width, label_height), _) = cv2.getTextSize(
        label,
        fontFace=font,
        fontScale=font_scale,
        thickness=2
    )

    cv2.rectangle(
        img,
        (int(x1), int(y2)),
        (int(x1 + label_width + label_width * 0.05),
         #  int(y1 - label_height - label_height * 0.25)),
         int(y2 + label_height + label_height * 0.25)),
        color=color,
        thickness=cv2.FILLED
    )
    cv2.putText(
        img,
        label,
        # org=(int(x1), int(y1 - label_height - label_height * 0.25)),
        # org=(int(x1), int(y1)),
        org=(int(x1), int(y2 + label_height + label_height * 0.25)),
        fontFace=font,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=2
    )
    return img


def character_recognition(save_state, imagePath, model):
    # ## Final Recognition Stage
    # In[18]:
    iqlab = [1]
    idgham_bigunnah = [23, 24, 25, 27]
    idgham_bilagunnah = [9, 22]
    idzhar_halqi = [0, 26, 17, 18, 5, 6]
    ikhfa_hakiki = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21]
    ikhfa_syafawi = [1]
    idgham_mimi = [23]
    idzhar_syafawi = [23, 1]  # NOT
    font_text = cv2.FONT_HERSHEY_PLAIN
    WHITE = (255, 0, 0)
    GREEN = (0, 255, 0)

    # In[27]:

    original_image = cv2.imread(imagePath)
    # save_state = image_processing_blok(imagePath)
    #     print(save_state)
    # Final segmented char recognition
    char_recog = []
    saved_char_recog = []
    skip_index = []
    character_y = {}
    for x in save_state:
        if len(save_state[x]) < 2:
            continue
        # cv2.imshow('save state image', save_state[x][4])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # h, w = save_state[x][4].shape
        start_point = detect_horizontal_line_up_down(
            horizontal_projection(save_state[x][4]))
    #     print(x)
        print(start_point)
        if len(start_point) < 2:
            skip_index.append(x)
            continue
        elif len(start_point) >= 2:
            if start_point[1] - start_point[0] < 3:
                skip_index.append(x)
                continue
        cut_image = save_state[x][4][start_point[0]:start_point[1], :]
        character_y[x] = (save_state[x][2][0]+start_point[0],
                          save_state[x][2][0]+start_point[1])

    #     cv2.imshow('r', cut_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        # DD
        image_32_dd = cv2.resize(cut_image, (32, 32))
        # DK
    #         square_img = concat_image(cut_image)
    #         image_32_dk = cv2.resize(square_img, (32, 32))
    #         plt.figure(x)
    #         plt.imshow(image_32_dd, cmap='gray')
    #         char_recog.append(image_32_dk)
        char_recog.append(image_32_dd)

    saved_char_recog = char_recog.copy()
    if len(save_state) > 0:
        char_recog = np.array(char_recog)
        char_recog = char_recog.reshape(-1, 32, 32, 1).astype(np.float32)/255

        y_pred = model.predict(char_recog)
        y_pred = np.argmax(y_pred, axis=1)

    count = -1
    char_list_nameonly = [
        'Alif‬', 'Bā’', 'Tā’', 'Ṡā’‬', 'Jīm', 'h_Ḥā’‬', 'Khā’‬',
        'Dāl‬', 'Żāl‬', 'Rā’‬', 'zai‬', 'sīn‬', 'syīn‬', 's_ṣād',
        'd_ḍād', 't_ṭā’‬', 'z_ẓȧ’‬', '‘ain', 'gain‬', 'fā’‬', 'qāf‬',
        'kāf‬', 'lām‬', 'mīm‬', 'nūn‬', 'wāw‬', 'hā’‬', 'yā’‬'
    ]
    pred_result = []
    char_count = {}
    char_count['IDM'] = 0
    char_count['IDG'] = 0
    char_count['IDL'] = 0 
    char_count['IZH'] = 0
    char_count['IZS'] = 0
    char_count['IKH'] = 0
    char_count['IKS'] = 0
    char_count['IQB'] = 0
    for x in save_state:
        # skipping
        skip = False
        for index in skip_index:
            if x == index:
                skip = True
                break
        if len(save_state[x]) < 2:
            # pred_result.append('n/a')
            # skip_index.append(x)
            continue
        elif skip:
            pred_result.append('n/a')
            continue
        else:
            count += 1
            # print('y_pred', y_pred)
            # pred_result.append(y_pred[count])
            pred_result.append(char_list_nameonly[y_pred[count]])
            # print(pred_result)
            # print(char_list_nameonly[y_pred[count]])

        h, w = save_state[x][4].shape
        marker = save_state[x][0].split('_')
        if isinstance(save_state[x][3], type([])):
            marker_coordinat = np.array(save_state[x][3])
        elif isinstance(save_state[x][3], type(np.array([]))):
            marker_coordinat = save_state[x][3]
        char_box = marker_coordinat / save_state[x][1]
        char_box = [int(char_box[0]), int(char_box[1]),
                    int(char_box[2]), int(char_box[3])]
        # print('char box', char_box)
    #         final_box = [
    #             int(char_box[0]) - int(w / save_state[x][1]),
    #             int(char_box[1]) + save_state[x][2], int(char_box[2]),
    #             int(char_box[3]) + save_state[x][2] + int(char_box[3]) - int(char_box[1])
    #         ]
    #     final_box = [
    #         int(save_state[x][5] / save_state[x][1]),
    #         save_state[x][2], int(char_box[2]),
    #         int(char_box[3]) + save_state[x][2]
    #     ]
        # final_box = [
        #     int(save_state[x][5] / save_state[x][1]),
        #     int(char_box[1]),
        #     int(char_box[2]), int(char_box[3])
        # ]
        if char_box[1] < character_y[x][0]:
            y1 = char_box[1]
        else:
            y1 = character_y[x][0]
        if char_box[3] > character_y[x][1]:
            y2 = char_box[3]

        else:
            y2 = character_y[x][1]

        final_box = [
            int(save_state[x][5] / save_state[x][1]), y1,
            int(char_box[2]), y2
        ]
        found = False
        coordinat = [final_box[0], final_box[1], final_box[2], final_box[3]]
        if marker[0] == 'nun' or marker[0] == 'tanwin':
            if y_pred[count] == iqlab[0]:
                char_count['IQB'] += 1
                original_image = draw_bounding_box(original_image,
                                                   coordinat,
                                                   'IQB_'+str(char_count['IQB']),
                                                   (8, 0, 255))
                print('iqlab')
                continue
            for c in idgham_bilagunnah:
                if y_pred[count] == c:
                    print('idgham bilagunnah')
                    char_count['IDL'] += 1
                    original_image = draw_bounding_box(original_image,
                                                       coordinat,
                                                       'IDL_'+str(char_count['IDL']),
                                                       (0, 153, 0))
                    found = True
                    break
            if found:
                continue
            for c in idgham_bigunnah:
                if y_pred[count] == c:
                    print('idgham bigunnah')
                    char_count['IDG'] += 1
                    original_image = draw_bounding_box(original_image,
                                                       coordinat,
                                                       'IDG_'+str(char_count['IDG']),
                                                       (0, 255, 0))
                    found = True
                    break
            if found:
                continue
            for c in idzhar_halqi:
                if y_pred[count] == c:
                    print('idzhar halqi')
                    char_count['IZH'] += 1
                    original_image = draw_bounding_box(original_image,
                                                       coordinat,
                                                       'IZH_'+str(char_count['IZH']),
                                                       (204, 0, 0))
                    found = True
                    break
            if found:
                continue
            for c in ikhfa_hakiki:
                if y_pred[count] == c:
                    print('ikhfa hakiki')
                    char_count['IKH'] += 1
                    original_image = draw_bounding_box(original_image,
                                                       coordinat,
                                                       'IKH_'+str(char_count['IKH']),
                                                       (96, 96, 96))
                    found = True
                    break
            if found:
                continue

        elif marker[0] == 'mim':
            if y_pred[count] == ikhfa_syafawi[0]:
                print('ikfha syafawi')
                char_count['IKS'] += 1
                original_image = draw_bounding_box(original_image,
                                                   coordinat,
                                                   'IKS_'+str(char_count['IKS']),
                                                   (128, 128, 128))
                continue
            elif y_pred[count] == idgham_mimi[0]:
                print('idgham mimi')
                char_count['IDM'] += 1
                original_image = draw_bounding_box(original_image,
                                                   coordinat,
                                                   'IDM_'+str(char_count['IDM']),
                                                   (102, 255, 102))
                continue
            else:
                print('idzhar syafawi')
                char_count['IZS'] += 1
                original_image = draw_bounding_box(original_image,
                                                   coordinat,
                                                   'IZS_'+str(char_count['IZS']),
                                                   (255, 102, 102))

    # cv2.imshow('Final Result', original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(pred_result)
    final_image_result = original_image

    return final_image_result, pred_result, saved_char_recog, skip_index

# DONE!!!


# %%
