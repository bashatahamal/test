import copy
import numpy as np
# import cv2


class ImageProcessing():

    def vertical_projection(self, image_v):
        image = image_v.copy()
        image[image < 127] = 1
        image[image >= 127] = 0
        self.v_projection = np.sum(image, axis=0)

        return self.v_projection

    def horizontal_projection(self, image_h):
        image = image_h.copy()
        image[image < 127] = 1
        image[image >= 127] = 0
        self.h_projection = np.sum(image, axis=1)

        return self.h_projection

    def detect_horizontal_line(self, image, pixel_limit_ste, pixel_limit_ets,
                               view=True):
        # Detect line horizontal
        if len(image.shape) == 3:
            height, width, _ = image.shape
            # color_temp = image.copy()
        else:
            height, width = image.shape
        h_projection = self.h_projection
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

        self.start_point_h = start_point

    def detect_vertical_line(self, image, pixel_limit_ste, view=True):
        # Detect line vertical
        v_projection = self.v_projection
        # print(v_projection)
        # original_image = image
        up_flag = 0
        down_flag = 0
        start_to_end = 0
        # end_to_start = pixel_limit_ets + 1
        start_point = []
        for x in range(len(v_projection)):
            if v_projection[x] > 0 and down_flag == 0:
                start_to_end += 1

            if v_projection[x] == 0 and up_flag == 1:
                # print(start_to_end)
                start_point.append(x)
                # print(start_point)
                if start_to_end < pixel_limit_ste:
                    del(start_point[len(start_point) - 1])
                    del(start_point[len(start_point) - 1])
                down_flag = 1
                up_flag = 0
                start_to_end = 0

            if v_projection[x] > 0 and up_flag == 0:
                start_point.append(x)
                up_flag = 1
                down_flag = 0
                # end_to_start = 0

        if len(start_point) % 2 != 0:
            if v_projection[len(v_projection) - 1] > 0:
                start_point.append(len(v_projection) - 1)
        self.start_point_v = start_point

    def find_connectivity(self, x, y, height, width, image):
        # count = 0
        x_y = []
        # Left
        if x - 1 > 0:
            if y + 1 < height:
                if image[y + 1, x - 1] < 127:
                    x_y.append((y + 1, x - 1))
            if image[y, x - 1] < 127:
                x_y.append((y, x - 1))
            if y - 1 > 0:
                if image[y - 1, x - 1] < 127:
                    x_y.append((y - 1, x - 1))
        # Middle
        if y + 1 < height:
            if image[y + 1, x] < 127:
                x_y.append((y + 1, x))
        x_y.append((y, x))
        if y - 1 > 0:
            if image[y - 1, x] < 127:
                x_y.append((y - 1, x))
        # Right
        if x + 1 < width:
            if y + 1 < height:
                if image[y + 1, x + 1] < 127:
                    x_y.append((y + 1, x + 1))
            if image[y, x + 1] < 127:
                x_y.append((y, x + 1))
            if y - 1 > 0:
                if image[y - 1, x + 1] < 127:
                    x_y.append((y - 1, x + 1))

        return x_y

    def eight_connectivity(self, image):
        height, width = image.shape
        image_process = image.copy()
        # For image flag
        image_process[:] = 255
        self.conn_pack = {}
        reg = 1
        # Doing eight conn on every pixel one by one
        for x in range(width):
            for y in range(height):
                if image_process[y, x] == 0:
                    continue
                if image[y, x] < 127:
                    self.conn_pack['region_{:03d}'.format(reg)] = []
                    x_y = self.find_connectivity(x, y, height, width, image)
                    length_ = len(self.conn_pack['region_{:03d}'.format(reg)])
                    for val in x_y:
                        self.conn_pack['region_{:03d}'.format(reg)].append(val)
                    # print(self.conn_pack['region_{:03d}'.format(reg)])
                    # cv2.waitKey(0)
                    first = True
                    sub = False
                    init = True
                    while(True):
                        if first:
                            length_ = length_
                            first = False
                        else:
                            length_ = l_after_sub

                        if init:
                            l_after_init = len(self.conn_pack[
                                'region_{:03d}'.format(reg)])
                            for k in range(length_, l_after_init):
                                x_y_sub = self.find_connectivity(
                                    self.conn_pack[
                                        'region_{:03d}'.format(reg)][k][1],
                                    self.conn_pack[
                                        'region_{:03d}'.format(reg)][k][0],
                                    height, width, image
                                )
                                if len(x_y_sub) > 1:
                                    sub = True
                                    for vl in x_y_sub:
                                        if vl not in self.conn_pack[
                                                'region_{:03d}'.format(reg)]:
                                            self.conn_pack[
                                                'region_{:03d}'.format(reg)
                                                ].append(vl)
                        init = False
                        if sub:
                            # print(self.conn_pack['region_{:03d}'.format(reg)])
                            # cv2.waitKey(0)
                            l_after_sub = len(self.conn_pack[
                                'region_{:03d}'.format(reg)])
                            for k in range(l_after_init, l_after_sub):
                                x_y_sub = self.find_connectivity(
                                    self.conn_pack[
                                        'region_{:03d}'.format(reg)][k][1],
                                    self.conn_pack[
                                        'region_{:03d}'.format(reg)][k][0],
                                    height, width, image
                                )
                                if len(x_y_sub) > 1:
                                    init = True
                                    for vl in x_y_sub:
                                        if vl not in self.conn_pack[
                                                'region_{:03d}'.format(reg)]:
                                            self.conn_pack[
                                                'region_{:03d}'.format(reg)
                                                ].append(vl)
                        sub = False

                        if not sub and not init:
                            break

                    for val in self.conn_pack['region_{:03d}'.format(reg)]:
                        image_process[val] = 0
                    reg += 1
                    # cv2.imshow('eight conn process', image_process)
                    # print(self.conn_pack)
                    # cv2.waitKey(0)

        # Get every region length
        temp_max = 0
        for key in self.conn_pack:
            if len(self.conn_pack[key]) > temp_max:
                temp_max = len(self.conn_pack[key])
                key_max = key
        self.conn_pack_minus_body = copy.deepcopy(self.conn_pack)
        # The longest region it's the body
        for key in self.conn_pack_minus_body:
            if key == key_max:
                continue
            else:
                del(self.conn_pack[key])
        # Get body only and minus body region
        del(self.conn_pack_minus_body[key_max])
        # Paint body only region
        self.image_body = image.copy()
        # for region in self.conn_pack_minus_body:
        #     value = self.conn_pack_minus_body[region]
        #     for x in value:
        #         self.image_body[x] = 255
        self.image_body[:] = 255
        for region in self.conn_pack:
            value = self.conn_pack[region]
            for x in value:
                self.image_body[x] = image[x]

    def dot_checker(self, image_marker):
        # Dot detection
        self.horizontal_projection(image_marker)
        self.detect_horizontal_line(image_marker.copy(), 0, 0, False)
        one_marker = image_marker[self.start_point_h[0]:
                                  self.start_point_h[1], :]
        self.vertical_projection(one_marker)
        self.detect_vertical_line(one_marker.copy(), 0, False)
        x1 = self.start_point_v[0]
        x2 = self.start_point_v[1]
        one_marker = one_marker[:, x1:x2]
        height, width = one_marker.shape
        scale = 1.3
        write_canvas = False
        # Square, Portrait or Landscape image
        if width < scale * height:
            if height < scale * width:
                # print('_square_')
                black = False
                white = False
                middle_hole = False
                # Possibly sukun
                for y in range(height):
                    if one_marker[y, round(width/2)] < 127:
                        black = True
                    if black and one_marker[y, round(width/2)] >= 127:
                        white = True
                    if white and one_marker[y, round(width/2)] < 127:
                        middle_hole = True
                if middle_hole:
                    # print('_white hole in the middle_')
                    write_canvas = False
                else:
                    # Checking all pixel
                    white_hole = False
                    for x in range(width):
                        if white_hole:
                            break
                        black = False
                        white = False
                        white_val = 0
                        for y in range(height):
                            if one_marker[y, x] >= 127:
                                white_val += 1
                            if one_marker[y, x] < 127:
                                black = True
                            if black and one_marker[y, x] >= 127:
                                white = True
                            if white and one_marker[y, x] < 127:
                                white_hole = True
                                break
                    if white_hole:
                        # print('_there is a hole_')
                        write_canvas = False
                    else:
                        # Check on 2nd one third region
                        touch_up = False
                        touch_down = False
                        for x in range(round(width/4)-1, 3*round(width/4)):
                            if one_marker[0, x] < 127:
                                touch_up = True
                            if one_marker[height-1, x] < 127:
                                touch_down = True
                            if touch_up and touch_down:
                                break
                        # Check on after 1/5(mitigate noise) till 1/2
                        if touch_up and touch_down:
                            too_many_whites = False
                            for x in range(round(width/5), round(width/2)):
                                white_val = 0
                                for y in range(height):
                                    if one_marker[y, x] >= 127:
                                        white_val += 1
                                if white_val > round(height/1.5):
                                    too_many_whites = True
                                    break
                            if too_many_whites:
                                # print('_too many white value in 1/5 till 1/2_')
                                write_canvas = False
                            else:
                                # print('_DOT CONFIRM_')
                                write_canvas = True
                        # else:
                        #     print('not touching')

                if not write_canvas:
                    # Split image into two vertically and looking for bwb
                    # (Kaf Hamzah)
                    # bwb_up = False
                    bwb_down = False
                    bwb_count = 0
                    bwb_thresh = round(height/2.1)
                    addition = round(height/8)
                    up_limit = round(height/2)
                    down_limit = round(height/2)
                    for x in range(width):
                        if bwb_count > bwb_thresh:
                            break
                        black = False
                        white = False
                        for y in range(0, up_limit):
                            if one_marker[y, x] < 127:
                                black = True
                            if black and one_marker[y, x] >= 127:
                                white = True
                            if white and one_marker[y, x] < 127:
                                # bwb_up = True
                                bwb_count += 1
                                break
                    for x in range(width):
                        if bwb_count > bwb_thresh and bwb_down:
                            break
                        black = False
                        white = False
                        for y in range(down_limit, height):
                            if one_marker[y, x] < 127:
                                black = True
                            if black and one_marker[y, x] >= 127:
                                white = True
                            if white and one_marker[y, x] < 127:
                                bwb_down = True
                                bwb_count += 1
                                break
                    # Check for possible dammahtanwin on last 1/4 region
                    # if to many repeated bw then it's dammahtanwin
                    bw_max = 0
                    for x in range(3*round(width/4) + 1, width):
                        black = False
                        bw = False
                        bw_count = 0
                        for y in range(height):
                            if one_marker[y, x] < 127:
                                black = True
                            if black and one_marker[y, x] >= 127:
                                bw = True
                            if bw:
                                bw_count += 1
                                black = False
                                bw = False
                        if bw_count > bw_max:
                            bw_max = bw_count
                    if bwb_count >= bwb_thresh and bwb_down and bw_max < 3:
                        # print('_KAF HAMZAH CONFIRM_')
                        write_canvas = True
                    else:
                        # print('_also not kaf hamzah_')
                        write_canvas = False
            else:
                # print('_portrait image_')
                # Split image into two vertically and looking for bwb
                # (Kaf Hamzah)
                bwb_up = False
                bwb_down = False
                for x in range(width):
                    if bwb_up:
                        break
                    black = False
                    white = False
                    for y in range(0, round(height/2)):
                        if one_marker[y, x] < 127:
                            black = True
                        if black and one_marker[y, x] >= 127:
                            white = True
                        if white and one_marker[y, x] < 127:
                            bwb_up = True
                            break
                for x in range(width):
                    if bwb_down:
                        break
                    black = False
                    white = False
                    for y in range(round(height/2), height):
                        if one_marker[y, x] < 127:
                            black = True
                        if black and one_marker[y, x] >= 127:
                            white = True
                        if white and one_marker[y, x] < 127:
                            bwb_down = True
                            break
                if bwb_up and bwb_down:
                    # print('_KAF HAMZAH CONFIRM_')
                    write_canvas = True
                else:
                    write_canvas = False
        else:
            # print('_landscape image_')
            black = False
            white = False
            wbw_confirm = False
            over_pattern = False
            # Possibly straight harakat or tasdid
            for y in range(height):
                if one_marker[y, round(width/2)] >= 127:
                    white = True
                if white and one_marker[y, round(width/2)] < 127:
                    black = True
                if black and one_marker[y, round(width/2)] >= 127:
                    wbw_confirm = True
                if wbw_confirm and one_marker[y, round(width/2)] < 127:
                    over_pattern = True
                    break
            if over_pattern:
                # print('_too many wbw + b_')
                write_canvas = False
            elif wbw_confirm:
                # print('_mid is wbw_')
                too_many_white_val = False
                # cut in the middle up vertically wether the pixel all white
                for x in range(round(width/5), round(width/3)):
                    white_val = 0
                    for y in range(0, height):
                        if one_marker[y, x] >= 127:
                            white_val += 1
                    if white_val > round(height/1.9):
                        too_many_white_val = True
                        break
                if too_many_white_val:
                    # print('_too many white val in 1/5 till 1/3_')
                    write_canvas = False
                else:
                    half_img = one_marker[:, 0:round(width/2)]
                    self.horizontal_projection(half_img)
                    self.detect_horizontal_line(half_img.copy(), 0, 0, True)
                    half_img = one_marker[
                        self.start_point_h[0]:self.start_point_h[1],
                        0:round(width/2)
                    ]
                    half_height, half_width = half_img.shape
                    # print(half_height, half_width)
                    # one_3rd = round(half_width/3)
                    # one_4th = round(half_width/4)
                    one_8th = round(half_width/8)
                    touch_up = False
                    touch_down = False
                    # for x in range(one_3rd-1, 2*one_3rd):
                    # for x in range(one_4th-1, 3*one_4th):
                    for x in range(one_8th-1, 7*one_8th):
                        if half_img[0, x] < 127:
                            touch_up = True
                        if half_img[half_height-1, x] < 127:
                            touch_down = True
                        if touch_up and touch_down:
                            break
                    if touch_up and touch_down:
                        # print('_DOT CONFIRM_')
                        write_canvas = True
                    else:
                        # print('_not touching_')
                        write_canvas = False
            else:
                # print('_middle is not wbw_')
                write_canvas = False
                # Split image into two vertically and looking for bwb
                # (Kaf Hamzah)
                bwb_up = False
                bwb_down = False
                for x in range(width):
                    if bwb_up:
                        break
                    black = False
                    white = False
                    for y in range(0, round(height/2)):
                        if one_marker[y, x] < 127:
                            black = True
                        if black and one_marker[y, x] >= 127:
                            white = True
                        if white and one_marker[y, x] < 127:
                            bwb_up = True
                            break
                for x in range(width):
                    if bwb_down:
                        break
                    black = False
                    white = False
                    for y in range(round(height/2), height):
                        if one_marker[y, x] < 127:
                            black = True
                        if black and one_marker[y, x] >= 127:
                            white = True
                        if white and one_marker[y, x] < 127:
                            bwb_down = True
                            break
                if bwb_up and bwb_down:
                    # print('_KAF HAMZAH CONFIRM_')
                    write_canvas = True
                else:
                    write_canvas = False

        return write_canvas


def body_and_dot_region(image):
    process = ImageProcessing()
    process.eight_connectivity(image)
    image_marker = image.copy()
    for region in process.conn_pack_minus_body:
        value = process.conn_pack_minus_body[region]
        if len(value) < 3:
            continue
        image_marker[:] = 255
        for x in value:
            image_marker[x] = image[x]
        # print(len(value))
        dot = process.dot_checker(image_marker)
        # print(str(dot))
        if dot:
            for x in value:
                process.image_body[x] = image[x]
        else:
            continue

    return process.image_body
