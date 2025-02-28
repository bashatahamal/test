import matplotlib.pyplot as plt
import numpy as np
import imutils
import glob
import cv2
import copy
import time


class Marker:
    def __init__(self, **kwargs):
        self._Data = kwargs

    def get_template_location(self):
        return self._Data["template_loc"]
    # def get_image_location(self):
    #     return self._Data["image_loc"]

    def get_template_thresh(self):
        return self._Data["template_thresh"]

    def get_nms_thresh(self):
        return self._Data["nms_thresh"]

    def get_image(self):
        return self._Data["image"]

    def non_max_suppression_slow(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list, add the index
            # value to the list of picked indexes, then initialize
            # the suppression list (i.e. indexes that will be deleted)
            # using the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]

            # loop over all indexes in the indexes list
            for pos in range(0, last):
                # grab the current index
                j = idxs[pos]

                # find the largest (x, y) coordinates for the start of
                # the bounding box and the smallest (x, y) coordinates
                # for the end of the bounding box
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                # compute the width and height of the bounding box
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / area[j]

                # if there is sufficient overlap, suppress the
                # current bounding box
                if overlap > overlapThresh:
                    suppress.append(pos)

            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
        return boxes[pick]

    def match_template(self, visualize=False, numstep=10, bw_method=0):
        # Get template
        # print('.')
        template = cv2.imread(self.get_template_location())
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # Canny edge
        # template = cv2.Canny(template, 50, 200)
        # Otsu threshold
        ret_temp, template = cv2.threshold(template, 0, 255,
                                           cv2.THRESH_BINARY
                                           + cv2.THRESH_OTSU)
        # Simple threshold
        # ret_temp, template = cv2.threshold(template, 127, 255,
        #                                    cv2.THRESH_BINARY)
        # Adaptive threshold value is the mean of neighbourhood area
        # template = cv2.adaptiveThreshold(template, 255,
        #                                  cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                  cv2.THRESH_BINARY,
        #                                  11, 2)

        # Adaptive threshold value is the weighted sum of neighbourhood values
        # where weights are a gaussian window
        # template = cv2.adaptiveThreshold(template, 255,
        #                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                  cv2.THRESH_BINARY,
        #                                  11, 2)
        # cv2.imshow("Template", template)
        # plt.imshow(template)

        (tH, tW) = template.shape[:2]

        # Get image
        # image = cv2.imread(self.get_image_location())
        # print(self.get_image_location())
        # self.image=image
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)
        image = self.get_image()
        # boundingBoxes = np.ones((1,4), dtype=int)
        boundingBoxes = []
        max_value_list = []
        self.imagelist_visualize = []
        self.image_visualize_white_block = []
        # print(numstep)
        if numstep < 1:
            start_scale = 1
            end_scale = 1
            numstep = 1
        else:
            start_scale = 0.2
            end_scale = 2.0
        # Loop over scaled image (start stop numstep) from the back
        # t1 = time.time()
        temp_scale = np.linspace(start_scale, end_scale, numstep)
        to_low = []
        to_up = []
        to_scale = []
        mid = int(len(temp_scale)/2)
        for x in range(mid)[::-1]:
            to_low.append(temp_scale[x])
        for x in range(mid, len(temp_scale)):
            to_up.append(temp_scale[x])
        if len(to_low) >= len(to_up):
            half = len(to_low)
        else:
            half = len(to_up)
        for x in range(half):
            if x <= len(to_low)-1:
                to_scale.append(to_low[x])
            if x <= len(to_up)-1:
                to_scale.append(to_up[x])
        print(to_scale)

        # for scale in np.linspace(start_scale, end_scale, numstep)[::-1]:
        for scale in to_scale:
            print(scale)
            resized = imutils.resize(image, width=int(image.shape[1] * scale))
            r = image.shape[1] / float(resized.shape[1])
            # If resized image smaller than template, then break the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # Preprocessing resized image and then apply template matching
            # Cannyedge
            # edged = cv2.Canny(resized, 50, 200)
            if bw_method == 0:
                # Otsu threshold
                ret_img, resized = cv2.threshold(resized, 0, 255,
                                                 cv2.THRESH_BINARY
                                                 + cv2.THRESH_OTSU)
            if bw_method == 1:
                # Simple threshold
                ret_img, resized = cv2.threshold(resized, 127, 255,
                                                 cv2.THRESH_BINARY)
            if bw_method == 2:
                # Adaptive threshold value is the mean of neighbourhood area
                resized = cv2.adaptiveThreshold(resized, 255,
                                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
            if bw_method == 3:
                # Adaptive threshold value is the weighted sum of neighbourhood
                # values where weights are a gaussian window
                resized = cv2.adaptiveThreshold(resized, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if visualize:
                # clone = np.dstack([edged, edged, edged])
                clone = np.dstack([resized, resized, resized])
                # print(self.get_template_location())
                # print(maxVal)
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                              (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                self.imagelist_visualize.append(clone)
                # cv2.imshow("Visualizing", clone)
                # cv2.waitKey(0)
                # plt.imshow(clone)

            loc = np.where( result >= self.get_template_thresh())
            print(loc)
            for maxLoc in zip(*loc[::-1]):
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r),
                                int((maxLoc[1] + tH) * r))
                temp = [startX, startY, endX, endY]
                boundingBoxes.append(temp)
                max_value_list.append(maxVal)
            
            if len(loc[0]) > 0:
                break

            # if maxVal > self.get_template_thresh():
            #     (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            #     (endX, endY) = (int((maxLoc[0] + tW) * r),
            #                     int((maxLoc[1] + tH) * r))
            #     temp = [startX, startY, endX, endY]
            #     # boundingBoxes = np.append(boundingBoxes, [temp], axis=0)
            #     # max_value_list = np.append(max_value_list, [maxVal], axis=0)
            #     boundingBoxes.append(temp)
            #     max_value_list.append(maxVal)
            #     # print("Max val = {} location {}".format(maxVal, temp))
            #     cv2.rectangle(image, (startX, startY), (endX, endY),
            #                   (255, 255, 255), -1)
            #     cv2.rectangle(resized, (int(maxLoc[0]), int(maxLoc[1])),
            #                   (int(maxLoc[0] + tW), int(maxLoc[1] + tH)),
            #                   (255, 255, 255), -1)
            #     while(maxVal > self.get_template_thresh()):
            #         print('checking in scale size: ', scale)
            #         result = cv2.matchTemplate(resized, template,
            #                                    cv2.TM_CCOEFF_NORMED)
            #         (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            #         cv2.rectangle(resized, (int(maxLoc[0]), int(maxLoc[1])),
            #                       (int(maxLoc[0] + tW), int(maxLoc[1] + tH)),
            #                       (255, 255, 255), -1)
            #         (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            #         (endX, endY) = (int((maxLoc[0] + tW) * r),
            #                         int((maxLoc[1] + tH) * r))
            #         temp = [startX, startY, endX, endY]
            #         boundingBoxes.append(temp)
            #     self.image_visualize_white_block.append(resized)
#                     print(boundingBoxes)
        # t2 = time.time()
        # thefile = open('/home/mhbrt/Desktop/Wind/Multiscale/templates/Saved Time.txt', 'a')
        # thefile.write('TM = ' + str(t2-t1) + ' s')
        # thefile.close()

        # If detected on this scale size
        # if len(boundingBoxes) > 1:
        #     boundingBoxes = np.delete(boundingBoxes, 0, axis=0)
            # print("{} {}".format(len(pick), self.get_template_location()))
            # print(pick)
            # for (startX, startY, endX, endY) in pick:
            #     cv2.rectangle(image, (startX, startY),(endX, endY),
            #     (0, 255, 0), 2)
        boundingBoxes = np.array(boundingBoxes)
        pick = self.non_max_suppression_slow(boundingBoxes,
                                             self.get_nms_thresh())
        # pick = boundingBoxes

        if len(max_value_list) >= 1:
            max_local_value = max(max_value_list)
        # if len(max_value_list) == 1:
        #     max_local_value = max_value_list
        if max_value_list == []:
            pick = 0
            max_local_value = 0
        # cv2.imshow('IMAGE MATCH ()',image)
        return pick, max_local_value


class FontWrapper(Marker):
    def __init__(self, nms_thresh=0.3, visualize=False, **kwargs):
        # print("INIT!")
        self._Data = kwargs
        self.nms_thresh = nms_thresh
        self.visualize = visualize
        self.imagelist_visualize_white_blok = []
        # self.visualize = True

        self.image_location = self._Data["image_loc"]
        self.marker_location = self._Data["loc_list"]
        self.marker_thresh = self._Data["thresh_list"]
        self.image = self._Data["image"]

        for data in self._Data:
            if data == 'numstep':
                self.numstep = self._Data["numstep"]
            else:
                self.numstep = 0

        self.pocket = {}
        # super().__init__()
        colour_tanwin = [
            (255, 0, 0), (204, 0, 0), (255, 51, 51)
        ]
        colour_nun = [
            (0, 255, 0), (0, 204, 0), (51, 255, 51),
            # (102, 255, 102), (0, 153, 0)
        ]
        colour_mim = [
            (0, 255, 255), (0, 0, 204), (51, 51, 255),
            # (102, 102, 255), (0, 0, 153)
        ]
        t = 0
        m = 0
        n = 0
        self.temp_colour = self.get_marker_thresh().copy()
        for key in self.get_marker_thresh().keys():
            # print(key)
            x = key.split('_')
            if x[0] == 'tanwin':
                self.temp_colour[key] = colour_tanwin[t]
                if t+1 > len(colour_tanwin) - 1:
                    t = 0
                else:
                    t += 1
            if x[0] == 'nun':
                self.temp_colour[key] = colour_nun[n]
                if n+1 > len(colour_nun) - 1:
                    n = 0
                else:
                    n += 1
            if x[0] == 'mim':
                self.temp_colour[key] = colour_mim[m]
                if m+1 > len(colour_mim) - 1:
                    m = 0
                else:
                    m += 1

    def get_marker_thresh(self):
        return self.marker_thresh

    def get_marker_location(self):
        return self.marker_location

    # def get_image_location(self):
    #     return self.image_location

    # def get_original_image(self):
    #     original_image = cv2.imread(self.get_image_location())
    #     return original_image

    def get_object_result(self):
        return self.pocket

    def get_image(self):
        return self.image.copy()

    def get_object_name(self):
        return self.get_marker_location()[0].split('/')[2]

    def draw_bounding_box(self, img, coordinat, label, color, font_scale=1, font=cv2.FONT_HERSHEY_PLAIN):
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

    def display_marker_result(self, input_image, skip_marker):
        # rectangle_image = self.get_original_image()
        rectangle_image = input_image.copy()
        # found = False
        char_count = {}
        char_count['tanwin'] = 0
        char_count['nun'] = 0
        char_count['mim'] = 0
        for key in self.get_marker_thresh().keys():
            x = key.split('_')
            if x[0] == 'tanwin':
                label = '0'
                colour = (255, 51, 51)
            if x[0] == 'nun':
                label = '1'
                colour = (51, 255, 51)
            if x[0] == 'mim':
                label = '2'
                colour = (51, 51, 255)
            # split = key.split('_')
            # marker_name = split[0] + '_' + split[1]
            marker_name = key
            if marker_name in skip_marker:
                continue
            if isinstance(self.get_object_result()['box_' + key],
                          type(np.array([]))):
                for (startX, startY, endX, endY) in \
                        self.get_object_result()['box_' + key]:
                    # cv2.rectangle(rectangle_image, (startX, startY),
                    #               (endX, endY), self.temp_colour[key], 2)
                    char_count[x[0]] += 1
                    rectangle_image = self.draw_bounding_box(
                        rectangle_image,
                        [startX, startY, endX, endY],
                        label + '_' + str(char_count[x[0]]),
                        colour,
                        font_scale=1
                    )
                # print(self.pick_colour[x])
                # found = True
        # if found:
        #     print('<<<<<<<< View Result >>>>>>>>')
        #     cv2.imshow("Detected Image_" + self.get_object_name(),
        #                rectangle_image)
        # else:
        #     cv2.imshow("Original Image", rectangle_image)
        #     print('not found')
        # print('>')
        # cv2.waitKey(0)
        # cv2.destroyWindow("Detected Image_" + self.get_object_name())
        return rectangle_image

    def run(self, view=False, numstep=10, bw_method=0, skip_marker=''):
        # Tanwin
        # print(self.get_marker_thresh())
        print('run() Marker Font')
        # gray = cv2.cvtColor(self.get_original_image(), cv2.COLOR_BGR2GRAY)
        # if self.numstep == 0:
        #     numstep = numstep
        # else:
        #     numstep = self.numstep
        numstep = self.numstep
        print("numstep = ", numstep)
        pocketData = {}
        first = True
        # print(self.get_marker_thresh())
        t1 = time.time()
        for x in range(len(self.get_marker_thresh())):
            print(self.get_marker_location()[x])
            # print(len(template_thresh))
            # print(list(template_thresh.values())[x])
            marker_name = list(self.get_marker_thresh().keys())[x]
            # split = marker_name.split('_')
            # marker_name = split[0] + '_' + split[1]
            if marker_name in skip_marker:
                print('skip ', marker_name)
                continue
            super().__init__(image=self.get_image,
                             template_thresh=list(
                                 self.get_marker_thresh().values())[x],
                             template_loc=self.get_marker_location()[x],
                             nms_thresh=self.nms_thresh)
            (pocketData[x], pocketData[x+len(self.get_marker_thresh())]) \
                = super().match_template(visualize=self.visualize,
                                         numstep=numstep, bw_method=bw_method)
            if first:
                first = False
                self.imagelist_visualize_white_blok.append(
                    self.image_visualize_white_block)
        t2 = time.time()
        thefile = open('/home/mhbrt/Desktop/Wind/Multiscale/templates/Saved Time.txt', 'a')
        thefile.write('TM = ' + str(t2-t1) + ' s \n')
        thefile.close()

            # print(type(pocketData[x]))

        # Change the name by +id
        for x in range(len(self.get_marker_thresh())):
            temp = list(self.get_marker_thresh().keys())[x]
            # split = temp.split('_')
            # marker_name = split[0] + '_' + split[1]
            marker_name = temp
            if marker_name in skip_marker:
                continue
            # box = 'box_'+ temp
            self.get_object_result()[temp] = pocketData[
                x+len(self.get_marker_thresh())]
            self.get_object_result()['box_' + temp] = pocketData[x]

        if view:
            self.display_marker_result()

        # return pocket

    def find_connectivity(self, x, y, height, width, image):
        # count = 0
        x_y = []
        # Left
        if x - 1 > 0:
            if y + 1 < height:
                if image[y + 1, x - 1] == 0:
                    x_y.append((y + 1, x - 1))
                    # x_y.append((x - 1, y + 1))
                    # count += 1
                    # print('l1')
            if image[y, x - 1] == 0:
                x_y.append((y, x - 1))
                # x_y.append((x - 1, y))
                # count += 1
                # print('l2')
            if y - 1 > 0:
                if image[y - 1, x - 1] == 0:
                    x_y.append((y - 1, x - 1))
                    # x_y.append((x - 1, y - 1))
                    # count += 1
                    # print('l3')
        # Middle
        if y + 1 < height:
            if image[y + 1, x] == 0:
                x_y.append((y + 1, x))
                # x_y.append((x, y + 1))
                # count += 1
            # print('m1')
        x_y.append((y, x))
        # x_y.append((x, y))
        # count += 1
        # print('m2')
        if y - 1 > 0:
            if image[y - 1, x] == 0:
                x_y.append((y - 1, x))
                # x_y.append((x, y - 1))
                # count += 1
                # print('m3')
        # Right
        if x + 1 < width:
            if y + 1 < height:
                if image[y + 1, x + 1] == 0:
                    x_y.append((y + 1, x + 1))
                    # x_y.append((x + 1, y + 1))
                    # count += 1
                    # print('r1')
            if image[y, x + 1] == 0:
                x_y.append((y, x + 1))
                # x_y.append((x + 1, y))
                # count += 1
                # print('r2')
            if y - 1 > 0:
                if image[y - 1, x + 1] == 0:
                    x_y.append((y - 1, x + 1))
                    # x_y.append((x + 1, y - 1))
                    # count += 1
                    # print('r3')

        return x_y

    def eight_connectivity(self, image, seed, left=False, right=False,
                           y1_limit=0, y2_limit=10000, view=False):
        height, width = image.shape
#         print('image eight con width: ', width)
        image_process = image.copy()
        # For image flag
        image_process[:] = 255
        # oneline_height = oneline_baseline[1] - oneline_baseline[0]
        # if oneline_height <= 1:
        #     oneline_height_sorted = 3
        # else:
        #     oneline_height_sorted = oneline_height
        self.conn_pack = {}
        final_conn_pack = {}
        reg = 1
        # Doing eight conn on every pixel one by one
        for x in range(seed[0], seed[2]):
            w_count = -1
            for y in range(seed[1], seed[3]):
                if image_process[y, x] == 0:
                    continue
                min_x_region = 11111
                max_x_region = 0
                if image[y, x] == 0:
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
#                     print('wcount_o:', w_count)
#                     if w_count > width:
#                         continue
                    while(True):
                        # w_count += 1
                        # print('wcount:', w_count)
                        if w_count > width:  # stop if it's to large
                            # print('_more than width_')
                            break
                        if first:
                            length_ = length_
                            first = False
                        else:
                            length_ = l_after_sub

                        if init:
                            l_after_init = len(self.conn_pack[
                                'region_{:03d}'.format(reg)])
                            for k in range(length_, l_after_init):
                                w_count += 1
                                x_y_sub = self.find_connectivity(
                                    self.conn_pack[
                                        'region_{:03d}'.format(reg)][k][1],
                                    self.conn_pack[
                                        'region_{:03d}'.format(reg)][k][0],
                                    height, width, image
                                )
                                if len(x_y_sub) > 1 \
                                        and x_y_sub[0][0] > y1_limit \
                                        and x_y_sub[0][0] < y2_limit:
                                    sub = True
                                    for vl in x_y_sub:
                                        if vl[0] < y1_limit or vl[0] > y2_limit:
                                            continue
                                        if vl[1] > max_x_region:
                                            max_x_region = vl[1]
                                        if vl[1] < min_x_region:
                                            min_x_region = vl[1]
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
                                w_count += 1
                                x_y_sub = self.find_connectivity(
                                    self.conn_pack[
                                        'region_{:03d}'.format(reg)][k][1],
                                    self.conn_pack[
                                        'region_{:03d}'.format(reg)][k][0],
                                    height, width, image
                                )
                                if len(x_y_sub) > 1 \
                                        and x_y_sub[0][0] > y1_limit \
                                        and x_y_sub[0][0] < y2_limit:
                                    init = True
                                    for vl in x_y_sub:
                                        if vl[0] < y1_limit or vl[0] > y2_limit:
                                            continue
                                        if vl[1] > max_x_region:
                                            max_x_region = vl[1]
                                        if vl[1] < min_x_region:
                                            min_x_region = vl[1]
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
                    if left:
                        if max_x_region < seed[2]:
                            final_conn_pack['region_{:03d}'.format(reg)] \
                                = self.conn_pack['region_{:03d}'.format(reg)]
                    if right:
                        if min_x_region > seed[0]:
                            final_conn_pack['region_{:03d}'.format(reg)] \
                                = self.conn_pack['region_{:03d}'.format(reg)]
                    if not right and not left:
                        final_conn_pack['region_{:03d}'.format(reg)] \
                            = self.conn_pack['region_{:03d}'.format(reg)]
                    reg += 1
                    if view:
                        cv2.imshow('eight conn process', image_process)
#                         print(self.conn_pack)
                        cv2.waitKey(0)

        return final_conn_pack

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

    def detect_horizontal_line_up_down(self, h_projection):
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
            if h_projection[x] > 0:
                start_point.append(x)
                break
        for x in range(len(h_projection))[::-1]:
            if h_projection[x] > 0:
                start_point.append(x)
                break

#         if len(start_point) % 2 != 0:
#             if h_projection[len(h_projection) - 1] > 0 or len(start_point)==1:
#                 start_point.append(len(h_projection) - 1)
        if len(start_point) % 2 != 0:
            if h_projection[len(h_projection) - 1] > 0 or len(start_point) == 1:
                start_point.append(len(h_projection) - 1)
        # print('start point from mess:', start_point)
        if len(start_point) > 0:
            self.start_point_h = [start_point[0], start_point[1]]
        else:
            self.start_point_h = []

        # Even is begining of line and Odd is end of line
        if view and len(start_point) > 0:
            for x in range(0, 2):
                if x % 2 == 0:     # Start_point
                    cv2.line(image, (0, start_point[x]),
                             (width, start_point[x]), (100, 150, 0), 2)
                    # print(x)
                else:         # End_point
                    cv2.line(image, (0, start_point[x]),
                             (width, start_point[x]), (100, 150, 0), 2)
#             cv2.imshow('horizontal line', image)
#             cv2.waitKey(0)

        return image

    def base_line(self, one_line_image, view=True):
        # Got self.base_start, self.base_end, self.one_line_image
        self.base_end = 0
        self.base_start = 0
        h_projection = self.h_projection
        # print(h_projection)
        # original_image = one_line_image
        self.one_line_image = one_line_image
        h, self.width = one_line_image.shape
        diff = [0]
        for x in range(len(h_projection)):
            if x > 0:
                temp_diff = abs(int(h_projection[x]) - int(h_projection[x-1]))
                diff.append(temp_diff)

        # print('diff from baseline mess:', diff)
        temp = 0
        for x in range(len(diff)):
            if diff[x] > temp:
                temp = diff[x]
                self.base_end = x
        # Get the 2nd greatest to base_start
        temp = 0
        for x in range(len(diff)):
            if x == self.base_end:
                continue
            if diff[x] > temp:
                temp = diff[x]
                self.base_start = x

        if view:
            cv2.line(self.one_line_image, (0, self.base_start),
                     (self.width, self.base_start), (0, 255, 0), 2)
            cv2.line(self.one_line_image, (0, self.base_end),
                     (self.width, self.base_end), (0, 255, 0), 2)

    def detect_vertical_line(self, image, pixel_limit_ste, view=True):
        # Detect line vertical
        v_projection = self.v_projection
        # print(v_projection)
        original_image = image
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
        # Even is begining of line and Odd is end of line
#         if view:
#             for x in range(len(start_point)):
#                 if x % 2 == 0:
#                     cv2.line(original_image, (start_point[x], 0),
#                              (start_point[x], self.height), (0, 0, 0), 2)
#                 else:
#                     cv2.line(original_image, (start_point[x], 0),
#                              (start_point[x], self.height), (100, 100, 100), 2)

#             cv2.imshow('line', original_image)
#             print('>')
#             cv2.waitKey(0)
#             # print(self.start_point_v)

    def crop_image(self, input_image, h_point=False, v_point=False, view=True):
        if h_point:
            start_point = h_point
            original_image = input_image
#             print('>')
#             cv2.waitKey(0)
            bag_of_h_crop = {}
            for x in range(len(start_point)):
                if x + 2 > len(start_point):
                    # print('x')
                    continue
                if x % 2 == 0:
                    bag_of_h_crop[x] = original_image[
                        start_point[x]:start_point[x+1] + 1, :]
            # print(bag_of_h_crop)
#             if view:
#                 for image in bag_of_h_crop:
#                     cv2.imshow('bag_h'+str(image), bag_of_h_crop[image])
#                     print('>')
#                     cv2.waitKey(0)
#                     cv2.destroyWindow('bag_h'+str(image))
            self.bag_of_h_crop = bag_of_h_crop

        if v_point:
            start_point_v = v_point
            # if input_image!=False:
            original_image = input_image
            bag_of_v_crop = {}
            count = 0
            for image in bag_of_h_crop:
                for x in range(len(start_point_v)):
                    count += 1
                    if x % 2 == 0:
                        x1 = start_point[image]
                        x2 = start_point[image+1]
                        y1 = start_point_v[x]
                        y2 = start_point_v[x+1]
                        bag_of_v_crop[count] = original_image[x1:x2, y1:y2]
                    # print(x1,'_', x2,'_', y1,'_', y2)

#             if view:
#                 for image in bag_of_v_crop:
#                     cv2.imshow('Crop Result', bag_of_v_crop[image])
#                     print('>')
#                     cv2.waitKey(0)
                    # cv2.destroyAllWindows()
            self.bag_of_v_crop = bag_of_v_crop

#     def match_normal_eight_connectivity(self, image, oneline_baseline):
#         height, width = image.shape
#         image_process = image.copy()
#         # For image flag
#         image_process[:] = 255
#         oneline_height = oneline_baseline[1] - oneline_baseline[0]
#         if oneline_height <= 1:
#             oneline_height_sorted = 3
#         else:
#             oneline_height_sorted = oneline_height
#         self.conn_pack = {}
#         reg = 1
#         # Doing eight conn on every pixel one by one
#         for x in range(width):
#             for y in range(height):
#                 if image_process[y, x] == 0:
#                     continue
#                 if image[y, x] == 0:
#                     self.conn_pack['region_{:03d}'.format(reg)] = []
#                     x_y = self.find_connectivity(x, y, height, width, image)
#                     length_ = len(self.conn_pack['region_{:03d}'.format(reg)])
#                     for val in x_y:
#                         self.conn_pack['region_{:03d}'.format(reg)].append(val)
#                     # print(self.conn_pack['region_{:03d}'.format(reg)])
#                     # cv2.waitKey(0)
#                     first = True
#                     sub = False
#                     init = True
#                     while(True):
#                         if first:
#                             length_ = length_
#                             first = False
#                         else:
#                             length_ = l_after_sub

#                         if init:
#                             l_after_init = len(self.conn_pack[
#                                 'region_{:03d}'.format(reg)])
#                             for k in range(length_, l_after_init):
#                                 x_y_sub = self.find_connectivity(
#                                     self.conn_pack[
#                                         'region_{:03d}'.format(reg)][k][1],
#                                     self.conn_pack[
#                                         'region_{:03d}'.format(reg)][k][0],
#                                     height, width, image
#                                 )
#                                 if len(x_y_sub) > 1:
#                                     sub = True
#                                     for vl in x_y_sub:
#                                         if vl not in self.conn_pack[
#                                                 'region_{:03d}'.format(reg)]:
#                                             self.conn_pack[
#                                                 'region_{:03d}'.format(reg)
#                                             ].append(vl)
#                         init = False
#                         if sub:
#                             # print(self.conn_pack['region_{:03d}'.format(reg)])
#                             # cv2.waitKey(0)
#                             l_after_sub = len(self.conn_pack[
#                                 'region_{:03d}'.format(reg)])
#                             for k in range(l_after_init, l_after_sub):
#                                 x_y_sub = self.find_connectivity(
#                                     self.conn_pack[
#                                         'region_{:03d}'.format(reg)][k][1],
#                                     self.conn_pack[
#                                         'region_{:03d}'.format(reg)][k][0],
#                                     height, width, image
#                                 )
#                                 if len(x_y_sub) > 1:
#                                     init = True
#                                     for vl in x_y_sub:
#                                         if vl not in self.conn_pack[
#                                                 'region_{:03d}'.format(reg)]:
#                                             self.conn_pack[
#                                                 'region_{:03d}'.format(reg)
#                                             ].append(vl)
#                         sub = False

#                         if not sub and not init:
#                             break

#                     for val in self.conn_pack['region_{:03d}'.format(reg)]:
#                         image_process[val] = 0
#                     reg += 1
#                     # cv2.imshow('eight conn process', image_process)
#                     # print(self.conn_pack)
#                     # cv2.waitKey(0)

#         temp_marker = []
#         temp_delete = []
#         # Noise cancelation
#         k = 3
#         for key in self.conn_pack:
#             if len(self.conn_pack[key]) > k * oneline_height_sorted:
#                 temp_marker.append(key)
#         self.conn_pack_sorted = {}
#         for mark in temp_marker:
#             self.conn_pack_sorted[mark] = (self.conn_pack[mark])
#         # If region is not in the baseline then it's not a body image
#         for key in self.conn_pack_sorted:
#             found = False
#             for reg in self.conn_pack_sorted[key]:
#                 if found:
#                     break
#                 # Catch region in range 2x the oneline baseline height
#                 # for a better image body detection
#                 for base in range(oneline_baseline[0] - oneline_height_sorted,
#                                   oneline_baseline[1]+1):
#                     if reg[0] == base:
#                         found = True
#                         break
#             if found is False:
#                 temp_delete.append(key)
#         self.conn_pack_minus_body = {}
#         # Get body only and minus body region
#         for delt in temp_delete:
#             self.conn_pack_minus_body[delt] = self.conn_pack_sorted[delt]
#             del(self.conn_pack_sorted[delt])
#         # Paint body only region
#         self.image_body = image.copy()
#         self.image_body[:] = 255
#         for region in self.conn_pack_sorted:
#             value = self.conn_pack_sorted[region]  # imagebody region dict
#             for x in value:
#                 self.image_body[x] = 0

#         self.image_join = self.image_body.copy()
#         for region in self.conn_pack_minus_body:
#             value = self.conn_pack_minus_body[region]
#             for x in value:
#                 self.image_join[x] = 0

#         self.vertical_projection(self.image_body)
#         self.detect_vertical_line(
#             image=self.image_body.copy(),
#             pixel_limit_ste=0,
#             view=True
#         )
#         # print(self.start_point_v)
#         # Make sure every start point has an end
#         len_h = len(self.start_point_v)
#         if len_h % 2 != 0:
#             del(self.start_point_v[len_h - 1])
#         group_body_by_wall = {}
#         for x in range(len(self.start_point_v)):
#             if x % 2 == 0:
#                 wall = (self.start_point_v[x], self.start_point_v[x+1])
#                 group_body_by_wall[wall] = []
#                 for region in self.conn_pack_sorted:
#                     value = self.conn_pack_sorted[region]
#                     for y_x in value:
#                         # Grouping image body region by its wall (x value)
#                         if self.start_point_v[x] <= y_x[1] \
#                                 <= self.start_point_v[x+1]:
#                             group_body_by_wall[wall].append(region)
#                             break
#         # print(group_body_by_wall)
#         for region in group_body_by_wall.values():
#             max_length = 0
#             if len(region) > 1:
#                 for x in region:
#                     if len(self.conn_pack_sorted[x]) > max_length:
#                         max_length = len(self.conn_pack_sorted[x])
#                 for x in region:
#                     # Fixing hamzah dumping out from image body problem
#                     sorted_region_val = sorted(self.conn_pack_sorted[x])
#                     y_up_region = sorted_region_val[0][0]
#                     region_height = oneline_baseline[0] - y_up_region
#                     # Check how long the char from above the baseline
#                     if region_height > 3 * oneline_height_sorted:
#                         continue
#                     # If region is not 1/4 of the max length then move it
#                     # from image body to marker only
#                     # NB. why just not using the longest reg to sort? coz when
#                     # it does there's a case where two separate words
#                     # overlapping each other by a tiny margin (no white space)
#                     elif len(self.conn_pack_sorted[x]) < 1/4*max_length:
#                         self.conn_pack_minus_body[x] = self.conn_pack_sorted[x]
#                         del(self.conn_pack_sorted[x])

#         self.image_final_sorted = image.copy()
#         self.image_final_sorted[:] = 255
#         for region in self.conn_pack_sorted:
#             value = self.conn_pack_sorted[region]
#             for x in value:
#                 self.image_final_sorted[x] = 0
# #         cv2.imshow('image final sorted', self.image_final_sorted)
#         self.image_final_marker = image.copy()
#         self.image_final_marker[:] = 255
#         for region in self.conn_pack_minus_body:
#             value = self.conn_pack_minus_body[region]
#             for x in value:
#                 self.image_final_marker[x] = 0
# #         cv2.imshow('image final marker', self.image_final_marker)
# #         cv2.waitKey(0)

#     def grouping_marker(self):
#         img_body_v_proj = self.start_point_v
#         # Make sure every start point has an end
#         len_h = len(img_body_v_proj)
#         if len_h % 2 != 0:
#             del(img_body_v_proj[len_h - 1])
#         self.group_marker_by_wall = {}
#         for x_x in range(len(img_body_v_proj)):
#             if x_x % 2 == 0:
#                 wall = (img_body_v_proj[x_x], img_body_v_proj[x_x+1])
#                 self.group_marker_by_wall[wall] = []
#         for region in self.conn_pack_minus_body:
#             value = self.conn_pack_minus_body[region]
#             x_y_value = []
#             # Flip (y,x) to (x,y) and sort
#             for x in value:
#                 x_y_value.append(x[::-1])
#             x_y_value = sorted(x_y_value)
#             # print(x_y_value)
#             x_y_value_l = []
#             for val in range(round(len(x_y_value)/2)):
#                 x_y_value_l.append(x_y_value[0])
#                 del(x_y_value[0])
#             x_y_value_l = x_y_value_l[::-1]
#             # print(x_y_value_l)
#             # print('_________________')
#             # print(x_y_value)
#             x_y_value_from_mid = []
#             count = -1
#             for x_1 in x_y_value:
#                 count += 1
#                 x_y_value_from_mid.append(x_1)
#                 for x_2 in x_y_value_l:
#                     if count < len(x_y_value_l):
#                         x_y_value_from_mid.append(x_y_value_l[count])
#                     break

#             for x_x in range(len(img_body_v_proj)):
#                 if x_x % 2 == 0:
#                     wall = (img_body_v_proj[x_x], img_body_v_proj[x_x+1])
#                     for x in x_y_value_from_mid:
#                         if wall[0] <= x[0] <= wall[1]:
#                             self.group_marker_by_wall[wall].append(region)
#                             break
# #         print('Group marker by wall')
# #         print(self.group_marker_by_wall)

#     def modified_eight_connectivity(self, image, oneline_baseline):
#         height, width = image.shape
#         image_process = image.copy()
#         # For image flag
#         image_process[:] = 255
#         oneline_height = oneline_baseline[1] - oneline_baseline[0]
#         if oneline_height <= 1:
#             oneline_height_sorted = 3
#         else:
#             oneline_height_sorted = oneline_height
#         self.conn_pack = {}
#         reg = 1
#         # Doing eight conn on every pixel one by one
#         for x in range(width):
#             for y in range(height):
#                 if image_process[y, x] == 0:
#                     continue
#                 if image[y, x] == 0:
#                     self.conn_pack['region_{:03d}'.format(reg)] = []
#                     x_y = self.find_connectivity(x, y, height, width, image)
#                     length_ = len(self.conn_pack['region_{:03d}'.format(reg)])
#                     for val in x_y:
#                         self.conn_pack['region_{:03d}'.format(reg)].append(val)
#                     # print(self.conn_pack['region_{:03d}'.format(reg)])
#                     # cv2.waitKey(0)
#                     first = True
#                     sub = False
#                     init = True
#                     while(True):
#                         if first:
#                             length_ = length_
#                             first = False
#                         else:
#                             length_ = l_after_sub

#                         if init:
#                             l_after_init = len(self.conn_pack[
#                                 'region_{:03d}'.format(reg)])
#                             for k in range(length_, l_after_init):
#                                 x_y_sub = self.find_connectivity(
#                                     self.conn_pack[
#                                         'region_{:03d}'.format(reg)][k][1],
#                                     self.conn_pack[
#                                         'region_{:03d}'.format(reg)][k][0],
#                                     height, width, image
#                                 )
#                                 if len(x_y_sub) > 1:
#                                     sub = True
#                                     for vl in x_y_sub:
#                                         if vl not in self.conn_pack[
#                                                 'region_{:03d}'.format(reg)]:
#                                             self.conn_pack[
#                                                 'region_{:03d}'.format(reg)
#                                             ].append(vl)
#                         init = False
#                         if sub:
#                             # print(self.conn_pack['region_{:03d}'.format(reg)])
#                             # cv2.waitKey(0)
#                             l_after_sub = len(self.conn_pack[
#                                 'region_{:03d}'.format(reg)])
#                             for k in range(l_after_init, l_after_sub):
#                                 x_y_sub = self.find_connectivity(
#                                     self.conn_pack[
#                                         'region_{:03d}'.format(reg)][k][1],
#                                     self.conn_pack[
#                                         'region_{:03d}'.format(reg)][k][0],
#                                     height, width, image
#                                 )
#                                 if len(x_y_sub) > 1:
#                                     init = True
#                                     for vl in x_y_sub:
#                                         if vl not in self.conn_pack[
#                                                 'region_{:03d}'.format(reg)]:
#                                             self.conn_pack[
#                                                 'region_{:03d}'.format(reg)
#                                             ].append(vl)
#                         sub = False

#                         if not sub and not init:
#                             break

#                     for val in self.conn_pack['region_{:03d}'.format(reg)]:
#                         image_process[val] = 0
#                     reg += 1
#                     # cv2.imshow('eight conn process', image_process)
#                     # print(self.conn_pack)
#                     # cv2.waitKey(0)

#         temp_marker = []
#         temp_delete = []
#         # Noise cancelation
#         k = 3
#         for key in self.conn_pack:
#             if len(self.conn_pack[key]) > k * oneline_height_sorted:
#                 temp_marker.append(key)
#         self.conn_pack_sorted = {}
#         for mark in temp_marker:
#             self.conn_pack_sorted[mark] = (self.conn_pack[mark])
#         # If region is not in the baseline then it's not a body image
#         for key in self.conn_pack_sorted:
#             found = False
#             for reg in self.conn_pack_sorted[key]:
#                 if found:
#                     break
#                 # Catch region in range 2x the oneline baseline height
#                 # for a better image body detection
#                 for base in range(oneline_baseline[0],
#                                   oneline_baseline[1]+1):
#                     if reg[0] == base:
#                         found = True
#                         break
#             if found is False:
#                 temp_delete.append(key)
#         self.conn_pack_minus_body = {}
#         # Get body only and minus body region
#         for delt in temp_delete:
#             self.conn_pack_minus_body[delt] = self.conn_pack_sorted[delt]
#             del(self.conn_pack_sorted[delt])
#         # Paint body only region
#         self.image_body = image.copy()
#         self.image_body[:] = 255
#         for region in self.conn_pack_sorted:
#             value = self.conn_pack_sorted[region]  # imagebody region dict
#             for x in value:
#                 self.image_body[x] = 0

#         self.image_join = self.image_body.copy()
#         for region in self.conn_pack_minus_body:
#             value = self.conn_pack_minus_body[region]
#             for x in value:
#                 self.image_join[x] = 0

#         self.vertical_projection(self.image_body)
#         self.detect_vertical_line(
#             image=self.image_body.copy(),
#             pixel_limit_ste=0,
#             view=True
#         )
#         # print(self.start_point_v)
#         # Make sure every start point has an end
#         len_h = len(self.start_point_v)
#         if len_h % 2 != 0:
#             del(self.start_point_v[len_h - 1])
#         group_body_by_wall = {}
#         for x in range(len(self.start_point_v)):
#             if x % 2 == 0:
#                 wall = (self.start_point_v[x], self.start_point_v[x+1])
#                 group_body_by_wall[wall] = []
#                 for region in self.conn_pack_sorted:
#                     value = self.conn_pack_sorted[region]
#                     for y_x in value:
#                         # Grouping image body region by its wall (x value)
#                         if self.start_point_v[x] <= y_x[1] \
#                                 <= self.start_point_v[x+1]:
#                             group_body_by_wall[wall].append(region)
#                             break
#         # print(group_body_by_wall)
#         for region in group_body_by_wall.values():
#             max_length = 0
#             if len(region) > 1:
#                 for x in region:
#                     if len(self.conn_pack_sorted[x]) > max_length:
#                         max_length = len(self.conn_pack_sorted[x])
#                 for x in region:
#                     # Fixing hamzah dumping out from image body problem
#                     sorted_region_val = sorted(self.conn_pack_sorted[x])
#                     y_up_region = sorted_region_val[0][0]
#                     region_height = oneline_baseline[0] - y_up_region
#                     # Check how long the char from above the baseline
#                     if region_height > 3 * oneline_height_sorted:
#                         continue
#                     # If region is not 1/4 of the max length then move it
#                     # from image body to marker only
#                     # NB. why just not using the longest reg to sort? coz when
#                     # it does there's a case where two separate words
#                     # overlapping each other by a tiny margin (no white space)
#                     elif len(self.conn_pack_sorted[x]) < 1/4*max_length:
#                         self.conn_pack_minus_body[x] = self.conn_pack_sorted[x]
#                         del(self.conn_pack_sorted[x])

#         self.image_final_sorted = image.copy()
#         self.image_final_sorted[:] = 255
#         for region in self.conn_pack_sorted:
#             value = self.conn_pack_sorted[region]
#             for x in value:
#                 self.image_final_sorted[x] = 0
# #         cv2.imshow('image final sorted', self.image_final_sorted)
#         self.image_final_marker = image.copy()
#         self.image_final_marker[:] = 255
#         for region in self.conn_pack_minus_body:
#             value = self.conn_pack_minus_body[region]
#             for x in value:
#                 self.image_final_marker[x] = 0
# #         cv2.imshow('image final marker', self.image_final_marker)
# #         cv2.waitKey(0)

    def dot_checker(self, image_marker):
        # Dot detection
        next_step = False
        self.horizontal_projection(image_marker)
        # self.detect_horizontal_line(image_marker.copy(), 0, 0, False)
        start_point = self.detect_horizontal_line_up_down(self.h_projection)
        # if len(self.start_point_h) > 1:
        if len(start_point) > 1:
            # one_marker = image_marker[self.start_point_h[0]:
            #                           self.start_point_h[1], :]
            one_marker = image_marker[start_point[0]:
                                      start_point[1], :]
            self.vertical_projection(one_marker)
            self.detect_vertical_line(one_marker.copy(), 0, False)
            if len(self.start_point_v) > 1:
                x1 = self.start_point_v[0]
                x2 = self.start_point_v[1]
                one_marker = one_marker[:, x1:x2]
                next_step = True
            else:
                return False
        else:
            return False
        if next_step:
            height, width = one_marker.shape
            scale = 1.3
            write_canvas = False
            # Square, Portrait or Landscape image
            if width < scale * height:
                if height < scale * width:
                    print('_square_')
                    black = False
                    white = False
                    middle_hole = False
                    # Possibly sukun
                    for y in range(height):
                        if one_marker[y, round(width/2)] == 0:
                            black = True
                        if black and one_marker[y, round(width/2)] > 0:
                            white = True
                        if white and one_marker[y, round(width/2)] == 0:
                            middle_hole = True
                    if middle_hole:
                        print('_white hole in the middle_')
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
                                if one_marker[y, x] > 0:
                                    white_val += 1
                                if one_marker[y, x] == 0:
                                    black = True
                                if black and one_marker[y, x] > 0:
                                    white = True
                                if white and one_marker[y, x] == 0:
                                    white_hole = True
                                    break
                        if white_hole:
                            print('_there is a hole_')
                            write_canvas = False
                        else:
                            # Check on 1/4 till 3/4 region
                            touch_up = False
                            touch_down = False
                            for x in range(round(width/4)-1, 3*round(width/4)):
                                if one_marker[0, x] == 0:
                                    touch_up = True
                                if one_marker[height-1, x] == 0:
                                    touch_down = True
                                if touch_up and touch_down:
                                    break
                            # Check on after 1/5(mitigate noise) till 1/2
                            if touch_up and touch_down:
                                too_many_whites = False
                                for x in range(round(width/5), round(width/2)):
                                    white_val = 0
                                    for y in range(height):
                                        if one_marker[y, x] > 0:
                                            white_val += 1
                                    if white_val > round(height/1.5):
                                        too_many_whites = True
                                        break
                                if too_many_whites:
                                    print('_too many white value in 1/5 till 1/2_')
                                    write_canvas = False
                                else:
                                    print('_DOT CONFIRM_')
                                    write_canvas = True
    #                         else:
    #                             print('not touching')

                    if not write_canvas:
                        # Split image into two vertically and looking for bwb
                        # (Kaf Hamzah)
                        # bwb_up = False
                        bwb_down = False
                        bwb_count = 0
                        bwb_thresh = round(height/2)
                        addition = round(height/8)
                        up_limit = round(height/2)
                        down_limit = round(height/2)
                        for x in range(width):
                            if bwb_count > bwb_thresh:
                                break
                            black = False
                            white = False
                            for y in range(0, up_limit):
                                if one_marker[y, x] == 0:
                                    black = True
                                if black and one_marker[y, x] > 0:
                                    white = True
                                if white and one_marker[y, x] == 0:
                                    # bwb_up = True
                                    bwb_count += 1
                                    break
                        for x in range(width):
                            if bwb_count > bwb_thresh and bwb_down:
                                break
                            black = False
                            white = False
                            for y in range(down_limit, height):
                                if one_marker[y, x] == 0:
                                    black = True
                                if black and one_marker[y, x] > 0:
                                    white = True
                                if white and one_marker[y, x] == 0:
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
                                if one_marker[y, x] == 0:
                                    black = True
                                if black and one_marker[y, x] > 0:
                                    bw = True
                                if bw:
                                    bw_count += 1
                                    black = False
                                    bw = False
                            if bw_count > bw_max:
                                bw_max = bw_count
                        if bwb_count >= bwb_thresh and bwb_down and bw_max < 3:
                            print('_KAF HAMZAH CONFIRM_')
                            write_canvas = True
                        else:
                            print('_also not kaf hamzah_')
                            write_canvas = False
                else:
                    print('_portrait image_')
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
                            if one_marker[y, x] == 0:
                                black = True
                            if black and one_marker[y, x] > 0:
                                white = True
                            if white and one_marker[y, x] == 0:
                                bwb_up = True
                                break
                    bwb_down_count = 0
                    for x in range(width):
                        # if bwb_down:
                        #     break
                        black = False
                        white = False
                        for y in range(round(height/1.5), height):
                            if one_marker[y, x] == 0:
                                black = True
                            if black and one_marker[y, x] > 0:
                                white = True
                            if white and one_marker[y, x] == 0:
                                bwb_down = True
                                bwb_down_count += 1
                                break
                    if bwb_up and bwb_down and bwb_down_count > 1/5 * width:
                        print('_KAF HAMZAH CONFIRM_')
                        write_canvas = True
                    else:
                        write_canvas = False
            else:
                print('_landscape image_')
                black = False
                white = False
                wbw_confirm = False
                over_pattern = False
                # Possibly straight harakat or tasdid
                for y in range(height):
                    if one_marker[y, round(width/2)] > 0:
                        white = True
                    if white and one_marker[y, round(width/2)] == 0:
                        black = True
                    if black and one_marker[y, round(width/2)] > 0:
                        wbw_confirm = True
                    if wbw_confirm and one_marker[y, round(width/2)] == 0:
                        over_pattern = True
                        break
                if over_pattern:
                    print('_too many wbw + b_')
                    write_canvas = False
                elif wbw_confirm:
                    print('_mid is wbw_')
                    too_many_white_val = False
                    # cut in the middle up vertically wether the pixel all white
                    for x in range(round(width/5), round(width/3)):
                        white_val = 0
                        for y in range(0, height):
                            if one_marker[y, x] > 0:
                                white_val += 1
                        if white_val > round(height/1.9):
                            too_many_white_val = True
                            break
                    if too_many_white_val:
                        print('_too many white val in 1/5 till 1/3_')
                        write_canvas = False
                    else:
                        half_img = one_marker[:, 0:round(width/2)]
                        self.horizontal_projection(half_img)
                        self.detect_horizontal_line(
                            half_img.copy(), 0, 0, True)
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
                            if half_img[0, x] == 0:
                                touch_up = True
                            if half_img[half_height-1, x] == 0:
                                touch_down = True
                            if touch_up and touch_down:
                                break
                        if touch_up and touch_down:
                            print('_DOT CONFIRM_')
                            write_canvas = True
                        else:
                            print('_not touching_')
                            write_canvas = False
                else:
                    print('_middle is not wbw_')
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
                            if one_marker[y, x] == 0:
                                black = True
                            if black and one_marker[y, x] > 0:
                                white = True
                            if white and one_marker[y, x] == 0:
                                bwb_up = True
                                break
                    for x in range(width):
                        if bwb_down:
                            break
                        black = False
                        white = False
                        for y in range(round(height/2), height):
                            if one_marker[y, x] == 0:
                                black = True
                            if black and one_marker[y, x] > 0:
                                white = True
                            if white and one_marker[y, x] == 0:
                                bwb_down = True
                                break
                    if bwb_up and bwb_down:
                        print('_KAF HAMZAH CONFIRM_')
                        write_canvas = True
                    else:
                        write_canvas = False

            return write_canvas


# def font(imagePath, image):
#     # LPMQ_Font
#     # print("LPMQ")
#     loc_list_LPMQ = sorted(glob.glob('./marker/LPMQ/*.png'))
#     font_LPMQ = FontWrapper(thresh_list={'tanwin_1': 0.7,
#                                          'tanwin_2': 0.7,
#                                          'nun_isolated': 0.7,
#                                          'nun_begin_1': 0.7,
#                                          'nun_begin_2': 0.7,
#                                          'nun_middle': 0.7,
#                                          'nun_end': 0.6,
#                                          'mim_isolated': 0.7,
#                                          'mim_begin': 0.8,
#                                          'mim_middle': 0.7,
#                                          'mim_end_1': 0.7,
#                                          'mim_end_2': 0.7},
#                             loc_list=loc_list_LPMQ, image_loc=imagePath,
#                             image=image, visualize=False, nms_thresh=0.3,
#                             numstep=30)
#     # AlQalam_Font
#     # print("AlQalam")
#     loc_list_AlQalam = sorted(glob.glob('./marker/AlQalam/*.png'))
#     font_AlQalam = FontWrapper(thresh_list={'tanwin_1': 0.7,
#                                             'tanwin_2': 0.7,
#                                             'nun_isolated': 0.7,
#                                             'nun_begin': 0.7,
#                                             'nun_middle': 0.7,
#                                             'nun_end': 0.7,
#                                             'mim_isolated': 0.7,
#                                             'mim_begin': 0.7,
#                                             'mim_middle': 0.7,
#                                             'mim_end': 0.7},
#                                loc_list=loc_list_AlQalam, image_loc=imagePath,
#                                image=image, visualize=False, nms_thresh=0.3)
#     # meQuran_Font
#     # print("meQuran")
#     loc_list_meQuran = sorted(glob.glob('./marker/meQuran/*.png'))
#     font_meQuran = FontWrapper(thresh_list={'tanwin_1': 0.7,
#                                             'tanwin_2': 0.65,
#                                             'nun_isolated': 0.7,
#                                             'nun_begin_1': 0.7,
#                                             'nun_begin_2': 0.7,
#                                             'nun_middle': 0.7,
#                                             'nun_end': 0.7,
#                                             'mim_isolated': 0.7,
#                                             'mim_begin': 0.7,
#                                             'mim_middle': 0.7,
#                                             'mim_end_1': 0.7,
#                                             'mim_end_2': 0.68},
#                                loc_list=loc_list_meQuran, image_loc=imagePath,
#                                image=image, visualize=False, nms_thresh=0.3)
#     # PDMS_Font
#     # print("PDMS")
#     loc_list_PDMS = sorted(glob.glob('./marker/PDMS/*.png'))
#     font_PDMS = FontWrapper(thresh_list={'tanwin_1': 0.7,
#                                          'tanwin_2': 0.7,
#                                          'nun_isolated': 0.7,
#                                          'nun_begin': 0.65,
#                                          'nun_middle': 0.7,
#                                          'nun_end': 0.7,
#                                          'mim_isolated': 0.7,
#                                          'mim_begin': 0.7,
#                                          'mim_middle': 0.7,
#                                          'mim_end': 0.65},
#                             loc_list=loc_list_PDMS, image_loc=imagePath,
#                             image=image, visualize=False, nms_thresh=0.3)

#     # AlKareem_Font
#     loc_list_AlKareem = sorted(glob.glob('./marker/AlKareem/*.png'))
#     font_AlKareem = FontWrapper(thresh_list={'tanwin_1': 0.7,
#                                          'tanwin_2': 0.7,
#                                          'nun_isolated': 0.7,
#                                          'nun_begin': 0.65,
#                                          'nun_middle': 0.7,
#                                          'nun_end': 0.7,
#                                          'mim_isolated': 0.7,
#                                          'mim_begin': 0.7,
#                                          'mim_middle': 0.7,
#                                          'mim_end': 0.65,
#                                          'mim_end': 0.65},
#                             loc_list=loc_list_AlKareem, image_loc=imagePath,
#                             image=image, visualize=False, nms_thresh=0.3)

#     # KFGQPC_Font
#     loc_list_KFGQPC = sorted(glob.glob('./marker/KFGQPC/*.png'))
#     font_KFGQPC = FontWrapper(thresh_list={'tanwin_1': 0.7,
#                                          'tanwin_2': 0.7,
#                                          'nun_isolated': 0.7,
#                                          'nun_begin': 0.65,
#                                          'nun_begin': 0.65,
#                                          'nun_middle': 0.7,
#                                          'nun_end': 0.7,
#                                          'mim_isolated': 0.7,
#                                          'mim_begin': 0.7,
#                                          'mim_middle': 0.7,
#                                          'mim_end': 0.65},
#                             loc_list=loc_list_KFGQPC, image_loc=imagePath,
#                             image=image, visualize=False, nms_thresh=0.3)

#     # Amiri_Font
#     loc_list_Amiri = sorted(glob.glob('./marker/amiri/*.png'))
#     font_Amiri = FontWrapper(thresh_list={'tanwin_1': 0.7,
#                                          'tanwin_2': 0.7,
#                                          'nun_isolated': 0.7,
#                                          'nun_begin': 0.65,
#                                          'nun_begin': 0.65,
#                                          'nun_begin': 0.65,
#                                          'nun_middle': 0.7,
#                                          'nun_end': 0.7,
#                                          'mim_isolated': 0.7,
#                                          'mim_begin': 0.7,
#                                          'mim_middle': 0.7,
#                                          'mim_end': 0.8,
#                                          'mim_end': 0.8},
#                             loc_list=loc_list_Amiri, image_loc=imagePath,
#                             image=image, visualize=False, nms_thresh=0.3)

#     # Norehidayat_Font
#     loc_list_Norehidayat = sorted(glob.glob('./marker/norehidayat/*.png'))
#     font_Norehidayat = FontWrapper(thresh_list={'tanwin_1': 0.7,
#                                          'tanwin_2': 0.7,
#                                          'nun_isolated': 0.7,
#                                          'nun_begin': 0.65,
#                                          'nun_begin': 0.65,
#                                          'nun_end': 0.7,
#                                          'mim_isolated': 0.7,
#                                          'mim_begin': 0.7,
#                                          'mim_middle': 0.7,
#                                          'mim_end': 0.65},
#                             loc_list=loc_list_Norehidayat, image_loc=imagePath,
#                             image=image, visualize=False, nms_thresh=0.3)

#     # Norehira_Font
#     loc_list_Norehira = sorted(glob.glob('./marker/norehira/*.png'))
#     font_Norehira = FontWrapper(thresh_list={'tanwin_1': 0.9,
#                                          'tanwin_2': 0.7,
#                                          'nun_isolated': 0.7,
#                                          'nun_begin': 0.65,
#                                          'nun_middle': 0.7,
#                                          'nun_end': 0.65,
#                                          'mim_isolated': 0.7,
#                                          'mim_begin': 0.7,
#                                          'mim_middle': 0.7,
#                                          'mim_end': 0.65,
#                                          'mim_end': 0.65},
#                             loc_list=loc_list_Norehira, image_loc=imagePath,
#                             image=image, visualize=False, nms_thresh=0.3)

#     # Norehuda_Font
#     loc_list_Norehuda = sorted(glob.glob('./marker/norehuda/*.png'))
#     font_Norehuda = FontWrapper(thresh_list={'tanwin_1': 0.9,
#                                          'tanwin_2': 0.7,
#                                          'nun_isolated': 0.7,
#                                          'nun_begin': 0.65,
#                                          'nun_middle': 0.7,
#                                          'nun_end': 0.7,
#                                          'mim_isolated': 0.7,
#                                          'mim_begin': 0.7,
#                                          'mim_middle': 0.7,
#                                          'mim_end': 0.65},
#                             loc_list=loc_list_Norehuda, image_loc=imagePath,
#                             image=image, visualize=False, nms_thresh=0.3)

#     # list_object_font = [font_LPMQ, font_AlQalam, font_meQuran, font_PDMS]
#     list_object_font = [font_LPMQ, font_AlQalam, font_meQuran, font_PDMS,
#                         font_AlKareem, font_KFGQPC, font_Amiri, font_Norehidayat,
#                         font_Norehira, font_Norehuda]

#     return list_object_font

# def font(imagePath, image, setting, markerPath):
#     # LPMQ_Font
#     # print("LPMQ")
#     loc_list_LPMQ = sorted(glob.glob(markerPath + '/LPMQ/*.png'))
#     font_LPMQ = FontWrapper(thresh_list={'tanwin_1': float(setting['LPMQ'][0][0]),
#                                          'tanwin_2': float(setting['LPMQ'][0][1]),
#                                          'nun_isolated': float(setting['LPMQ'][0][2]),
#                                          'nun_begin_1': float(setting['LPMQ'][0][3]),
#                                          'nun_begin_2': float(setting['LPMQ'][0][4]),
#                                          'nun_middle': float(setting['LPMQ'][0][5]),
#                                          'nun_end': float(setting['LPMQ'][0][6]),
#                                          'mim_isolated': float(setting['LPMQ'][0][7]),
#                                          'mim_begin': float(setting['LPMQ'][0][8]),
#                                          'mim_middle': float(setting['LPMQ'][0][9]),
#                                          'mim_end_1': float(setting['LPMQ'][0][10]),
#                                          'mim_end_2': float(setting['LPMQ'][0][11]),},
#                             loc_list=loc_list_LPMQ, image_loc=imagePath,
#                             image=image, visualize=True, nms_thresh=0.3,
#                             numstep=int(setting['LPMQ'][1]))
#     # AlQalam_Font
#     # print("AlQalam")
#     loc_list_AlQalam = sorted(glob.glob(markerPath + '/AlQalam/*.png'))
#     font_AlQalam = FontWrapper(thresh_list={'tanwin_1': float(setting['AlQalam'][0][0]),
#                                             'tanwin_2': float(setting['AlQalam'][0][1]),
#                                             'nun_isolated': float(setting['AlQalam'][0][2]),
#                                             'nun_begin': float(setting['AlQalam'][0][3]),
#                                             'nun_middle': float(setting['AlQalam'][0][4]),
#                                             'nun_end': float(setting['AlQalam'][0][5]),
#                                             'mim_isolated': float(setting['AlQalam'][0][6]),
#                                             'mim_begin': float(setting['AlQalam'][0][7]),
#                                             'mim_middle': float(setting['AlQalam'][0][8]),
#                                             'mim_end': float(setting['AlQalam'][0][9])},
#                                loc_list=loc_list_AlQalam, image_loc=imagePath,
#                                image=image, visualize=True, nms_thresh=0.3,
#                                numstep=int(setting['AlQalam'][1]))
#     # meQuran_Font
#     # print("meQuran")
#     loc_list_meQuran = sorted(glob.glob(markerPath + '/meQuran/*.png'))
#     font_meQuran = FontWrapper(thresh_list={'tanwin_1': float(setting['meQuran'][0][0]),
#                                             'tanwin_2': float(setting['meQuran'][0][1]),
#                                             'nun_isolated': float(setting['meQuran'][0][2]),
#                                             'nun_begin_1': float(setting['meQuran'][0][3]),
#                                             'nun_begin_2': float(setting['meQuran'][0][4]),
#                                             'nun_middle': float(setting['meQuran'][0][5]),
#                                             'nun_end': float(setting['meQuran'][0][6]),
#                                             'mim_isolated': float(setting['meQuran'][0][7]),
#                                             'mim_begin': float(setting['meQuran'][0][8]),
#                                             'mim_middle': float(setting['meQuran'][0][9]),
#                                             'mim_end_1': float(setting['meQuran'][0][10]),
#                                             'mim_end_2': float(setting['meQuran'][0][11])},
#                                loc_list=loc_list_meQuran, image_loc=imagePath,
#                                image=image, visualize=True, nms_thresh=0.3,
#                                numstep=int(setting['meQuran'][1]))
#     # PDMS_Font
#     # print("PDMS")
#     loc_list_PDMS = sorted(glob.glob(markerPath + '/PDMS/*.png'))
#     font_PDMS = FontWrapper(thresh_list={'tanwin_1': float(setting['PDMS'][0][0]),
#                                          'tanwin_2': float(setting['PDMS'][0][1]),
#                                          'nun_isolated': float(setting['PDMS'][0][2]),
#                                          'nun_begin': float(setting['PDMS'][0][3]),
#                                          'nun_middle': float(setting['PDMS'][0][4]),
#                                          'nun_end': float(setting['PDMS'][0][5]),
#                                          'mim_isolated': float(setting['PDMS'][0][6]),
#                                          'mim_begin': float(setting['PDMS'][0][7]),
#                                          'mim_middle': float(setting['PDMS'][0][8]),
#                                          'mim_end': float(setting['PDMS'][0][9])},
#                             loc_list=loc_list_PDMS, image_loc=imagePath,
#                             image=image, visualize=True, nms_thresh=0.3,
#                             numstep=int(setting['PDMS'][1]))

#     # AlKareem_Font
#     loc_list_AlKareem = sorted(glob.glob(markerPath + '/AlKareem/*.png'))
#     font_AlKareem = FontWrapper(thresh_list={'tanwin_1': float(setting['AlKareem'][0][0]),
#                                          'tanwin_2': float(setting['AlKareem'][0][1]),
#                                          'nun_isolated': float(setting['AlKareem'][0][2]),
#                                          'nun_begin': float(setting['AlKareem'][0][3]),
#                                          'nun_middle': float(setting['AlKareem'][0][4]),
#                                          'nun_end': float(setting['AlKareem'][0][5]),
#                                          'mim_isolated': float(setting['AlKareem'][0][6]),
#                                          'mim_begin': float(setting['AlKareem'][0][7]),
#                                          'mim_middle': float(setting['AlKareem'][0][8]),
#                                          'mim_end_1': float(setting['AlKareem'][0][9]),
#                                          'mim_end_2': float(setting['AlKareem'][0][10])},
#                             loc_list=loc_list_AlKareem, image_loc=imagePath,
#                             image=image, visualize=True, nms_thresh=0.3,
#                             numstep=int(setting['AlKareem'][1]))

#     # KFGQPC_Font
#     loc_list_KFGQPC = sorted(glob.glob(markerPath + '/KFGQPC/*.png'))
#     font_KFGQPC = FontWrapper(thresh_list={'tanwin_1': float(setting['KFGQPC'][0][0]),
#                                          'tanwin_2': float(setting['KFGQPC'][0][1]),
#                                          'nun_isolated': float(setting['KFGQPC'][0][2]),
#                                          'nun_begin_1': float(setting['KFGQPC'][0][3]),
#                                          'nun_begin_2': float(setting['KFGQPC'][0][4]),
#                                          'nun_middle': float(setting['KFGQPC'][0][5]),
#                                          'nun_end': float(setting['KFGQPC'][0][6]),
#                                          'mim_isolated': float(setting['KFGQPC'][0][7]),
#                                          'mim_begin': float(setting['KFGQPC'][0][8]),
#                                          'mim_middle': float(setting['KFGQPC'][0][9]),
#                                          'mim_end': float(setting['KFGQPC'][0][10])},
#                             loc_list=loc_list_KFGQPC, image_loc=imagePath,
#                             image=image, visualize=True, nms_thresh=0.3,
#                             numstep=int(setting['KFGQPC'][1]))

#     # Amiri_Font
#     loc_list_Amiri = sorted(glob.glob(markerPath + '/amiri/*.png'))
#     font_Amiri = FontWrapper(thresh_list={'tanwin_1': float(setting['amiri'][0][0]),
#                                          'tanwin_2': float(setting['amiri'][0][1]),
#                                          'nun_isolated': float(setting['amiri'][0][2]),
#                                          'nun_begin_1': float(setting['amiri'][0][3]),
#                                          'nun_begin_2': float(setting['amiri'][0][4]),
#                                          'nun_begin_3': float(setting['amiri'][0][5]),
#                                          'nun_middle': float(setting['amiri'][0][6]),
#                                          'nun_end': float(setting['amiri'][0][7]),
#                                          'mim_isolated': float(setting['amiri'][0][8]),
#                                          'mim_begin': float(setting['amiri'][0][9]),
#                                          'mim_middle': float(setting['amiri'][0][10]),
#                                          'mim_end_1': float(setting['amiri'][0][11]),
#                                          'mim_end_2': float(setting['amiri'][0][12])},
#                             loc_list=loc_list_Amiri, image_loc=imagePath,
#                             image=image, visualize=True, nms_thresh=0.3,
#                             numstep=int(setting['amiri'][1]))

#     # Norehidayat_Font
#     loc_list_Norehidayat = sorted(glob.glob(markerPath + '/norehidayat/*.png'))
#     font_Norehidayat = FontWrapper(thresh_list={'tanwin_1': float(setting['norehidayat'][0][0]),
#                                          'tanwin_2': float(setting['norehidayat'][0][1]),
#                                          'nun_isolated': float(setting['norehidayat'][0][2]),
#                                          'nun_begin_1': float(setting['norehidayat'][0][3]),
#                                          'nun_begin_2': float(setting['norehidayat'][0][4]),
#                                          'nun_end': float(setting['norehidayat'][0][5]),
#                                          'mim_isolated': float(setting['norehidayat'][0][6]),
#                                          'mim_begin': float(setting['norehidayat'][0][7]),
#                                          'mim_middle': float(setting['norehidayat'][0][8]),
#                                          'mim_end': float(setting['norehidayat'][0][9])},
#                             loc_list=loc_list_Norehidayat, image_loc=imagePath,
#                             image=image, visualize=True, nms_thresh=0.3,
#                             numstep=int(setting['norehidayat'][1]))

#     # Norehira_Font
#     loc_list_Norehira = sorted(glob.glob(markerPath + '/norehira/*.png'))
#     font_Norehira = FontWrapper(thresh_list={'tanwin_1': float(setting['norehira'][0][0]),
#                                          'tanwin_2': float(setting['norehira'][0][1]),
#                                          'nun_isolated': float(setting['norehira'][0][2]),
#                                          'nun_begin': float(setting['norehira'][0][3]),
#                                          'nun_middle': float(setting['norehira'][0][4]),
#                                          'nun_end': float(setting['norehira'][0][5]),
#                                          'mim_isolated': float(setting['norehira'][0][6]),
#                                          'mim_begin': float(setting['norehira'][0][7]),
#                                          'mim_middle': float(setting['norehira'][0][8]),
#                                          'mim_end_1': float(setting['norehira'][0][9]),
#                                          'mim_end_2': float(setting['norehira'][0][10]),},
#                             loc_list=loc_list_Norehira, image_loc=imagePath,
#                             image=image, visualize=True, nms_thresh=0.3,
#                             numstep=int(setting['norehira'][1]))

#     # Norehuda_Font
#     loc_list_Norehuda = sorted(glob.glob(markerPath + '/norehuda/*.png'))
#     font_Norehuda = FontWrapper(thresh_list={'tanwin_1': float(setting['norehuda'][0][0]),
#                                          'tanwin_2': float(setting['norehuda'][0][1]),
#                                          'nun_isolated': float(setting['norehuda'][0][2]),
#                                          'nun_begin': float(setting['norehuda'][0][3]),
#                                          'nun_middle': float(setting['norehuda'][0][4]),
#                                          'nun_end': float(setting['norehuda'][0][5]),
#                                          'mim_isolated': float(setting['norehuda'][0][6]),
#                                          'mim_begin': float(setting['norehuda'][0][7]),
#                                          'mim_middle': float(setting['norehuda'][0][8]),
#                                          'mim_end': float(setting['norehuda'][0][9])},
#                             loc_list=loc_list_Norehuda, image_loc=imagePath,
#                             image=image, visualize=True, nms_thresh=0.3,
#                             numstep=int(setting['norehuda'][1]))

#     # list_object_font = [font_LPMQ, font_AlQalam, font_meQuran, font_PDMS]
#     # list_object_font = [font_LPMQ, font_AlQalam, font_meQuran, font_PDMS,
#     #                     font_AlKareem, font_KFGQPC, font_Amiri, font_Norehidayat,
#     #                     font_Norehira, font_Norehuda]
#     list_object_font = [font_AlKareem, font_AlQalam, font_KFGQPC, font_LPMQ,
#                         font_PDMS, font_Amiri, font_meQuran, font_Norehidayat,
#                         font_Norehira, font_Norehuda]
#     font_name = ['AlKareem', 'AlQalam', 'KFGQPC', 'LPMQ', 'PDMS',
#                 'amiri', 'meQuran', 'norehidayat', 'norehira', 'norehuda']
#     temp_path = [loc_list_AlKareem, loc_list_AlQalam, loc_list_KFGQPC,
#                  loc_list_LPMQ, loc_list_PDMS, loc_list_Amiri,loc_list_meQuran,
#                  loc_list_Norehidayat, loc_list_Norehira, loc_list_Norehuda]
#     loc_path = {}
#     for x in range(len(font_name)):
#         loc_path[font_name[x]] = temp_path[x]


#     return list_object_font, loc_path

def font(imagePath, image, setting, markerPath):
    # LPMQ_Font
    # print("LPMQ")
    loc_list_LPMQ = sorted(glob.glob(markerPath + '/LPMQ/*.png'))
    font_LPMQ = FontWrapper(thresh_list={'tanwin_1': float(setting['LPMQ'][0][0]),
                                         'tanwin_2': float(setting['LPMQ'][0][1]),
                                         'nun_isolated': float(setting['LPMQ'][0][2]),
                                         'nun_begin_1': float(setting['LPMQ'][0][3]),
                                         'nun_begin_2': float(setting['LPMQ'][0][4]),
                                         'nun_begin_3': float(setting['LPMQ'][0][3]),
                                         'nun_begin_4': float(setting['LPMQ'][0][3]),
                                         'nun_begin_5': float(setting['LPMQ'][0][4]),
                                         'nun_middle_1': float(setting['LPMQ'][0][5]),
                                         'nun_middle_2': float(setting['LPMQ'][0][5]),
                                         'nun_middle_3': float(setting['LPMQ'][0][5]),
                                         'nun_middle_4': float(setting['LPMQ'][0][5]),
                                         'nun_end': float(setting['LPMQ'][0][6]),
                                         'mim_isolated': float(setting['LPMQ'][0][11]),
                                         'mim_begin': float(setting['LPMQ'][0][11]),
                                         'mim_middle_1': float(setting['LPMQ'][0][11]),
                                         'mim_middle_2': float(setting['LPMQ'][0][11]),
                                         'mim_end_1': float(setting['LPMQ'][0][11]),
                                         'mim_end_2': float(setting['LPMQ'][0][11]),
                                         'mim_end_3': float(setting['LPMQ'][0][11])},
                            loc_list=loc_list_LPMQ, image_loc=imagePath,
                            image=image, visualize=True, nms_thresh=0.3,
                            numstep=int(setting['LPMQ'][1]))
    # AlQalam_Font
    # print("AlQalam")
    loc_list_AlQalam = sorted(glob.glob(markerPath + '/AlQalam/*.png'))
    font_AlQalam = FontWrapper(thresh_list={'tanwin_1': float(setting['AlQalam'][0][0]),
                                            'tanwin_2': float(setting['AlQalam'][0][1]),
                                            'nun_isolated': float(setting['AlQalam'][0][2]),
                                            'nun_begin_1': float(setting['AlQalam'][0][3]),
                                            'nun_begin_2': float(setting['AlQalam'][0][3]),
                                            'nun_middle_1': float(setting['AlQalam'][0][4]),
                                            'nun_middle_2': float(setting['AlQalam'][0][4]),
                                            'nun_middle_3': float(setting['AlQalam'][0][4]),
                                            'nun_middle_4': float(setting['AlQalam'][0][4]),
                                            'nun_middle_5': float(setting['AlQalam'][0][4]),
                                            'nun_middle_6': float(setting['AlQalam'][0][4]),
                                            'nun_end': float(setting['AlQalam'][0][5]),
                                            'nun_end_2': float(setting['AlQalam'][0][5]),
                                            'mim_isolated': float(setting['AlQalam'][0][6]),
                                            'mim_begin': float(setting['AlQalam'][0][7]),
                                            'mim_middle': float(setting['AlQalam'][0][8]),
                                            'mim_end': float(setting['AlQalam'][0][9])},
                               loc_list=loc_list_AlQalam, image_loc=imagePath,
                               image=image, visualize=True, nms_thresh=0.3,
                               numstep=int(setting['AlQalam'][1]))
    # meQuran_Font
    # print("meQuran")
    loc_list_meQuran = sorted(glob.glob(markerPath + '/meQuran/*.png'))
    font_meQuran = FontWrapper(thresh_list={'tanwin_1': float(setting['meQuran'][0][0]),
                                            'tanwin_2': float(setting['meQuran'][0][1]),
                                            'nun_isolated': float(setting['meQuran'][0][2]),
                                            'nun_begin_1': float(setting['meQuran'][0][3]),
                                            'nun_begin_2': float(setting['meQuran'][0][4]),
                                            'nun_middle_1': float(setting['meQuran'][0][5]),
                                            'nun_middle_2': float(setting['meQuran'][0][5]),
                                            'nun_end': float(setting['meQuran'][0][6]),
                                            'mim_isolated': float(setting['meQuran'][0][7]),
                                            'mim_begin': float(setting['meQuran'][0][8]),
                                            'mim_middle': float(setting['meQuran'][0][9]),
                                            'mim_end_1': float(setting['meQuran'][0][10]),
                                            'mim_end_2': float(setting['meQuran'][0][11]),
                                            'mim_end_3': float(setting['meQuran'][0][11]),
                                            'mim_end_4': float(setting['meQuran'][0][11])},
                               loc_list=loc_list_meQuran, image_loc=imagePath,
                               image=image, visualize=True, nms_thresh=0.3,
                               numstep=int(setting['meQuran'][1]))
    # PDMS_Font
    # print("PDMS")
    loc_list_PDMS = sorted(glob.glob(markerPath + '/PDMS/*.png'))
    font_PDMS = FontWrapper(thresh_list={'tanwin_1': float(setting['PDMS'][0][0]),
                                         'tanwin_2': float(setting['PDMS'][0][1]),
                                         'nun_isolated': float(setting['PDMS'][0][2]),
                                         'nun_begin': float(setting['PDMS'][0][3]),
                                         'nun_middle_1': float(setting['PDMS'][0][4]),
                                         'nun_middle_2': float(setting['PDMS'][0][4]),
                                         'nun_middle_3': float(setting['PDMS'][0][4]),
                                         'nun_end': float(setting['PDMS'][0][5]),
                                         'mim_isolated': float(setting['PDMS'][0][6]),
                                         'mim_begin': float(setting['PDMS'][0][7]),
                                         'mim_middle_1': float(setting['PDMS'][0][8]),
                                         'mim_middle_2': float(setting['PDMS'][0][8]),
                                         'mim_end': float(setting['PDMS'][0][9])},
                            loc_list=loc_list_PDMS, image_loc=imagePath,
                            image=image, visualize=True, nms_thresh=0.3,
                            numstep=int(setting['PDMS'][1]))

    # AlKareem_Font
    loc_list_AlKareem = sorted(glob.glob(markerPath + '/AlKareem/*.png'))
    font_AlKareem = FontWrapper(thresh_list={'tanwin_1': float(setting['AlKareem'][0][0]),
                                             'tanwin_2': float(setting['AlKareem'][0][1]),
                                             'nun_isolated': float(setting['AlKareem'][0][2]),
                                             'nun_begin': float(setting['AlKareem'][0][3]),
                                             'nun_middle': float(setting['AlKareem'][0][4]),
                                             'nun_end': float(setting['AlKareem'][0][5]),
                                             'mim_isolated': float(setting['AlKareem'][0][6]),
                                             'mim_begin': float(setting['AlKareem'][0][7]),
                                             'mim_middle_1': float(setting['AlKareem'][0][8]),
                                             'mim_middle_2': float(setting['AlKareem'][0][8]),
                                             'mim_end_1': float(setting['AlKareem'][0][9]),
                                             'mim_end_2': float(setting['AlKareem'][0][10])},
                                loc_list=loc_list_AlKareem, image_loc=imagePath,
                                image=image, visualize=True, nms_thresh=0.3,
                                numstep=int(setting['AlKareem'][1]))

    # FAKE AlKareem_Font
    loc_list_AlKareem = sorted(glob.glob(markerPath + '/AlKareem/*.png'))
    font_AlKareem = FontWrapper(thresh_list={'tanwin_1': float(setting['AlKareem'][0][0]),
                                             'tanwin_2': float(setting['AlKareem'][0][1]),
                                             'nun_isolated': float(setting['AlKareem'][0][2]),
                                             'nun_begin': float(setting['AlKareem'][0][3]),
                                             'nun_middle': float(setting['AlKareem'][0][4]),
                                             'nun_end': float(setting['AlKareem'][0][5]),
                                             'mim_isolated': float(setting['AlKareem'][0][6]),
                                             'mim_begin': float(setting['AlKareem'][0][7]),
                                             'mim_middle_1': float(setting['AlKareem'][0][8]),
                                             'mim_middle_2': float(setting['AlKareem'][0][8]),
                                             'mim_end_1': float(setting['AlKareem'][0][9]),},
                                loc_list=loc_list_AlKareem, image_loc=imagePath,
                                image=image, visualize=True, nms_thresh=0.3,
                                numstep=int(setting['AlKareem'][1]))

    # KFGQPC_Font
    loc_list_KFGQPC = sorted(glob.glob(markerPath + '/KFGQPC/*.png'))
    font_KFGQPC = FontWrapper(thresh_list={'tanwin_1': float(setting['KFGQPC'][0][0]),
                                           'tanwin_2': float(setting['KFGQPC'][0][1]),
                                           'tanwin_3': float(setting['KFGQPC'][0][1]),
                                           'tanwin_4': float(setting['KFGQPC'][0][1]),
                                           'tanwin_5': float(setting['KFGQPC'][0][1]),
                                           'nun_isolated': float(setting['KFGQPC'][0][2]),
                                           'nun_begin_1': float(setting['KFGQPC'][0][3]),
                                           'nun_begin_2': float(setting['KFGQPC'][0][4]),
                                           'nun_begin_3': float(setting['KFGQPC'][0][4]),
                                           'nun_begin_4': float(setting['KFGQPC'][0][4]),
                                           'nun_middle_1': float(setting['KFGQPC'][0][5]),
                                           'nun_middle_2': float(setting['KFGQPC'][0][5]),
                                           'nun_middle_3': float(setting['KFGQPC'][0][5]),
                                           'nun_end_1': float(setting['KFGQPC'][0][6]),
                                           'nun_end_2': float(setting['KFGQPC'][0][6]),
                                           'mim_isolated': float(setting['KFGQPC'][0][7]),
                                           'mim_begin': float(setting['KFGQPC'][0][8]),
                                           'mim_middle': float(setting['KFGQPC'][0][9]),
                                           'mim_end': float(setting['KFGQPC'][0][10])},
                              loc_list=loc_list_KFGQPC, image_loc=imagePath,
                              image=image, visualize=True, nms_thresh=0.3,
                              numstep=int(setting['KFGQPC'][1]))

    # Amiri_Font
    loc_list_Amiri = sorted(glob.glob(markerPath + '/amiri/*.png'))
    font_Amiri = FontWrapper(thresh_list={'tanwin_1': float(setting['amiri'][0][0]),
                                          'tanwin_2': float(setting['amiri'][0][1]),
                                          'nun_isolated': float(setting['amiri'][0][2]),
                                          'nun_begin_1': float(setting['amiri'][0][3]),
                                          'nun_begin_2': float(setting['amiri'][0][4]),
                                          'nun_begin_3': float(setting['amiri'][0][3]),
                                          'nun_begin_4': float(setting['amiri'][0][4]),
                                          'nun_begin_5': float(setting['amiri'][0][3]),
                                          'nun_begin_6': float(setting['amiri'][0][4]),
                                          'nun_begin_7': float(setting['amiri'][0][3]),
                                          'nun_middle_1': float(setting['amiri'][0][6]),
                                          'nun_middle_2': float(setting['amiri'][0][6]),
                                          'nun_middle_3': float(setting['amiri'][0][6]),
                                          'nun_middle_4': float(setting['amiri'][0][6]),
                                          'nun_middle_5': float(setting['amiri'][0][6]),
                                          'nun_end': float(setting['amiri'][0][7]),
                                          'mim_isolated': float(setting['amiri'][0][8]),
                                          'mim_begin_1': float(setting['amiri'][0][9]),
                                          'mim_begin_2': float(setting['amiri'][0][9]),
                                          'mim_middle_1': float(setting['amiri'][0][10]),
                                          'mim_middle_2': float(setting['amiri'][0][10]),
                                          'mim_end_1': float(setting['amiri'][0][11]),
                                          'mim_end_2': float(setting['amiri'][0][12]),
                                          'mim_end_3': float(setting['amiri'][0][12]),
                                          'mim_end_4': float(setting['amiri'][0][12]),
                                          'mim_end_5': float(setting['amiri'][0][12]),
                                          'mim_end_6': float(setting['amiri'][0][12])},
                             loc_list=loc_list_Amiri, image_loc=imagePath,
                             image=image, visualize=True, nms_thresh=0.3,
                             numstep=int(setting['amiri'][1]))

    # Norehidayat_Font
    loc_list_Norehidayat = sorted(glob.glob(markerPath + '/norehidayat/*.png'))
    font_Norehidayat = FontWrapper(thresh_list={'tanwin_1': float(setting['norehidayat'][0][0]),
                                                'tanwin_2': float(setting['norehidayat'][0][1]),
                                                'nun_isolated': float(setting['norehidayat'][0][2]),
                                                'nun_begin_1': float(setting['norehidayat'][0][3]),
                                                'nun_begin_2': float(setting['norehidayat'][0][4]),
                                                'nun_begin_3': float(setting['norehidayat'][0][4]),
                                                'nun_middle_1': float(setting['norehidayat'][0][4]),
                                                'nun_middle_2': float(setting['norehidayat'][0][4]),
                                                'nun_end_1': float(setting['norehidayat'][0][5]),
                                                'nun_end_2': float(setting['norehidayat'][0][5]),
                                                'mim_isolated': float(setting['norehidayat'][0][6]),
                                                'mim_begin': float(setting['norehidayat'][0][7]),
                                                'mim_middle': float(setting['norehidayat'][0][8]),
                                                'mim_end': float(setting['norehidayat'][0][9])},
                                   loc_list=loc_list_Norehidayat, image_loc=imagePath,
                                   image=image, visualize=True, nms_thresh=0.3,
                                   numstep=int(setting['norehidayat'][1]))

    # Norehira_Font
    loc_list_Norehira = sorted(glob.glob(markerPath + '/norehira/*.png'))
    font_Norehira = FontWrapper(thresh_list={'tanwin_1': float(setting['norehira'][0][0]),
                                             'tanwin_2': float(setting['norehira'][0][1]),
                                             'tanwin_3': float(setting['norehira'][0][1]),
                                             'nun_isolated': float(setting['norehira'][0][2]),
                                             'nun_begin_1': float(setting['norehira'][0][3]),
                                             'nun_begin_2': float(setting['norehira'][0][3]),
                                             'nun_begin_3': float(setting['norehira'][0][3]),
                                             'nun_middle_1': float(setting['norehira'][0][4]),
                                             'nun_middle_2': float(setting['norehira'][0][4]),
                                             'nun_middle_3': float(setting['norehira'][0][4]),
                                             'nun_end_1': float(setting['norehira'][0][5]),
                                             'nun_end_2': float(setting['norehira'][0][5]),
                                             'mim_isolated': float(setting['norehira'][0][6]),
                                             'mim_begin': float(setting['norehira'][0][7]),
                                             'mim_middle_1': float(setting['norehira'][0][8]),
                                             'mim_middle_2': float(setting['norehira'][0][8]),
                                             'mim_end_1': float(setting['norehira'][0][9]),
                                             'mim_end_2': float(setting['norehira'][0][10]),
                                             'mim_end_3': float(setting['norehira'][0][9]) },
                                loc_list=loc_list_Norehira, image_loc=imagePath,
                                image=image, visualize=True, nms_thresh=0.3,
                                numstep=int(setting['norehira'][1]))

    # Norehuda_Font
    loc_list_Norehuda = sorted(glob.glob(markerPath + '/norehuda/*.png'))
    font_Norehuda = FontWrapper(thresh_list={'tanwin_1': float(setting['norehuda'][0][0]),
                                             'tanwin_2': float(setting['norehuda'][0][1]),
                                             'nun_isolated': float(setting['norehuda'][0][2]),
                                             'nun_begin_1': float(setting['norehuda'][0][3]),
                                             'nun_begin_2': float(setting['norehuda'][0][3]),
                                             'nun_middle_1': float(setting['norehuda'][0][4]),
                                             'nun_middle_2': float(setting['norehuda'][0][4]),
                                             'nun_middle_3': float(setting['norehuda'][0][4]),
                                             'nun_end': float(setting['norehuda'][0][5]),
                                             'mim_isolated': float(setting['norehuda'][0][6]),
                                             'mim_begin': float(setting['norehuda'][0][7]),
                                             'mim_middle': float(setting['norehuda'][0][8]),
                                             'mim_end': float(setting['norehuda'][0][9])},
                                loc_list=loc_list_Norehuda, image_loc=imagePath,
                                image=image, visualize=True, nms_thresh=0.3,
                                numstep=int(setting['norehuda'][1]))

    # list_object_font = [font_LPMQ, font_AlQalam, font_meQuran, font_PDMS]
    # list_object_font = [font_LPMQ, font_AlQalam, font_meQuran, font_PDMS,
    #                     font_AlKareem, font_KFGQPC, font_Amiri, font_Norehidayat,
    #                     font_Norehira, font_Norehuda]
    list_object_font = [font_AlKareem, font_AlQalam, font_KFGQPC, font_LPMQ,
                        font_PDMS, font_Amiri, font_meQuran, font_Norehidayat,
                        font_Norehira, font_Norehuda]
    font_name = ['AlKareem', 'AlQalam', 'KFGQPC', 'LPMQ', 'PDMS',
                 'amiri', 'meQuran', 'norehidayat', 'norehira', 'norehuda']
    temp_path = [loc_list_AlKareem, loc_list_AlQalam, loc_list_KFGQPC,
                 loc_list_LPMQ, loc_list_PDMS, loc_list_Amiri, loc_list_meQuran,
                 loc_list_Norehidayat, loc_list_Norehira, loc_list_Norehuda]
    loc_path = {}
    for x in range(len(font_name)):
        loc_path[font_name[x]] = temp_path[x]

    return list_object_font, loc_path
