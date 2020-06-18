# import argparse
import matplotlib.pyplot as plt
import numpy as np
import imutils
import glob
import cv2
import copy

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

    def match_template(self, visualize=False, numstep=100):
        # Get template
        # print('.')
        template = cv2.imread(self.get_template_location())
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # Canny edge
        # template = cv2.Canny(template, 50, 200)
        # Otsu threshold
        # ret_temp, template = cv2.threshold(template, 0, 255,
        #                                    cv2.THRESH_BINARY
        #                                    + cv2.THRESH_OTSU)
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
        template = cv2.adaptiveThreshold(template, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY,
                                         11, 2)
        # cv2.imshow("Template", template)
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
        # print(numstep)
        # Loop over scaled image (start stop numstep) from the back
        for scale in np.linspace(0.2, 2.0, numstep)[::-1]:
            resized = imutils.resize(image, width=int(image.shape[1] * scale))
            r = image.shape[1] / float(resized.shape[1])
            # If resized image smaller than template, then break the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # Preprocessing resized image and then apply template matching
            # Cannyedge
            # edged = cv2.Canny(resized, 50, 200)
            # Otsu threshold
            # ret_img, resized = cv2.threshold(resized, 0, 255,
            #                                  cv2.THRESH_BINARY
            #                                  + cv2.THRESH_OTSU)
            # Simple threshold
            # ret_img, resized = cv2.threshold(resized, 127, 255,
            #                                  cv2.THRESH_BINARY)
            # Adaptive threshold value is the mean of neighbourhood area
            # resized = cv2.adaptiveThreshold(resized, 255,
            #                                 cv2.ADAPTIVE_THRESH_MEAN_C,
            #                                 cv2.THRESH_BINARY, 11, 2)
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
                print(self.get_template_location())
                print(maxVal)
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                              (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualizing", clone)
                cv2.waitKey(0)

            if maxVal > self.get_template_thresh():
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r),
                                int((maxLoc[1] + tH) * r))
                temp = [startX, startY, endX, endY]
                # boundingBoxes = np.append(boundingBoxes, [temp], axis=0)
                # max_value_list = np.append(max_value_list, [maxVal], axis=0)
                boundingBoxes.append(temp)
                max_value_list.append(maxVal)
                # print("Max val = {} location {}".format(maxVal, temp))

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

        if len(max_value_list) > 1:
            max_local_value = max(max_value_list)
        if len(max_value_list) == 1:
            max_local_value = max_value_list
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
            (255, 0, 255), (0, 0, 255), (128, 0, 128),
            (0, 0, 128)
        ]
        colour_nun = [
            (255, 0, 0), (128, 0, 0), (255, 99, 71),
            (220, 20, 60), (139, 0, 0)
        ]
        colour_mim = [
            (154, 205, 50), (107, 142, 35), (85, 107, 47),
            (0, 128, 0), (34, 139, 34)
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
        return self.image

    def get_object_name(self):
        return self.get_marker_location()[0].split('/')[2]

    def display_marker_result(self, input_image):
        # rectangle_image = self.get_original_image()
        rectangle_image = input_image.copy()
        found = False
        for key in self.get_marker_thresh().keys():
            if isinstance(self.get_object_result()['box_' + key],
                          type(np.array([]))):
                for (startX, startY, endX, endY) in \
                        self.get_object_result()['box_' + key]:
                    cv2.rectangle(rectangle_image, (startX, startY),
                                  (endX, endY), self.temp_colour[key], 2)
                # print(self.pick_colour[x])
                found = True
#         if found:
#             print('<<<<<<<< View Result >>>>>>>>')
#             cv2.imshow("Detected Image_" + self.get_object_name(),
#                        rectangle_image)
#         else:
#             cv2.imshow("Original Image", rectangle_image)
#             print('not found')
#         print('>')
#         cv2.waitKey(0)
        # cv2.destroyWindow("Detected Image_" + self.get_object_name())

    def run(self, view=False, numstep=100):
        # Tanwin
        # print(self.get_marker_thresh())
        print('run() Marker Font')

        # gray = cv2.cvtColor(self.get_original_image(), cv2.COLOR_BGR2GRAY)
        if self.numstep == 0:
            numstep = numstep
        else:
            numstep = self.numstep
        print("numstep = ", numstep)
        pocketData = {}
        for x in range(len(self.get_marker_thresh())):
            # print(len(template_thresh))
            # print(list(template_thresh.values())[x])
            super().__init__(image=self.get_image,
                             template_thresh=list(
                                 self.get_marker_thresh().values())[x],
                             template_loc=self.get_marker_location()[x],
                             nms_thresh=self.nms_thresh)
            (pocketData[x], pocketData[x+len(self.get_marker_thresh())]) \
                = super().match_template(visualize=self.visualize,
                                         numstep=numstep)
            # print(type(pocketData[x]))

        for x in range(len(self.get_marker_thresh())):
            temp = list(self.get_marker_thresh().keys())[x]
            # box = 'box_'+ temp
            self.get_object_result()[temp] = pocketData[
                x+len(self.get_marker_thresh())]
            self.get_object_result()['box_' + temp] = pocketData[x]

        if view:
            self.display_marker_result()

        # return pocket


class ImageProcessing():
    def __init__(self, **kwargs):
        # print('init')
        self._Data = kwargs
        self.original_image = self._Data['original_image']
        if len(self.original_image.shape) == 3:
            self.height, self.width, _ = self.original_image.shape
        else:
            self.height, self.width = self.original_image.shape
        # Instance variable list
        # self.v_projection = 0
        # self.h_projection = 0
        # self.start_point_h = []
        # self.start_point_v = []
        # self.base_start = 0
        # self.base_end = 0
        # self.bag_of_h_crop
        # self.bag_of_v_crop

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

        # Even is begining of line and Odd is end of line
        if view:
            for x in range(len(start_point)):
                if x % 2 == 0:     # Start_point
                    cv2.line(image, (0, start_point[x]),
                             (width, start_point[x]), (0, 0, 255), 2)
                    # print(x)
                else:         # End_point
                    cv2.line(image, (0, start_point[x]),
                             (width, start_point[x]), (100, 100, 255), 2)
#             cv2.imshow('horizontal line', image)
#             cv2.waitKey(0)
        return image

    def base_line(self, one_line_image, view=True):
        # Got self.base_start, self.base_end, self.one_line_image
        h_projection = self.h_projection
        # print(h_projection)
        # original_image = one_line_image
        self.one_line_image = one_line_image
        diff = [0]
        for x in range(len(h_projection)):
            if x > 0:
                temp_diff = abs(int(h_projection[x]) - int(h_projection[x-1]))
                diff.append(temp_diff)

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

    def eight_connectivity(self, image, oneline_baseline):
        height, width = image.shape
        image_process = image.copy()
        # For image flag
        image_process[:] = 255
        oneline_height = oneline_baseline[1] - oneline_baseline[0]
        if oneline_height <= 1:
            oneline_height_sorted = 3
        else:
            oneline_height_sorted = oneline_height
        self.conn_pack = {}
        reg = 1
        # Doing eight conn on every pixel one by one
        for x in range(width):
            for y in range(height):
                if image_process[y, x] == 0:
                    continue
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

        temp_marker = []
        temp_delete = []
        # Noise cancelation
        k = 3
        for key in self.conn_pack:
            if len(self.conn_pack[key]) > k * oneline_height_sorted:
                temp_marker.append(key)
        self.conn_pack_sorted = {}
        for mark in temp_marker:
            self.conn_pack_sorted[mark] = (self.conn_pack[mark])
        # If region is not in the baseline then it's not a body image
        for key in self.conn_pack_sorted:
            found = False
            for reg in self.conn_pack_sorted[key]:
                if found:
                    break
                # Catch region in range 2x the oneline baseline height
                # for a better image body detection
                for base in range(oneline_baseline[0] - oneline_height_sorted,
                                  oneline_baseline[1]+1):
                    if reg[0] == base:
                        found = True
                        break
            if found is False:
                temp_delete.append(key)
        self.conn_pack_minus_body = {}
        # Get body only and minus body region
        for delt in temp_delete:
            self.conn_pack_minus_body[delt] = self.conn_pack_sorted[delt]
            del(self.conn_pack_sorted[delt])
        # Paint body only region
        self.image_body = image.copy()
        self.image_body[:] = 255
        for region in self.conn_pack_sorted:
            value = self.conn_pack_sorted[region]  # imagebody region dict
            for x in value:
                self.image_body[x] = 0

        self.image_join = self.image_body.copy()
        for region in self.conn_pack_minus_body:
            value = self.conn_pack_minus_body[region]
            for x in value:
                self.image_join[x] = 0

        self.vertical_projection(self.image_body)
        self.detect_vertical_line(
            image=self.image_body.copy(),
            pixel_limit_ste=0,
            view=True
        )
        # print(self.start_point_v)
        # Make sure every start point has an end
        len_h = len(self.start_point_v)
        if len_h % 2 != 0:
            del(self.start_point_v[len_h - 1])
        group_body_by_wall = {}
        for x in range(len(self.start_point_v)):
            if x % 2 == 0:
                wall = (self.start_point_v[x], self.start_point_v[x+1])
                group_body_by_wall[wall] = []
                for region in self.conn_pack_sorted:
                    value = self.conn_pack_sorted[region]
                    for y_x in value:
                        # Grouping image body region by its wall (x value)
                        if self.start_point_v[x] <= y_x[1] \
                                <= self.start_point_v[x+1]:
                            group_body_by_wall[wall].append(region)
                            break
        # print(group_body_by_wall)
        for region in group_body_by_wall.values():
            max_length = 0
            if len(region) > 1:
                for x in region:
                    if len(self.conn_pack_sorted[x]) > max_length:
                        max_length = len(self.conn_pack_sorted[x])
                for x in region:
                    # Fixing hamzah dumping out from image body problem
                    sorted_region_val = sorted(self.conn_pack_sorted[x])
                    y_up_region = sorted_region_val[0][0]
                    region_height = oneline_baseline[0] - y_up_region
                    # Check how long the char from above the baseline
                    if region_height > 3 * oneline_height_sorted:
                        continue
                    # If region is not 1/4 of the max length then move it
                    # from image body to marker only
                    # NB. why just not using the longest reg to sort? coz when
                    # it does there's a case where two separate words
                    # overlapping each other by a tiny margin (no white space)
                    elif len(self.conn_pack_sorted[x]) < 1/4*max_length:
                        self.conn_pack_minus_body[x] = self.conn_pack_sorted[x]
                        del(self.conn_pack_sorted[x])

        self.image_final_sorted = image.copy()
        self.image_final_sorted[:] = 255
        for region in self.conn_pack_sorted:
            value = self.conn_pack_sorted[region]
            for x in value:
                self.image_final_sorted[x] = 0
#         cv2.imshow('image final sorted', self.image_final_sorted)
        self.image_final_marker = image.copy()
        self.image_final_marker[:] = 255
        for region in self.conn_pack_minus_body:
            value = self.conn_pack_minus_body[region]
            for x in value:
                self.image_final_marker[x] = 0
#         cv2.imshow('image final marker', self.image_final_marker)
        cv2.waitKey(0)

    def grouping_marker(self):
        img_body_v_proj = self.start_point_v
        # Make sure every start point has an end
        len_h = len(img_body_v_proj)
        if len_h % 2 != 0:
            del(img_body_v_proj[len_h - 1])
        self.group_marker_by_wall = {}
        for x_x in range(len(img_body_v_proj)):
            if x_x % 2 == 0:
                wall = (img_body_v_proj[x_x], img_body_v_proj[x_x+1])
                self.group_marker_by_wall[wall] = []
        for region in self.conn_pack_minus_body:
            value = self.conn_pack_minus_body[region]
            x_y_value = []
            # Flip (y,x) to (x,y) and sort
            for x in value:
                x_y_value.append(x[::-1])
            x_y_value = sorted(x_y_value)
            # print(x_y_value)
            x_y_value_l = []
            for val in range(round(len(x_y_value)/2)):
                x_y_value_l.append(x_y_value[0])
                del(x_y_value[0])
            x_y_value_l = x_y_value_l[::-1]
            # print(x_y_value_l)
            # print('_________________')
            # print(x_y_value)
            x_y_value_from_mid = []
            count = -1
            for x_1 in x_y_value:
                count += 1
                x_y_value_from_mid.append(x_1)
                for x_2 in x_y_value_l:
                    if count < len(x_y_value_l):
                        x_y_value_from_mid.append(x_y_value_l[count])
                    break

            for x_x in range(len(img_body_v_proj)):
                if x_x % 2 == 0:
                    wall = (img_body_v_proj[x_x], img_body_v_proj[x_x+1])
                    for x in x_y_value_from_mid:
                        if wall[0] <= x[0] <= wall[1]:
                            self.group_marker_by_wall[wall].append(region)
                            break
#         print('Group marker by wall')
#         print(self.group_marker_by_wall)


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
#                 print('_square_')
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
#                     print('_white hole in the middle_')
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
#                         print('_there is a hole_')
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
#                                 print('_too many white value in 1/5 till 1/2_')
                                write_canvas = False
                            else:
#                                 print('_DOT CONFIRM_')
                                write_canvas = True
#                         else:
#                             print('not touching')

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
#                         print('_KAF HAMZAH CONFIRM_')
                        write_canvas = True
                    else:
#                         print('_also not kaf hamzah_')
                        write_canvas = False
            else:
#                 print('_portrait image_')
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
#                     print('_KAF HAMZAH CONFIRM_')
                    write_canvas = True
                else:
                    write_canvas = False
        else:
#             print('_landscape image_')
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
#                 print('_too many wbw + b_')
                write_canvas = False
            elif wbw_confirm:
#                 print('_mid is wbw_')
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
#                     print('_too many white val in 1/5 till 1/3_')
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
                        if half_img[0, x] == 0:
                            touch_up = True
                        if half_img[half_height-1, x] == 0:
                            touch_down = True
                        if touch_up and touch_down:
                            break
                    if touch_up and touch_down:
#                         print('_DOT CONFIRM_')
                        write_canvas = True
                    else:
#                         print('_not touching_')
                        write_canvas = False
            else:
#                 print('_middle is not wbw_')
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
#                     print('_KAF HAMZAH CONFIRM_')
                    write_canvas = True
                else:
                    write_canvas = False

        return write_canvas

    def find_final_processed_char(self, wall, oneline_baseline):
        # wall parameter is (x1=true wall, x2=x1_final_char/true wall)
        # depends on wether it's final char inside or beside
        wall_list = list(self.group_marker_by_wall.keys())
        for key in wall_list:
            if key[0] == wall[0]:
                wall_origin = key
                break
        marker_pos = {}
        for region in self.group_marker_by_wall[wall_origin]:
            value = self.conn_pack_minus_body[region]
            x_y_val = []
            # Flip (y,x) to (x,y) and sort
            for x in value:
                x_y_val.append(x[::-1])
            x_y_val = sorted(x_y_val)

            # Modify marker pos if it's not in wall_origin range
            if x_y_val[0][0] < wall_origin[0]:
                x2 = x_y_val[0][1]
                del(x_y_val[0])
                x_y_val.insert(0, (wall_origin[0], x2))
            if x_y_val[len(x_y_val)-1][0] > wall_origin[1]:
                x2 = x_y_val[len(x_y_val)-1][1]
                del(x_y_val[len(x_y_val)-1])
                x_y_val.insert(len(x_y_val), (wall_origin[1], x2))
            marker_pos[region] = (x_y_val[0][0], x_y_val[len(x_y_val)-1][0])

        # Run when x2(wall[1])==x1_final_char(inside)
        temp_marker_pos = marker_pos.copy()
        for region in temp_marker_pos:
            x_x = marker_pos[region]
            mid_x = x_x[0] + round((x_x[1]-x_x[0])/2)
            if mid_x > wall[1]:
                del(marker_pos[region])

        # Sort marker_pos by x so it can suit the next process
        pos = sorted(marker_pos)
        marker_pos = {}
        for reg in pos:
            marker_pos[reg] = temp_marker_pos[reg]
        key_marker_pos = list(marker_pos.keys())
        count = 1
        final_group_marker = {}
        final_group_dot = {}
        final_group_marker['char_{:02d}'.format(count)] = []
        final_group_dot['char_{:02d}'.format(count)] = []
        # Grouping marker and dot separately
#         print('Marker POS_____')
#         print(marker_pos)
#         cv2.waitKey(0)

        for x in range(len(marker_pos))[::-1]:
            canvas = self.image_final_marker.copy()
            canvas[:] = 255
            if x > 0:
                region = key_marker_pos[x]
                region_next = key_marker_pos[x-1]
                x2_next = marker_pos[region_next][1]
                x1_next = marker_pos[region_next][0]
                x2_now = marker_pos[region][1]
                x1_now = marker_pos[region][0]
                dist = x1_now - x1_next
                length_next = x2_next - x1_next

                for val in self.conn_pack_minus_body[region]:
                    canvas[val] = 0
#                 cv2.imshow('b4 dot detection', canvas)
#                 print('region__')
#                 print(region)
#                 cv2.waitKey(0)
                # pixel_count = len(self.conn_pack_minus_body[region])
                dot = self.dot_checker(canvas)
                # print('DOT______')
                # print(str(dot))
                # cv2.waitKey(0)
                if dot:
                    final_group_dot['char_{:02d}'.format(count)].append(region)

                # Append next_marker in one char if overlapped
                if (x1_now <= x1_next <= x2_now
                    or x1_next <= x2_now <= x2_next)\
                        or (x1_now <= x2_next <= x2_now
                            and dist < 2/3 * length_next):
                    if region not in final_group_marker[
                            'char_{:02d}'.format(count)]:
                        final_group_marker[
                            'char_{:02d}'.format(count)].append(region)
                    if region_next not in final_group_marker[
                            'char_{:02d}'.format(count)]:
                        final_group_marker['char_{:02d}'.format(count)].append(
                            region_next)
                # If not overlapped then append and create new char
                else:
                    if region not in final_group_marker[
                            'char_{:02d}'.format(count)]:
                        final_group_marker[
                            'char_{:02d}'.format(count)].append(region)
                    count += 1
                    final_group_marker['char_{:02d}'.format(count)] = []
                    final_group_dot['char_{:02d}'.format(count)] = []

            if x == 0:
                region = key_marker_pos[x]
                for val in self.conn_pack_minus_body[region]:
                    canvas[val] = 0
                # pixel_count = len(self.conn_pack_minus_body[region])
                dot = self.dot_checker(canvas)
                if dot:
                    final_group_dot['char_{:02d}'.format(count)].append(region)
                if region not in final_group_marker[
                        'char_{:02d}'.format(count)]:
                    final_group_marker[
                        'char_{:02d}'.format(count)].append(region)

        # check dot if theres is two dot that parallel and the whitespace is
        # not more than 1/3 dot size then append that to char as one char if
        # it's separated and that 2 appended char only get one dot
        '''dot_single_count = []
        for char in final_group_dot:
            if len(final_group_dot[char]) == 1:
                dot_single_count.append(char)

        if len(dot_single_count) > 1:
            for x in range(len(dot_single_count)):
                if x == len(dot_single_count) - 1:
                    break
                char_now = dot_single_count[x]
                char_next = dot_single_count[x+1]
                region_now = final_group_dot[char_now][0]
                region_next = final_group_dot[char_next][0]
                pos_now = marker_pos[region_now]
                pos_next = marker_pos[region_next]
                dist_now = pos_now[1] - pos_now[0]
                between_pos = pos_now[0] - pos_next[1]

                if between_pos < 1/2.4 * dist_now:
#                     print('_appending char group by its dot_')
                    for x in range(len(final_group_marker[char_next])):
                        final_group_marker[char_now].append(
                            final_group_marker[char_next][x]
                        )
                    final_group_dot[char_now].append(
                        final_group_dot[char_next][0]
                    )
                    del(final_group_marker[char_next])
                    del(final_group_dot[char_next])'''
        temp_marker_pos = marker_pos.copy()
        pos = sorted(marker_pos)
        marker_pos = {}
        for reg in pos:
            marker_pos[reg] = temp_marker_pos[reg]
        key_marker_pos = list(marker_pos.keys())
#         print('group marker')
#         print(final_group_marker)
#         print('group dot')
#         print(final_group_dot)
        self.imagelist_perchar_marker = []
        for char in final_group_marker:
            perchar_img = self.image_final_sorted.copy()
            perchar_img[:] = 255
            for region in final_group_marker[char]:
                value = self.conn_pack_minus_body[region]
                for x in value:
                    perchar_img[x] = 0
            perchar_img = perchar_img[:, wall[0]:wall[1]]
            self.imagelist_perchar_marker.append(perchar_img)

        # Paint dot on selected wall
        for char in final_group_dot:
            for region in final_group_dot[char]:
                value = self.conn_pack_minus_body[region]
#                 print(region)
#                 print(value)
#                 print(wall)
                for x in value:
                    # Move the wall border if dot region is larger
                    if x[1] < wall[0]:
                        wall = (x[1]-1, wall[1])
                    if x[1] > wall[1]:
                        wall = (wall[0], x[1]+1)
                    self.image_final_sorted[x] = 0

        # Crop final candidate
        oneline_height = oneline_baseline[1] - oneline_baseline[0]
        if oneline_height <= 1:
            oneline_height_sorted = 3
        else:
            oneline_height_sorted = oneline_height
        char_list = list(final_group_marker.keys())
        # If marker not found then it's not a char !!!
        if len(final_group_marker[char_list[0]]) == 0:  # No char at all
            print('__ It is not a character__')
            return 'continue', 'continue'

        if len(final_group_marker) == 1:  # There's only one char
            final_segmented_char_candidate = self.image_final_sorted[
                :, wall[0]:wall[1]
            ]
            pass_x1 = wall[0]
            # self.horizontal_projection(final_segmented_char_candidate)
            # self.base_line(final_segmented_char_candidate.copy())
            # word_baseline_height = self.base_end - self.base_start
            body_v_projection = self.vertical_projection(
                final_segmented_char_candidate
            )
            # Convert from oneline coordinat to word coordinat
            x_b4_marker = marker_pos[final_group_marker[char_list[0]][0]][0]\
                - wall[0]
            # Check if marker is not in 0 x coordinat to be able looping
            if x_b4_marker > 0:
                for x in range(0, x_b4_marker)[::-1]:
                    # if body_v_projection[x] > 2 * word_baseline_height:
                    if body_v_projection[x] > 2 * oneline_height_sorted:
                        final_segmented_char = final_segmented_char_candidate[
                            :, x+1:wall[1]
                        ]
                        pass_x1 += x+1
                        break
                    else:
                        final_segmented_char = final_segmented_char_candidate
            else:
                final_segmented_char = final_segmented_char_candidate
            print('__Final word only have one marker__')
            return final_segmented_char, pass_x1

        if len(final_group_marker) > 1:  # At least there are two chars or more
            final_segmented_char_candidate = self.image_final_sorted[
                # Cut image after the most right sided group marker on char_2
                :, marker_pos[final_group_marker[char_list[1]][0]][1]:wall[1]
            ]
            pass_x1 = marker_pos[final_group_marker[char_list[1]][0]][1]
#             cv2.imshow('final_char_candidate', final_segmented_char_candidate)
#             cv2.waitKey(0)
            # Getting differentiation list on every word pixel
            body_v_proj = self.vertical_projection(
                final_segmented_char_candidate
            )
            diff = [0]
            for x in range(len(body_v_proj)):
                if x < len(body_v_proj) - 1:
                    temp_diff = int(body_v_proj[x + 1])\
                                - int(body_v_proj[x])
                    diff.append(temp_diff)

#             print(diff)
#             plt.subplot(211), plt.imshow(final_segmented_char_candidate)
#             plt.subplot(212), plt.plot(
#                 np.arange(0, len(body_v_proj), 1), body_v_proj
#             )
#             plt.show()
#             cv2.waitKey(0)

            # Getting 1st char by it's 2nd marker
            # x1_2nd_marker = final_h_list_sorted[right_side_2nd
            #                                     ][1][0]
            x2_char = marker_pos[final_group_marker[char_list[1]][0]][1]
            x2_2nd_marker = 0
            x1_1st_marker = marker_pos[final_group_marker[char_list[0]][
                len(final_group_marker[char_list[0]])-1]][0] - x2_char
            save_sistent = {}
            counting = False
            temp = 0
            count_dinat = 0
            count_sistent = 0
            if x1_1st_marker - x2_2nd_marker > 0:  # marker is'nt ovrlapped
                for x in range(x2_2nd_marker, x1_1st_marker)[::-1]:
                    count_dinat += 1
                    # if abs(diff[x]) > oneline_height_sorted:
                    #     break
                    if diff[x] == 0:
                        count_sistent += 1
                        counting = True
                    if ((diff[x] > 0 or diff[x] < 0)
                            or x == x2_2nd_marker) and counting:
                        save_sistent[count_dinat] = count_sistent
                        count_sistent = 0
                        counting = False
#                 print(save_sistent)
#                 plt.subplot(211), plt.imshow(final_segmented_char_candidate[
#                     :, x2_2nd_marker:x1_1st_marker
#                 ])
#                 plt.subplot(212), plt.plot(
#                     np.arange(x2_2nd_marker, x1_1st_marker, 1),
#                     diff[x2_2nd_marker:x1_1st_marker]
#                 )
#                 plt.show()
#                 cv2.waitKey(0)
                # cut at the most consistent diff equal 0
                cut = 1/2
                if save_sistent != {}:
                    # key is coordinat
                    for key in save_sistent:
                        # compare by a/b c/d (is it close enough
                        # to be called consistent)
                        # enough = save_sistent[key] / key
                        enough = save_sistent[key] / (key-(round(
                                                      save_sistent[key]/2)))
                        if enough > temp:
                            the_sistent = key
                            temp = enough
                    x1_char = x2_2nd_marker \
                        + (x1_1st_marker - x2_2nd_marker - the_sistent)\
                        + round(cut * save_sistent[the_sistent])
                    # + the_sistent \
                    final_segmented_char = final_segmented_char_candidate[
                            :, x1_char:wall[1]
                        ]
                    pass_x1 += x1_char
#                     print('1/2 of the most consistent')
                else:
                    final_segmented_char = final_segmented_char_candidate
#                     print('Consistent hist not found between marker')
            else:
                final_segmented_char = final_segmented_char_candidate
#                 print('1st marker and 2nd marker is overlaped')

            return final_segmented_char, pass_x1
