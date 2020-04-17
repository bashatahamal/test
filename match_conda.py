# import argparse
import matplotlib.pyplot as plt
import numpy as np
import imutils
import glob
import cv2

# Argument parser
# ap = argparse.ArgumentParser()
# ap.add_argument(
#     "-t", "--template", required=True, help="Path to template image")
# ap.add_argument(
#     "-i", "--images", required=True, help="Path to matched images")
# ap.add_argument(
#     "-v", "--visualize", help="Flag indicating to visualize each iteration")
# args = vars(ap.parse_args())


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
            # self.key = self._Data[data]
            # print(type(self.data))
        # print(self.key)

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
        # print(self.temp_colour)
        # tanwin = 0
        # nun    = 0
        # mim    = 0
        # for key in self.get_marker_thresh().keys():
        #     x = key.split('_')
        #     if x[0] == 'tanwin':
        #         tanwin+=1
        #     if x[0] == 'nun':
        #         nun+=1
        #     if x[0] == 'mim':
        #         mim+=1
        # print('Tanwin {} nun {} mim {}'.format(tanwin, nun, mim))
        # self.pick_colour  =[]
        # reserved_tanwin = tanwin
        # reserved_nun    = nun
        # reserved_mim    = mim
        # for x in range(len(self.get_marker_thresh())):
        #     if reserved_tanwin <= len(colour_tanwin):
        #         if tanwin > 0:
        #             self.pick_colour.append(colour_tanwin[tanwin-1])
        #             tanwin-=1
        #             # print(tanwin)
        #     else :
        #         if tanwin > 0:
        #             self.pick_colour.append(colour_tanwin[0])
        #             tanwin-=1
        #     if reserved_nun <= len(colour_nun):
        #         if tanwin <= 0 and nun > 0:
        #             self.pick_colour.append(colour_nun[nun-1])
        #             nun-=1
        #     else:
        #         if tanwin <= 0 and nun > 0:
        #             self.pick_colour.append(colour_nun[0])
        #             nun-=1
        #     if reserved_mim <= len(colour_mim):
        #         if nun <= 0 and mim > 0:
        #             self.pick_colour.append(colour_mim[mim-1])
        #             mim-=1
        #     else:
        #         if nun <= 0 and mim > 0:
        #             self.pick_colour.append(colour_mim[0])
        #             mim-=1
        # print(self.pick_colour)

    def get_marker_thresh(self):
        return self.marker_thresh

    def get_marker_location(self):
        return self.marker_location

    def get_image_location(self):
        return self.image_location

    def get_original_image(self):
        original_image = cv2.imread(self.get_image_location())
        return original_image

    def get_object_result(self):
        return self.pocket

    def get_image(self):
        return self.image

    def get_object_name(self):
        return self.get_marker_location()[0].split('/')[2]

    def display_marker_result(self):
        rectangle_image = self.get_original_image()
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
        if found:
            print('<<<<<<<< View Result >>>>>>>>')
            cv2.imshow("Detected Image_" + self.get_object_name(),
                       rectangle_image)
        else:
            cv2.imshow("Original Image", rectangle_image)
            print('not found')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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


def font(imagePath, image):
    # LPMQ_Font
    # print("LPMQ")
    loc_list_LPMQ = sorted(glob.glob('./marker/LPMQ/*.png'))
    font_LPMQ = FontWrapper(thresh_list={'tanwin_1': 0.7, 'tanwin_2': 0.7,
                                         'nun_stand': 0.7, 'nun_beg_1': 0.7,
                                         'nun_beg_2': 0.7, 'nun_mid': 0.7,
                                         'nun_end': 0.7, 'mim_stand': 0.7,
                                         'mim_beg': 0.7, 'mim_mid': 0.7,
                                         'mim_end_1': 0.7, 'mim_end_2': 0.7},
                            loc_list=loc_list_LPMQ, image_loc=imagePath,
                            image=image, visualize=False, nms_thresh=0.3,
                            numstep=30)
    # AlQalam_Font
    # print("AlQalam")
    loc_list_AlQalam = sorted(glob.glob('./marker/AlQalam/*.png'))
    font_AlQalam = FontWrapper(thresh_list={'tanwin_1': 0.7, 'tanwin_2': 0.7,
                                            'nun_stand': 0.7, 'nun_beg': 0.7,
                                            'nun_mid': 0.7, 'nun_end': 0.7,
                                            'mim_stand': 0.7, 'mim_beg': 0.7,
                                            'mim_mid': 0.7, 'mim_end': 0.7},
                               loc_list=loc_list_AlQalam, image_loc=imagePath,
                               image=image, visualize=False, nms_thresh=0.3)
    # meQuran_Font
    # print("meQuran")
    loc_list_meQuran = sorted(glob.glob('./marker/meQuran/*.png'))
    font_meQuran = FontWrapper(thresh_list={'tanwin_1': 0.7, 'tanwin_2': 0.7,
                                            'nun_stand': 0.7, 'nun_beg_1': 0.7,
                                            'nun_beg_2': 0.7, 'nun_mid': 0.7,
                                            'nun_end': 0.7, 'mim_stand': 0.7,
                                            'mim_beg': 0.7, 'mim_mid': 0.7,
                                            'mim_end_1': 0.7, 'mim_end_2': 0.7},
                               loc_list=loc_list_meQuran, image_loc=imagePath,
                               image=image, visualize=False, nms_thresh=0.3)
    # PDMS_Font
    # print("PDMS")
    loc_list_PDMS = sorted(glob.glob('./marker/PDMS/*.png'))
    font_PDMS = FontWrapper(thresh_list={'tanwin_1': 0.7, 'tanwin_2': 0.7,
                                         'nun_stand': 0.7, 'nun_beg': 0.7,
                                         'nun_mid': 0.7, 'nun_end': 0.7,
                                         'mim_stand': 0.7, 'mim_beg': 0.7,
                                         'mim_mid': 0.7, 'mim_end': 0.7},
                            loc_list=loc_list_PDMS, image_loc=imagePath,
                            image=image, visualize=False, nms_thresh=0.3)

    list_object_font = [font_LPMQ, font_AlQalam, font_meQuran, font_PDMS]

    return list_object_font


def vertical_projection(image):
    image[image < 127] = 1
    image[image >= 127] = 0
    projection = np.sum(image, axis=0)

    return projection


def horizontal_projection(image):
    image[image < 127] = 1
    image[image >= 127] = 0
    projection = np.sum(image, axis=1)

    return projection

# def main():
    # for imagePath in sorted(glob.glob(args["images"] + "/*.png")):

for imagePath in sorted(glob.glob("temp" + "/*.png")):
    print('________________Next File_________________')
    original_image = cv2.imread(imagePath)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(gray, 50, 200)
    # Otsu threshold
    # ret_img, image1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY
    #                                + cv2.THRESH_OTSU)
    # Simple threshold
    # ret_img, image2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Adaptive threshold value is the mean of neighbourhood area
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               cv2.THRESH_BINARY, 11, 2)
    # Adaptive threshold value is the weighted sum of neighbourhood
    # values where weights are a gaussian window
    image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow('otsu', image1)
    # cv2.imshow('simple', image2)
    # cv2.imshow('adapt mean', image3)
    # cv2.imshow('adapt gaussian', image4)
    # cv2.waitKey(0)
    
    # Font_Processing
    font_list = font(imagePath=imagePath, image=gray)

    max_font_value = 0
    font_type = 0
    numstep = 20
    for font_object in font_list:
        font_object.run(numstep=numstep)
        for value in font_object.get_object_result().values():
            # print(value)
            if type(value) == float:
                if value > max_font_value:
                    max_font_value = value
                    font_type = font_object

    if isinstance(font_type, type(font_list[0])):
        font_type.display_marker_result()
    else:
        print('Not a valuable result found check the numstep!')


    pixel_gray = image
    # pixel_gray = cv2.cvtColor(pixel_value, cv2.COLOR_BGR2GRAY)
    v_projection = vertical_projection(pixel_gray.copy())
    h_projection = horizontal_projection(pixel_gray.copy())
    # plt.subplot(212), plt.imshow(pixel_gray)
    # plt.subplot(221), plt.plot(np.arange(0, len(v_projection), 1), v_projection)
    # plt.subplot(222), plt.plot(np.arange(0, len(h_projection), 1), h_projection)
    # # plt.xlim([0,256])
    # # plt.show()
    # cv2.waitKey(0)

    diff = [0]
    for x in range(len(h_projection)):
        if x > 0:
            temp_diff = abs(int(h_projection[x]) - int(h_projection[x-1]))
            diff.append(temp_diff)

    base_start = 0
    base_end = 0
    temp = 0
    for x in range(len(diff)):
        if diff[x] > temp:
            temp = diff[x]
            base_end = x

    temp = 0 
    for x in range(len(diff)):
        if x == base_end:
            continue
        if diff[x] > temp:
            temp = diff[x]
            base_start = x

    cv2.line(original_image, (0, base_start), (len(v_projection),
             base_start), (0, 255, 0), 2)
    cv2.line(original_image, (0, base_end), (len(v_projection),
             base_end), (0, 255, 0), 2)

    up_flag = 0
    down_flag = 0
    pixel_limit = 4
    start_to_end = 0
    end_to_start = 0
    start_point = []
    for x in range(len(h_projection)):
        if h_projection[x]==0 and up_flag==1: #and start_to_end>pixel_limit:
            start_point.append(x)
            down_flag = 1
                # if len(start_point)!=len(end_point):
                #     del(start_point[len(start_point)
                #     -(len(start_point)-len(end_point))])
                # count=0
            # print('end')
            up_flag = 0

        if up_flag==1:
            start_to_end += 1
        else:
            start_to_end = 0
        
        if down_flag==1:
            end_to_start += 1
            # print(end_to_start)
        else:
            end_to_start = 0
             
        if h_projection[x]>0 and up_flag==0:
            # if count>=pixel_limit
            start_point.append(x)
            if down_flag==1 and end_to_start<pixel_limit:
                del(start_point[len(start_point)-1])
                del(start_point[len(start_point)-1])
                # print('delete')
            # print(count)
            up_flag = 1
            down_flag = 0
        # count+=1

    up_flag = 0
    down_flag = 0
    pixel_limit_v = 4
    start_to_end_v = 0
    end_to_start_v = 0
    start_point_v = []
    for x in range(len(v_projection)):
        if v_projection[x]>0 and up_flag==0:
            start_point_v.append(x)
            up_flag = 1
            down_flag =0

        if v_projection[x]==0 and up_flag==1:
            start_point_v.append(x)
            down_flag = 1
            up_flag = 0

        if up_flag==1:
            start_to_end_v += 1
        else:
            start_to_end_v = 0

        if down_flag==1:
            end_to_start_v += 1
        else:
            end_to_start_v = 0
            
    # Even is begining of line and Odd is end of line
    for x in range(len(start_point)):
        # cv2.line(original_image, (0, start_point[x]), (len(v_projection),
        #          start_point[x]), (0, 0, 255), 2)
        # print(x)
        if x%2==0:     # Start_point
            cv2.line(original_image, (0, start_point[x]), (len(v_projection),
                     start_point[x]), (0, 0, 255), 2)
            # print(x)
        else:         # End_point
            cv2.line(original_image, (0, start_point[x]), (len(v_projection),
                     start_point[x]), (255, 0, 0), 2)
            # print(x)
            # print('end')

    for x in range(len(start_point_v)):
        if x%2==0:
            cv2.line(original_image, (start_point_v[x], 0), (start_point_v[x],
                     len(h_projection)), (0,0,255), 2)
        else:
            cv2.line(original_image, (start_point_v[x], 0), (start_point_v[x],
                     len(h_projection)), (255,0,0), 2)

    cv2.imshow('line', original_image)
    cv2.waitKey(0)

    print(start_point)
    bag_of_h_crop = {}
    for x in range(len(start_point)):
        if x+2 > len(start_point):
            print('x')
            continue
        if x%2==0:
            bag_of_h_crop[x]=original_image[start_point[x]:start_point[x+1], :]
    # print(bag_of_h_crop)
    for image in bag_of_h_crop:
        cv2.imshow('bag_h', bag_of_h_crop[image])
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

    bag_of_v_crop = {}
    count=0
    for image in bag_of_h_crop:
        # print(image)
        #y = int(image/2)
        for x in range(len(start_point_v)):
            #print('inside loop')
            print(image)
            count+=1
            if x%2==0:
                x1 = start_point[image]
                x2 = start_point[image+1]
                y1 = start_point_v[x]
                y2 = start_point_v[x+1]
                bag_of_v_crop[count]=original_image[x1:x2, y1:y2]
            # print(x1,'_', x2,'_', y1,'_', y2)


    for image in bag_of_v_crop:
        cv2.imshow('Crop Result', bag_of_v_crop[image])
        cv2.waitKey(0)
        # cv2.destroyAllWindows()


cv2.destroyAllWindows()

# cv2.imshow('crop', view)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
# exec(main())
