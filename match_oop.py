# import argparse
import numpy as np 
import imutils
import glob
import cv2

#__argument parser 
# ap = argparse.ArgumentParser()
# #ap.add_argument("-t", "--template", required=True, help="Path to template image")
# ap.add_argument("-i", "--images", required=True, help="Path to matched images")
# ap.add_argument("-v", "--visualize", help="Flag indicating wether or not to visualize each iteration")
# args = vars(ap.parse_args())

class Marker:
    def __init__ (self, **kwargs):
        self._Data=kwargs
    
    def GetTemplate_Location(self):
        return self._Data["template_loc"]
    # def GetImage_Location(self):
    #     return self._Data["image_loc"]
    def GetTemplate_Threshold(self):
        return self._Data["template_thresh"]
    def GetNMS_Threshold(self):
        return self._Data["nms_thresh"]
    def GetImage(self):
        return self._Data["image"]


    def non_max_suppression_slow(self, boxes, overlapThresh):
	    # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

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
 
    def Match_Template(self, visualize=False):
        #__get template
        # print('.')
        template = cv2.imread(self.GetTemplate_Location())
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        ##Canny edge
        #template = cv2.Canny(template, 50, 200)
        ##Otsu threshold
        #ret_temp,template = cv2.threshold(template,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ##Simple threshold
        #ret_temp, template = cv2.threshold(template,127,255,cv2.THRESH_BINARY)
        ##Adaptive threshold value is the mean of neighbourhood area
        #template = cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        ##Adaptive threshold value is the weighted sum of neighbourhood values where weights are a gaussian window
        template = cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        #cv2.imshow("Template", template)
        (tH,tW) = template.shape[:2]

        #__get image
        # image = cv2.imread(self.GetImage_Location())
        # print(self.GetImage_Location())
        # self.image=image
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray',gray)
        image = self.GetImage()
        # boundingBoxes = np.ones((1,4), dtype=int)
        boundingBoxes = []
        max_value_list = []
        #__loop over scaled image (start stop numstep) from the back
        for scale in np.linspace(0.2, 2.0, 100)[::-1]:
            resized = imutils.resize(image, width= int(image.shape[1] * scale))
            r = image.shape[1] / float(resized.shape[1])
            #__if resized image smaller than template, then break the loop
            if  resized.shape[0] < tH or resized.shape[1] < tW:
                break

            #__preprocessing resized image and then apply template matching
            ##Cannyedge
            #edged = cv2.Canny(resized, 50, 200)
            ##Otsu threshold
            #ret_img,resized = cv2.threshold(resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ##Simple threshold
            #ret_img, resized = cv2.threshold(resized,127,255,cv2.THRESH_BINARY)
            ##Adaptive threshold value is the mean of neighbourhood area
            #resized = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
            #Adaptive threshold value is the weighted sum of neighbourhood values where weights are a gaussian window
            resized = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            (_,maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if visualize == True:
                #clone = np.dstack([edged, edged, edged])
                clone = np.dstack([resized, resized, resized])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0,0,255), 2)
                cv2.imshow("Visualizing", clone)
                cv2.waitKey(0)
                
            if maxVal > self.GetTemplate_Threshold():
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
                temp = [startX, startY, endX, endY]
                # boundingBoxes = np.append(boundingBoxes, [temp], axis=0)
                # max_value_list = np.append(max_value_list, [maxVal], axis=0)
                boundingBoxes.append(temp)
                max_value_list.append(maxVal)
                #print("Max val = {} location {}".format(maxVal, temp))
                

        #__if detected on this scale size
        # if len(boundingBoxes) > 1:
        #     boundingBoxes = np.delete(boundingBoxes, 0, axis=0)
        boundingBoxes = np.array(boundingBoxes)
        pick = self.non_max_suppression_slow(boundingBoxes, self.GetNMS_Threshold())
            # print("{} {}".format(len(pick), self.GetTemplate_Location()))
            # print(pick)
            # for (startX, startY, endX, endY) in pick:
            #     cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # return image, boundingBoxes, max_value_list
        if len(max_value_list) > 1:
            max_local_value = max(max_value_list)
        if len(max_value_list) == 1:
            max_local_value = max_value_list
        if max_value_list == []:
            pick = 0
            max_local_value = 0
        # cv2.imshow('IMAGE MATCH ()',image)
        return pick, max_local_value
            

class Font_Wrapper(Marker):
    def __init__ (self, nms_thresh=0.3, visualize= False, **kwargs):
        # print("INIT!")
        self._Data = kwargs
        self.nms_thresh = nms_thresh
        self.visualize = visualize
        self.image_location = self._Data["image_loc"]
        self.marker_location = self._Data["loc_list"]
        self.marker_thresh = self._Data["thresh_list"]
        self.pocket = {}
        # super().__init__()
        colour_tanwin=[(255,0,255), (0,0,255), (128,0,128), (0,0,128)]
        colour_nun   =[(255,0,0), (128,0,0), (255,99,71), (220,20,60), (139,0,0)]
        colour_mim   =[(154,205,50), (107,142,35), (85,107,47), (0,128,0), (34,139,34)]
        t = 0
        m = 0
        n = 0
        self.temp_colour = self.GetMarker_Thresh().copy()
        for key in self.GetMarker_Thresh().keys():
            # print(key)
            x = key.split('_') 
            if x[0] == 'tanwin':
                self.temp_colour[key] = colour_tanwin[t]
                if t+1 > len(colour_tanwin) - 1:
                    t = 0
                else:
                    t+=1
            if x[0] == 'nun':
                self.temp_colour[key] = colour_nun[n]
                if n+1 > len(colour_nun) - 1:
                    n = 0
                else:
                    n+=1
            if x[0] == 'mim':
                self.temp_colour[key] = colour_mim[m]
                if m+1 > len(colour_mim) - 1:
                    m = 0
                else:
                    m+=1
        # print(self.temp_colour)
        # tanwin = 0
        # nun    = 0
        # mim    = 0
        # for key in self.GetMarker_Thresh().keys():
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
        # for x in range(len(self.GetMarker_Thresh())):
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

    def GetMarker_Thresh(self):
        return self.marker_thresh
    def GetMarker_Location(self):
        return self.marker_location
    def GetImage_Location(self):
        return self.image_location
    def GetOriginalImage(self):
        original_image = cv2.imread(self.GetImage_Location())
        return original_image
    def GetPocketData(self):
        return self.pocket
    def Display_Marker_Result(self):
        rectangle_image = self.GetOriginalImage()
        found = False
        for key in self.GetMarker_Thresh().keys():
            if type(self.GetPocketData()['box_' + key]) == type(np.array([])) :
                for (startX, startY, endX, endY) in self.GetPocketData()['box_' + key]:
                    cv2.rectangle(rectangle_image, (startX, startY), (endX, endY), self.temp_colour[key], 2)
                # print(self.pick_colour[x])
                found = True        
        if found == True:
            print('<<<<<<<< View Result >>>>>>>>')
            cv2.imshow("Detected Image", rectangle_image)
        else:
            cv2.imshow("Original Image", rectangle_image)  
            print('not found')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def run(self, view=False):
         #__tanwin
        # print(self.GetMarker_Thresh())
        print('run() Marker Font')
        # ori_image = cv2.imread(self.GetImage_Location())
        # print(self.GetOriginalImage())
        # self.original_image = cv2.imread(self.GetImage_Location())
        gray = cv2.cvtColor(self.GetOriginalImage(), cv2.COLOR_BGR2GRAY)
        # cv2.imshow('test', image)
       
        # image = self.GetOriginalImage()
        # image_orig=self.GetOrig_Image()
        pocketData={}
        for x in range(len(self.GetMarker_Thresh())):
            # print(len(template_thresh))
            # print(list(template_thresh.values())[x])
            super().__init__(image = gray, template_thresh = list(self.GetMarker_Thresh().values())[x], 
                             template_loc = self.GetMarker_Location()[x], nms_thresh = self.nms_thresh)
            (pocketData[x],pocketData[x+len(self.GetMarker_Thresh())]) = super().Match_Template(visualize=self.visualize)
            # print(type(pocketData[x]))

        for x in range(len(self.GetMarker_Thresh())):
            temp = list(self.GetMarker_Thresh().keys())[x]
            # box = 'box_'+ temp
            self.GetPocketData()[temp] = pocketData[x+len(self.GetMarker_Thresh())]
            self.GetPocketData()['box_'+ temp]  = pocketData[x]

        if view == True:
            self.Display_Marker_Result()
            # rectangle_image = self.GetOriginalImage()
            # found = False
            # for x in range(len(self.GetMarker_Thresh())):
            #     if type(pocketData[x]) == type(np.array([])) :
            #         for (startX, startY, endX, endY) in pocketData[x]:
            #             cv2.rectangle(rectangle_image, (startX, startY), (endX, endY), self.pick_colour[x], 2)
            #         # print(self.pick_colour[x])
            #         found = True        
            # if found == True:
            #     print('<<<<<<<< View Result >>>>>>>>')
            #     cv2.imshow("Detected Image", rectangle_image)
            # else:
            #     cv2.imshow("Original Image", rectangle_image)  
            #     print('not found')

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    
        # return pocket


def main():
    # for imagePath in sorted(glob.glob(args["images"] + "/*.png")):
    
    for imagePath in sorted(glob.glob("test" + "/*.png")):
        print('________________Next File_________________')
        #__LPMQ_Font
        # print("LPMQ")
        loc_list_LPMQ = sorted(glob.glob('./marker/LPMQ/*.png'))
        LPMQ = Font_Wrapper( thresh_list={'tanwin_1'    : 0.7, 'tanwin_2'    : 0.7,
                                          'nun_stand'   : 0.7, 'nun_beg_1'   : 0.7,
                                          'nun_beg_2'   : 0.7, 'nun_mid'     : 0.7, 
                                          'nun_end'     : 0.7, 'mim_stand'   : 0.7, 
                                          'mim_beg'     : 0.7, 'mim_mid'     : 0.7, 
                                          'mim_end_1'   : 0.7, 'mim_end_2'   : 0.7 },
                            loc_list=loc_list_LPMQ, image_loc= imagePath,
                            visualize=False, nms_thresh=0.3)
        #__AlQalam_Font
        # print("AlQalam")
        loc_list_AlQalam = sorted(glob.glob('./marker/AlQalam/*.png'))
        AlQalam = Font_Wrapper( thresh_list={'tanwin_1' : 0.7, 'tanwin_2'   : 0.7,
                                          'nun_stand'   : 0.7, 'nun_beg'    : 0.7,
                                          'nun_mid'     : 0.7, 'nun_end'    : 0.7,
                                          'mim_stand'   : 0.7, 'mim_beg'    : 0.7,
                                          'mim_mid'     : 0.7, 'mim_end'    : 0.7 },
                                loc_list = loc_list_AlQalam, image_loc = imagePath,
                                visualize=False, nms_thresh=0.3)
        #__meQuran_Font
        # print("meQuran")
        loc_list_meQuran = sorted(glob.glob('./marker/meQuran/*.png'))
        meQuran = Font_Wrapper( thresh_list={'tanwin_1' : 0.7, 'tanwin_2'   : 0.7,
                                          'nun_stand'   : 0.7, 'nun_beg_1'  : 0.7,
                                          'nun_beg_2'   : 0.7, 'nun_mid'    : 0.7,
                                          'nun_end'     : 0.7, 'mim_stand'  : 0.7, 
                                          'mim_beg'     : 0.7, 'mim_mid'    : 0.7, 
                                          'mim_end_1'   : 0.7, 'mim_end_2'  : 0.7 },
                                loc_list=loc_list_meQuran, image_loc= imagePath,
                                visualize=False, nms_thresh=0.3)
        #__PDMS_Font
        # print("PDMS")
        loc_list_PDMS = sorted(glob.glob('./marker/PDMS/*.png'))
        PDMS = Font_Wrapper( thresh_list={'tanwin_1'    : 0.7, 'tanwin_2'   : 0.7,
                                          'nun_stand'   : 0.7, 'nun_beg'    : 0.7,
                                          'nun_mid'     : 0.7, 'nun_end'    : 0.7,
                                          'mim_stand'   : 0.7, 'mim_beg'    : 0.7,
                                          'mim_mid'     : 0.7, 'mim_end'    : 0.7 },
                            loc_list=loc_list_PDMS, image_loc= imagePath,
                            visualize=False, nms_thresh=0.3)

        # __Font_Processing
        Processing_Font_Object = [LPMQ, AlQalam, meQuran, PDMS]

        max_font_value = 0
        for font_object in Processing_Font_Object:
            font_object.run()
            for value in font_object.GetPocketData().values():
                if type(value) == float:
                    if value > max_font_value:
                        max_font_value = value
                        font_type = font_object
                        
        print(font_type.GetMarker_Location()[0])                    
        font_type.Display_Marker_Result()


        # AlQalam.run()
        # for key in LPMQ.GetMarker_Thresh().keys():
        #     print(key)
        #     print(int(key))

        # print(type(LPMQ.GetPocketData()['tanwin_1']))
        # LPMQ.Display_Marker_Result()
        # print(LPMQ.GetPocketData()['tanwin_1'])
        # pocket_meQuran = meQuran.run(view=True)
        # pocket_PDMS = PDMS.run(view=True)
        # pocket_LPMQ = LPMQ.run(view=True)
        # cv2.imshow('main loop', LPMQ.GetOriginalImage())

        
        # print(pocketData)

        # print(LPMQ.GetMarker_Thresh())

    cv2.destroyAllWindows() 

if __name__ == '__main__':main()
