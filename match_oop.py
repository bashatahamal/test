import numpy as np 
import argparse
import imutils
import glob
import cv2

#__argument parser 
ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True, help="Path to matched images")
ap.add_argument("-v", "--visualize", help="Flag indicating wether or not to visualize each iteration")
args = vars(ap.parse_args())

class Marker:
    def __init__ (self, **kwargs):
        self._Data=kwargs
    
    def GetTemplate_Location(self):
        return self._Data["template_loc"]
    def GetImage_Location(self):
        return self._Data["image_loc"]
    def GetTemplate_Threshold(self):
        return self._Data["template_thresh"]
    def GetNMS_Threshold(self):
        return self._Data["nms_thresh"]
    def GetOriginalImage(self):
        return self.image

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
        
        image = cv2.imread(self.GetImage_Location())
        self.image=image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boundingBoxes = np.ones((1,4), dtype=int)
        max_value_list = []
        #__loop over scaled image (start stop numstep) from the back
        for scale in np.linspace(0.2, 2.0, 100)[::-1]:
            resized = imutils.resize(gray, width= int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
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
                boundingBoxes = np.append(boundingBoxes, [temp], axis=0)
                max_value_list = np.append(max_value_list, [maxVal], axis=0)
                #print("Max val = {} location {}".format(maxVal, temp))
                

        #__if detected on this scale size
        if len(boundingBoxes) > 1:
            boundingBoxes = np.delete(boundingBoxes, 0, axis=0)
            pick = self.non_max_suppression_slow(boundingBoxes, self.GetNMS_Threshold())
            print(len(pick))
            print(pick)
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
        return pick, max_local_value
            




def main():
    for imagePath in glob.glob(args["images"] + "/*.png"):
        
        #___tanwin_LPMQ
        template_loc_tanwin   = "/home/mhbrt/Desktop/Wind/Multiscale/marker/tanwin_LPMQ.png"
        template_loc_tanwin_1 = "/home/mhbrt/Desktop/Wind/Multiscale/marker/tanwin_1_LPMQ.png"
        thresh_tanwin   = 0.7
        thresh_tanwin_1 = 0.7
        nms_thresh = 0.3
        
        tanwin_LPMQ   = Marker(template_loc=template_loc_tanwin, image_loc=imagePath, template_thresh = thresh_tanwin, nms_thresh = nms_thresh )
        tanwin_1_LPMQ = Marker(template_loc=template_loc_tanwin_1, image_loc=imagePath, template_thresh = thresh_tanwin_1, nms_thresh = nms_thresh )

        (box_tanwin, value_tanwin) = tanwin_LPMQ.Match_Template(visualize=False)
        (box_tanwin_1, value_tanwin_1) = tanwin_1_LPMQ.Match_Template(visualize=False)

        #__nun_sukun_LPMQ
        template_loc_nun_stand_LPMQ = "/home/mhbrt/Desktop/Wind/Multiscale/marker/nun_stand_LPMQ.png"
        template_loc_nun_mid_LPMQ   = "/home/mhbrt/Desktop/Wind/Multiscale/marker/nun_mid_LPMQ.png"
        template_loc_nun_end_LPMQ   = "/home/mhbrt/Desktop/Wind/Multiscale/marker/nun_end_LPMQ.png"
        template_loc_nun_beg_LPMQ   = "/home/mhbrt/Desktop/Wind/Multiscale/marker/nun_beg_LPMQ.png"
        thresh_nun_stand_LPMQ = 0.7
        thresh_nun_mid_LPMQ   = 0.7
        thresh_nun_end_LPMQ   = 0.7
        thresh_nun_beg_LPMQ   = 0.7
        
        nun_stand_LPMQ = Marker(template_loc=template_loc_nun_stand_LPMQ, image_loc=imagePath, template_thresh = thresh_nun_stand_LPMQ, nms_thresh = nms_thresh )
        nun_mid_LPMQ   = Marker(template_loc=template_loc_nun_mid_LPMQ, image_loc=imagePath, template_thresh = thresh_nun_mid_LPMQ, nms_thresh = nms_thresh )
        nun_end_LPMQ   = Marker(template_loc=template_loc_nun_end_LPMQ, image_loc=imagePath, template_thresh = thresh_nun_end_LPMQ, nms_thresh = nms_thresh )
        nun_beg_LPMQ   = Marker(template_loc=template_loc_nun_beg_LPMQ, image_loc=imagePath, template_thresh = thresh_nun_beg_LPMQ, nms_thresh = nms_thresh )

        (box_nun_stand_LPMQ, value_nun_stand_LPMQ) = nun_stand_LPMQ.Match_Template(visualize=False)
        (box_nun_mid_LPMQ, value_nun_mid_LPMQ) = nun_mid_LPMQ.Match_Template(visualize=False)
        (box_nun_end_LPMQ, value_nun_end_LPMQ) = nun_end_LPMQ.Match_Template(visualize=False)
        (box_nun_beg_LPMQ, value_nun_beg_LPMQ) = nun_beg_LPMQ.Match_Template(visualize=False)

        #__mim_sukun_LPMQ
        template_loc_mim_stand_LPMQ = "/home/mhbrt/Desktop/Wind/Multiscale/marker/mim_stand_LPMQ.png"
        template_loc_mim_mid_LPMQ   = "/home/mhbrt/Desktop/Wind/Multiscale/marker/mim_mid_LPMQ.png"
        template_loc_mim_end_LPMQ   = "/home/mhbrt/Desktop/Wind/Multiscale/marker/mim_end_LPMQ.png"
        template_loc_mim_beg_LPMQ   = "/home/mhbrt/Desktop/Wind/Multiscale/marker/mim_beg_LPMQ.png"
        thresh_mim_stand_LPMQ = 0.7
        thresh_mim_mid_LPMQ   = 0.7
        thresh_mim_end_LPMQ   = 0.7
        thresh_mim_beg_LPMQ   = 0.7

        mim_stand_LPMQ = Marker(template_loc=template_loc_mim_stand_LPMQ, image_loc=imagePath, template_thresh = thresh_mim_stand_LPMQ, nms_thresh = nms_thresh )
        mim_mid_LPMQ   = Marker(template_loc=template_loc_mim_mid_LPMQ, image_loc=imagePath, template_thresh = thresh_mim_mid_LPMQ, nms_thresh = nms_thresh )
        mim_end_LPMQ   = Marker(template_loc=template_loc_mim_end_LPMQ, image_loc=imagePath, template_thresh = thresh_mim_end_LPMQ, nms_thresh = nms_thresh )
        mim_beg_LPMQ   = Marker(template_loc=template_loc_mim_beg_LPMQ, image_loc=imagePath, template_thresh = thresh_mim_beg_LPMQ, nms_thresh = nms_thresh )

        (box_mim_stand_LPMQ, value_mim_stand_LPMQ) = mim_stand_LPMQ.Match_Template(visualize=False)
        (box_mim_mid_LPMQ, value_mim_mid_LPMQ) = mim_mid_LPMQ.Match_Template(visualize=False)
        (box_mim_end_LPMQ, value_mim_end_LPMQ) = mim_end_LPMQ.Match_Template(visualize=False)
        (box_mim_beg_LPMQ, value_mim_beg_LPMQ) = mim_beg_LPMQ.Match_Template(visualize=False)

       #__display cummulative anotation detected
        image=tanwin_LPMQ.GetOriginalImage()
        found=False
        if value_tanwin_1 != 0:
            for (startX, startY, endX, endY) in box_tanwin_1:
                cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 255), 2)
            found=True
        if value_tanwin != 0:
            for (startX, startY, endX, endY) in box_tanwin:
                cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 127), 2)
            found=True
        if value_nun_stand_LPMQ != 0:
            for (startX, startY, endX, endY) in box_nun_stand_LPMQ:
                cv2.rectangle(image, (startX, startY), (endX, endY), (102, 102, 255), 2)
            found=True
        if value_nun_mid_LPMQ != 0:
            for (startX, startY, endX, endY) in box_nun_mid_LPMQ:
                cv2.rectangle(image, (startX, startY), (endX, endY), (51, 51, 255), 2)
            found=True
        if value_nun_end_LPMQ != 0:
            for (startX, startY, endX, endY) in box_nun_end_LPMQ:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            found=True
        if value_nun_beg_LPMQ != 0:
            for (startX, startY, endX, endY) in box_nun_beg_LPMQ:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 102), 2)
            found=True
        if value_mim_stand_LPMQ != 0:
            for (startX, startY, endX, endY) in box_mim_stand_LPMQ:
                cv2.rectangle(image, (startX, startY), (endX, endY), (102, 255, 255), 2)
            found=True
        if value_mim_mid_LPMQ != 0:
            for (startX, startY, endX, endY) in box_mim_mid_LPMQ:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 153, 0), 2)
            found=True
        if value_mim_end_LPMQ != 0:
            for (startX, startY, endX, endY) in box_mim_end_LPMQ:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 255), 2)
            found=True
        if value_mim_beg_LPMQ != 0:
            for (startX, startY, endX, endY) in box_mim_beg_LPMQ:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 204, 204), 2)
            found=True
        
        # cv2.imshow("tanwin_LPMQ",result_tanwin) 
        # cv2.imshow("tanwin_1_LPMQ",result_tanwin_1)
        if found == True:
            cv2.imshow("Cummulative", image)
        else:
            cv2.imshow("Original", image)  

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # if max_value != []:
        #     print(max(max_value))
        ##continue next file if not detected
        # if max_value == []:
        #     continue
        # print(max_value)
        

    cv2.destroyAllWindows() 

if __name__ == '__main__':main()
