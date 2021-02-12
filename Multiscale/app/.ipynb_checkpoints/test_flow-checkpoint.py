import sys
import cv2
import pickle
sys.path.insert(1, '/home/mhbrt/Desktop/Wind/Multiscale/')
import complete_flow as flow

model_name = '/home/mhbrt/Desktop/Wind/Multiscale/Colab/best_model_DenseNet_DD.pkl'
model = pickle.load(open(model_name, 'rb'))
print('_LOAD MODEL DONE_')

imagePath = '/home/mhbrt/Desktop/Wind/Multiscale/temp/0v4.jpg'
markerPath = '/home/mhbrt/Desktop/Wind/Multiscale/marker'
img = cv2.imread(imagePath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res = {'AlKareem': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10'], 'AlQalam': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10'], 'Amiri': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10'], 'KFGQPC': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10'], 'LPMQ': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10'], 'Norehidayat': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10'], 'Norehira': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10'], 'Norehuda': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10'], 'PDMS': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10'], 'meQuran': [['0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7', '0.7'], '10']}
# font_list = mess.font(imagePath=imagePath, image=gray, setting=res)
font_list, loc_path = flow.font_list(
    imagePath=imagePath, image=gray, setting=res, markerPath=markerPath)
temp_object = []
imagelist_template_matching_result = []
imagelist_template_scale_visualize = []
# print(font_list)
# print(font_list[1].marker_location)
for font_object in font_list:
    font_object.run()
    imagelist_template_scale_visualize.append(font_object.imagelist_visualize)
    temp_object.append(font_object.get_object_result())
    imagelist_template_matching_result.append(font_object.display_marker_result())
print(temp_object)
final_image_result, normal_processing_result, crop_ratio_processing_result,\
imagelist_horizontal_line_by_eight_conn = flow.big_blok(temp_object, imagePath,
                font_object, model, font_list)
