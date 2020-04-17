import cv2
import numpy as np
import dot


def vertical_projection(image_v):
    image = image_v.copy()
    # cv2.imshow('doing v_projection', image)
    # print(len(image))
    image[image < 127] = 1
    image[image >= 127] = 0
    v_projection = np.sum(image, axis=0)

    return v_projection


def horizontal_projection(image_h):
    image = image_h.copy()
    # cv2.imshow('h', image)
    image[image < 127] = 1
    image[image >= 127] = 0
    h_projection = np.sum(image, axis=1)

    return h_projection


# def base_line(h_projection):
#     diff = [0]
#     for x in range(len(h_projection)):
#         if x > 0:
#             temp_diff = abs(int(h_projection[x]) - int(h_projection[x-1]))
#             diff.append(temp_diff)

#     temp = 0
#     for x in range(len(diff)):
#         if diff[x] > temp:
#             temp = diff[x]
#             base_end = x
#     # Get the 2nd greatest to base_start
#     temp = 0
#     for x in range(len(diff)):
#         if x == base_end:
#             continue
#         if diff[x] > temp:
#             temp = diff[x]
#             base_start = x

#     return base_start, base_end

def just_projection(image_location):
    originalImage = cv2.imread(image_location)
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    ret_img, bw_image = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    height, width = bw_image.shape
    image_body = bw_image
    v_proj = vertical_projection(image_body)
    h_proj = horizontal_projection(image_body)

    # Get cropped coordinat
    y1 = 0
    for x in h_proj:
        if x > 0:
            break
        y1 += 1
    y2 = 0
    for x in h_proj[::-1]:
        if x > 0:
            break
        y2 += 1
    y2 = height - y2
    x1 = 0
    for x in v_proj:
        if x > 0:
            break
        x1 += 1
    x2 = 0
    for x in v_proj[::-1]:
        if x > 0:
            break
        x2 += 1
    x2 = width - x2

    # Get original image
    img_original = image_body[y1:y2, x1:x2]
    # Get concat image to square
    height, width = img_original.shape
    if height > width:
        concatenate_a = int(round(height - width)/2)
        concatenate_b = height - width - concatenate_a
        beside = True
    else:
        concatenate_a = int(round(width - height)/2)
        concatenate_b = width - height - concatenate_a
        beside = False

    if beside:
        concatenate_a = np.full((height, concatenate_a), 255, dtype=np.uint8)
        concatenate_b = np.full((height, concatenate_b), 255, dtype=np.uint8)
        concatImage = np.concatenate(
            (concatenate_a, img_original, concatenate_b), axis=1
        )
    else:
        concatenate_a = np.full((concatenate_a, width), 255, dtype=np.uint8)
        concatenate_b = np.full((concatenate_b, width), 255, dtype=np.uint8)
        concatImage = np.concatenate(
            (concatenate_a, img_original, concatenate_b), axis=0
        )

    return img_original, concatImage, y1, y2, x1, x2


def make_it_square(image_location):
    # targeted_size = 70
    originalImage = cv2.imread(image_location)
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    ret_img, bw_image = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    # bw_image = cv2.adaptiveThreshold(grayImage, 255,
    #                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                  cv2.THRESH_BINARY, 11, 2)
    height, width = bw_image.shape
    image_body = dot.body_and_dot_region(bw_image)
    v_proj = vertical_projection(image_body)
    h_proj = horizontal_projection(image_body)

    # Get cropped coordinat
    y1 = 0
    for x in h_proj:
        if x > 0:
            break
        y1 += 1
    y2 = 0
    for x in h_proj[::-1]:
        if x > 0:
            break
        y2 += 1
    y2 = height - y2
    x1 = 0
    for x in v_proj:
        if x > 0:
            break
        x1 += 1
    x2 = 0
    for x in v_proj[::-1]:
        if x > 0:
            break
        x2 += 1
    x2 = width - x2

    # Get original image
    img_original = image_body[y1:y2, x1:x2]
    # cv2.imshow('test', img_original)
    # cv2.imshow('ori', bw_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Get fully targeted concatenate image
    # height, width = img_original.shape
    # concatenate_l = int(round(targeted_size - width)/2)
    # concatenate_r = targeted_size - width - concatenate_l
    # concatenate_u = int(round(targeted_size - height)/2)
    # concatenate_d = targeted_size - height - concatenate_u

    # concatenate_l = np.full((height, concatenate_l), 255, dtype=np.uint8)
    # concatenate_r = np.full((height, concatenate_r), 255, dtype=np.uint8)
    # full_concatImage = np.concatenate(
    #     (concatenate_l, img_original, concatenate_r), axis=1
    # )
    # concatenate_u = np.full((concatenate_u, targeted_size), 255,
    #                         dtype=np.uint8)
    # concatenate_d = np.full((concatenate_d, targeted_size), 255,
    #                         dtype=np.uint8)
    # full_concatImage = np.concatenate(
    #     (concatenate_u, full_concatImage, concatenate_d), axis=0
    # )
    # cv2.imshow('full concat', full_concatImage)
    # print(height, width)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Get concat image to square
    height, width = img_original.shape
    if height > width:
        concatenate_a = int(round(height - width)/2)
        concatenate_b = height - width - concatenate_a
        beside = True
    else:
        concatenate_a = int(round(width - height)/2)
        concatenate_b = width - height - concatenate_a
        beside = False

    if beside:
        concatenate_a = np.full((height, concatenate_a), 255, dtype=np.uint8)
        concatenate_b = np.full((height, concatenate_b), 255, dtype=np.uint8)
        concatImage = np.concatenate(
            (concatenate_a, img_original, concatenate_b), axis=1
        )
    else:
        concatenate_a = np.full((concatenate_a, width), 255, dtype=np.uint8)
        concatenate_b = np.full((concatenate_b, width), 255, dtype=np.uint8)
        concatImage = np.concatenate(
            (concatenate_a, img_original, concatenate_b), axis=0
        )

    return img_original, concatImage, y1, y2, x1, x2
    # cv2.imshow('concat', concatImage)
    # print(height, width)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # bw_image = cv2.bitwise_not(image)
    # cv2.imwrite(output_name, concatImage,
    #             [cv2.IMWRITE_PNG_COMPRESSION, 3])
    # cv2.imwrite(output_name, concatImage)
    # cv2.imwrite(output_name, concatImage,
    #             [cv2.IMWRITE_PNG_BILEVEL]) # USE THIS !!!
    # cv2.imwrite(output_name, concatImage,
    #             [cv2.IMWRITE_JPEG_QUALITY, 100])
    # cv2.imwrite(output_name, concatImage,
    #             [cv2.IMWRITE_PXM_BINARY, 1])
    # cv2.imshow('test', bw_image)
    # cv2.imshow('concat', concatImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
