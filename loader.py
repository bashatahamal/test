import cv2
import os
import numpy as np


def load_image_path(loc_collection, shape):
    char_list_nameonly = [
        'Alif‬', 'Bā’', 'Tā’', 'Ṡā’‬', 'Jīm', 'h_Ḥā’‬', 'Khā’‬',
        'Dāl‬', 'Żāl‬', 'Rā’‬', 'zai‬', 'sīn‬', 'syīn‬', 's_ṣād',
        'd_ḍād', 't_ṭā’‬', 'z_ẓȧ’‬', '‘ain', 'gain‬', 'fā’‬', 'qāf‬',
        'kāf‬', 'lām‬', 'mīm‬', 'nūn‬', 'wāw‬', 'hā’‬', 'yā’‬'
    ]
    _lst_path_font = [os.path.join(loc_collection + shape, path)
                      for path in os.listdir(loc_collection + shape)]
    _lst_path_char = []

    _dict_path_char = {}
    for x in range(len(_lst_path_font)):
        for char_name in os.listdir(_lst_path_font[x]):
            _lst_temp = [os.path.join(_lst_path_font[x]+'/'+char_name, path)
                         for path in os.listdir(_lst_path_font[x]+'/'+char_name)]
            for char_num in range(len(char_list_nameonly)):
                if char_name == char_list_nameonly[char_num]:
                    if _dict_path_char.get(char_num) is None:
                        _dict_path_char[char_num] = []
                    for loc in _lst_temp:
                        split_loc = loc.split('.')
                        if split_loc[1] != 'png':
                            continue
                        # Modify here to append only png extension format
                        _dict_path_char[char_num].append(loc)

    # Check non image file
    all_images = True
    not_image = []
    for x in _dict_path_char:
        for y in _dict_path_char[x]:
            split = y.split('.')
            if split[1] != 'png':
                not_image.append(y)
                all_images = False
    if all_images:
        print('All files are .png')
    else:
        print(not_image)
    
    return _dict_path_char, all_images


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # resized = cv2.resize(image, dim, interpolation=inter)
    resized = cv2.resize(image, dim)

    return resized


def concat_image(img_original):
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

    return concatImage


def padding_square_to_size(image_square, size):
    height, width = image_square.shape
    if height < size:
        concatenate_a = int(round(size-height)/2)
        concatenate_b = size - height - concatenate_a
        concatenate_up = np.full((concatenate_a, width + concatenate_a + concatenate_b), 255, dtype=np.uint8)
        concatenate_down = np.full((concatenate_b, width + concatenate_a + concatenate_b), 255, dtype=np.uint8)
        concatenate_left = np.full((height, concatenate_a), 255, dtype=np.uint8)
        concatenate_right = np.full((height, concatenate_b), 255, dtype=np.uint8)
        concatImage = np.concatenate(
            (concatenate_left, image_square, concatenate_right), axis=1
        )
        concatImage = np.concatenate(
            (concatenate_up, concatImage, concatenate_down), axis=0
        )

        return concatImage
    else:
        return image_square


def strech_to_square(image):
    if len(image.shape) < 3:
        height, width = image.shape
    else:
        height, width, _ = image.shape
    if height > width:
        stsImage = cv2.resize(image, (height, height))
    else:
        stsImage = cv2.resize(image, (width, width))

    return stsImage


def main(loc_collection, shape):
#     loc_collection = 'Auto_Collection_Gray/'
#     shape = 'Square'
#     shape = 'No_Margin'
    _dict_path_char, all_images = load_image_path(loc_collection, shape)

    if all_images:
        smallest_height = 10000
        smallest_width = 10000
        largest_height = 0
        largest_width = 0
        smallest_img = 0
        largest_img = 0
        for char_num in _dict_path_char:
            for char_type in _dict_path_char[char_num]:
                img = cv2.imread(char_type)
                height, width, _ = img.shape
                if height < smallest_height:
                    smallest_height = height
                    smallest_img_by_height = img
                if width < smallest_width:
                    smallest_width = width
                    smallest_img_by_width = img
                if height > largest_height:
                    largest_height = height
                    largest_img_by_height = img
                if width > largest_width:
                    largest_width = width
                    largest_img_by_width = img
        height_sibh, width_sibh, _ = smallest_img_by_height.shape
        height_libh, width_libh, _ = largest_img_by_height.shape
        height_sibw, width_sibw, _ = smallest_img_by_width.shape
        height_libw, width_libw, _ = largest_img_by_width.shape
        if height_sibh < width_sibw:
            smallest_of_all = height_sibh
        else:
            smallest_of_all = width_sibw
        if height_libh > width_libw:
            largest_of_all = height_libh
        else:
            largest_of_all = width_libw
        ratio_to_sibh = float(height_sibh/height_libh)
        ratio_to_sibw = float(width_sibw/width_libw)
        ratio_to_32_bh = float(32/height_libh)
        ratio_to_32_bw = float(32/width_libw)
        print('sibh= ', height_sibh, width_sibh)
        print('sibw= ', height_sibw, width_sibw)
        print('libh= ', height_libh, width_libh)
        print('libw= ', height_libw, width_libw)
        print('ratio_to_sibh= ', ratio_to_sibh)
        print('ratio_to_sibw= ', ratio_to_sibw)
        print('ratio_to_32_bh= ', ratio_to_32_bh)
        print('ratio_to_32_bw= ', ratio_to_32_bw)
        # cv2.imshow('sibh', smallest_img_by_height)
        # cv2.imshow('sibw', smallest_img_by_width)
        # cv2.imshow('libh', largest_img_by_height)
        # cv2.imshow('libw', largest_img_by_width)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        '''
        Based on no margin:
    a. Rationya sama dan mempertahankan bentuk (resize lalu padding square)
    b. Rationya tidak sama dan mempertahankan bentuk (padding square lalu resize)
    c. Rationya sama dan tidak mempertahankan bentuk (strech square lalu padding)
    d. Rationya tidak sama dan tidak mempertahankan bentuk (langsung strech)
    e. Original no margin langsung dimasukin ke CNN (beda2 ukurannya)
    f. Original square langsung dimasukin ke CNN (beda2 ukurannya)        
        '''
        # a.
        sameratio_keepform_32_bh = {}
        sameratio_keepform_32_bw = {}
        sameratio_keepform_sibh = {}
        sameratio_keepform_sibw = {}
        sameratio_keepform_largest = {}
        # b.
        diffratio_keepform_32 = {}
        diffratio_keepform_sibh = {}
        diffratio_keepform_sibw = {}
        diffratio_keepform_libh = {}
        diffratio_keepform_libw = {}
        # c.
        sameratio_diffform_32 = {}
        sameratio_diffform_smallest = {}
        sameratio_diffform_largest = {}
        # d.
        diffratio_diffform_32 = {}
        diffratio_diffform_sibh = {}
        diffratio_diffform_sibw = {}
        diffratio_diffform_libh = {}
        diffratio_diffform_libw = {}

        add_sibh = True
        add_sibw = True
        for char_num in _dict_path_char:
            for char_type in _dict_path_char[char_num]:
                img = cv2.imread(char_type)
                # print(char_type)
                img_height, img_width, _ = img.shape
                if int(img_width*ratio_to_sibh) < 1:
                    add_sibh = False
                    # sameratio_keepform_sibh = {}
                if int(img_width*ratio_to_sibw) < 1:
                    add_sibw = False
                    # sameratio_keepform_sibw = {}

                if add_sibh:
                    image_sibh = cv2.resize(img,
                                            (int(img_width*ratio_to_sibh),
                                             int(img_height*ratio_to_sibh)))
                if add_sibw:
                    image_sibw = cv2.resize(img,
                                            (int(img_width*ratio_to_sibw),
                                             int(img_height*ratio_to_sibw)))
                image_32_bh = cv2.resize(img,
                                         (int(img_width*ratio_to_32_bh),
                                          int(img_height*ratio_to_32_bh)))
                image_32_bw = cv2.resize(img,
                                         (int(img_width*ratio_to_32_bw),
                                          int(img_height*ratio_to_32_bw)))
                # Original
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, bw_img = cv2.threshold(gray_img, 127, 255,
                                          cv2.THRESH_BINARY)
                # square_img = concat_image(bw_img)
                square_img = concat_image(gray_img)
                # Smallest Image by Height
                if add_sibh:
                    gray_img = cv2.cvtColor(image_sibh, cv2.COLOR_BGR2GRAY)
                    _, bw_img_sibh = cv2.threshold(gray_img, 127, 255,
                                                   cv2.THRESH_BINARY)
                    # square_img_sibh = concat_image(bw_img_sibh)
                    square_img_sibh = concat_image(gray_img)
                # Smallest Image by Width
                if add_sibw:
                    gray_img = cv2.cvtColor(image_sibw, cv2.COLOR_BGR2GRAY)
                    _, bw_img_sibw = cv2.threshold(gray_img, 127, 255,
                                                   cv2.THRESH_BINARY)
                    # square_img_sibw = concat_image(bw_img_sibw)
                    square_img_sibw = concat_image(gray_img)
                # 32 by Height
                gray_img = cv2.cvtColor(image_32_bh, cv2.COLOR_BGR2GRAY)
                _, bw_img_32_bh = cv2.threshold(gray_img, 127, 255,
                                                cv2.THRESH_BINARY)
                # square_img_32_bh = concat_image(bw_img_32_bh)
                square_img_32_bh = concat_image(gray_img)
                # 32 by Width
                gray_img = cv2.cvtColor(image_32_bw, cv2.COLOR_BGR2GRAY)
                _, bw_img_32_bw = cv2.threshold(gray_img, 127, 255,
                                                cv2.THRESH_BINARY)
                # square_img_32_bw = concat_image(bw_img_32_bw)
                square_img_32_bw = concat_image(gray_img)
                # a.
                if sameratio_keepform_32_bh.get(char_num) is None:
                    sameratio_keepform_32_bh[char_num] = []
                if sameratio_keepform_32_bw.get(char_num) is None:
                    sameratio_keepform_32_bw[char_num] = []
                if sameratio_keepform_sibh.get(char_num) is None:
                    sameratio_keepform_sibh[char_num] = []
                if sameratio_keepform_sibw.get(char_num) is None:
                    sameratio_keepform_sibw[char_num] = []
                if sameratio_keepform_largest.get(char_num) is None:
                    sameratio_keepform_largest[char_num] = []
                sameratio_keepform_32_bh[char_num].append(
                    padding_square_to_size(square_img_32_bh, 32)
                )
                sameratio_keepform_32_bw[char_num].append(
                    padding_square_to_size(square_img_32_bw, 32)
                )
                if add_sibh:
                    sameratio_keepform_sibh[char_num].append(
                        padding_square_to_size(square_img_sibh, height_sibh)
                    )
                else:
                    sameratio_keepform_sibh = {}
                if add_sibw:
                    sameratio_keepform_sibw[char_num].append(
                        padding_square_to_size(square_img_sibw, width_sibw)
                    )
                else:
                    sameratio_keepform_sibw = {}
                sameratio_keepform_largest[char_num].append(
                    padding_square_to_size(square_img, largest_of_all)
                )

                # b.
                image_sibh_dk = cv2.resize(square_img, (height_sibh, height_sibh))
                image_sibw_dk = cv2.resize(square_img, (width_sibw, width_sibw))
                image_32_dk = cv2.resize(square_img, (32, 32))
                image_libh_dk = cv2.resize(square_img, (height_libh, height_libh))
                image_libw_dk = cv2.resize(square_img, (width_libw, width_libw))
                if diffratio_keepform_32.get(char_num) is None:
                    diffratio_keepform_32[char_num] = []
                if diffratio_keepform_sibh.get(char_num) is None:
                    diffratio_keepform_sibh[char_num] = []
                if diffratio_keepform_sibw.get(char_num) is None:
                    diffratio_keepform_sibw[char_num] = []
                if diffratio_keepform_libh.get(char_num) is None:
                    diffratio_keepform_libh[char_num] = []
                if diffratio_keepform_libw.get(char_num) is None:
                    diffratio_keepform_libw[char_num] = []
                diffratio_keepform_32[char_num].append(image_32_dk)
                diffratio_keepform_sibh[char_num].append(image_sibh_dk)
                diffratio_keepform_sibw[char_num].append(image_sibw_dk)
                diffratio_keepform_libh[char_num].append(image_libh_dk)
                diffratio_keepform_libw[char_num].append(image_libw_dk)

                # c.
                image_sts = strech_to_square(img)
                sts_height, sts_width, _ = image_sts.shape
                if add_sibh or add_sibw:
                    image_smallest_sd = cv2.resize(
                        image_sts,
                        (int(sts_width*smallest_of_all/largest_of_all),
                         int(sts_width*smallest_of_all/largest_of_all))
                    )
                image_32_sd = cv2.resize(image_sts,
                                         (int(img_width*32/largest_of_all),
                                          int(img_width*32/largest_of_all)))

                # Original/Largest
                gray_img = cv2.cvtColor(image_sts, cv2.COLOR_BGR2GRAY)
                _, bw_img_sts = cv2.threshold(gray_img, 127, 255,
                                              cv2.THRESH_BINARY)
                # image_largest_sd = padding_square_to_size(bw_img_sts,
                #                                           largest_of_all)
                image_largest_sd = padding_square_to_size(gray_img,
                                                          largest_of_all)

                # Smallest Image
                if add_sibh or add_sibw:
                    gray_img = cv2.cvtColor(image_smallest_sd, cv2.COLOR_BGR2GRAY)
                    _, bw_img_sts_smallest = cv2.threshold(gray_img, 127, 255,
                                                           cv2.THRESH_BINARY)
                    # image_smallest_sd = padding_square_to_size(bw_img_sts_smallest,
                    #                                            smallest_of_all)
                    image_smallest_sd = padding_square_to_size(gray_img,
                                                               smallest_of_all)
                # 32 Image
                gray_img = cv2.cvtColor(image_32_sd, cv2.COLOR_BGR2GRAY)
                _, bw_img_sts_32 = cv2.threshold(gray_img, 127, 255,
                                                 cv2.THRESH_BINARY)
                # image_32_sd = padding_square_to_size(bw_img_sts_32, 32)
                image_32_sd = padding_square_to_size(gray_img, 32)
                if sameratio_diffform_32.get(char_num) is None:
                    sameratio_diffform_32[char_num] = []
                if sameratio_diffform_smallest.get(char_num) is None:
                    sameratio_diffform_smallest[char_num] = []
                if sameratio_diffform_largest.get(char_num) is None:
                    sameratio_diffform_largest[char_num] = []
                sameratio_diffform_32[char_num].append(image_32_sd)
                sameratio_diffform_smallest[char_num].append(image_smallest_sd)
                sameratio_diffform_largest[char_num].append(image_largest_sd)

                # d.
                image_32_dd = cv2.resize(img, (32, 32))
                image_sibh_dd = cv2.resize(img, (height_sibh, height_sibh))
                image_sibw_dd = cv2.resize(img, (width_sibw, width_sibw))
                image_libh_dd = cv2.resize(img, (height_libh, height_libh))
                image_libw_dd = cv2.resize(img, (width_libw, width_libw))
                gray_img = cv2.cvtColor(image_32_dd, cv2.COLOR_BGR2GRAY)
                # _, bw_img_32_dd = cv2.threshold(gray_img, 127, 255,
                #                                 cv2.THRESH_BINARY)
                bw_img_32_dd = gray_img
                gray_img = cv2.cvtColor(image_sibh_dd, cv2.COLOR_BGR2GRAY)
                # _, bw_img_sibh_dd = cv2.threshold(gray_img, 127, 255,
                #                                   cv2.THRESH_BINARY)
                bw_img_sibh_dd = gray_img
                gray_img = cv2.cvtColor(image_sibw_dd, cv2.COLOR_BGR2GRAY)
                # _, bw_img_sibw_dd = cv2.threshold(gray_img, 127, 255,
                #                                   cv2.THRESH_BINARY)
                bw_img_sibw_dd = gray_img
                gray_img = cv2.cvtColor(image_libh_dd, cv2.COLOR_BGR2GRAY)
                # _, bw_img_libh_dd = cv2.threshold(gray_img, 127, 255,
                #                                   cv2.THRESH_BINARY)
                bw_img_libh_dd = gray_img
                gray_img = cv2.cvtColor(image_libw_dd, cv2.COLOR_BGR2GRAY)
                # _, bw_img_libw_dd = cv2.threshold(gray_img, 127, 255,
                #                                   cv2.THRESH_BINARY)
                bw_img_libw_dd = gray_img
                if diffratio_diffform_32.get(char_num) is None:
                    diffratio_diffform_32[char_num] = []
                if diffratio_diffform_sibh.get(char_num) is None:
                    diffratio_diffform_sibh[char_num] = []
                if diffratio_diffform_sibw.get(char_num) is None:
                    diffratio_diffform_sibw[char_num] = []
                if diffratio_diffform_libh.get(char_num) is None:
                    diffratio_diffform_libh[char_num] = []
                if diffratio_diffform_libw.get(char_num) is None:
                    diffratio_diffform_libw[char_num] = []
                diffratio_diffform_32[char_num].append(bw_img_32_dd)
                diffratio_diffform_sibh[char_num].append(bw_img_sibh_dd)
                diffratio_diffform_sibw[char_num].append(bw_img_sibw_dd)
                diffratio_diffform_libh[char_num].append(bw_img_libh_dd)
                diffratio_diffform_libw[char_num].append(bw_img_libw_dd)
        sk_val = [sameratio_keepform_32_bh, sameratio_keepform_32_bw, sameratio_keepform_sibh, sameratio_keepform_sibw, sameratio_keepform_largest]
        dk_val = [diffratio_keepform_32, diffratio_keepform_sibh, diffratio_keepform_sibw, diffratio_keepform_libh, diffratio_keepform_libw]
        sd_val = [sameratio_diffform_32, sameratio_diffform_smallest, sameratio_diffform_largest]
        dd_val = [diffratio_diffform_32, diffratio_diffform_sibh, diffratio_diffform_sibw, diffratio_diffform_libh, diffratio_diffform_libw]
#     sk = ['sameratio_keepform_32_bh', 'sameratio_keepform_32_bw', 'sameratio_keepform_sibh', 'sameratio_keepform_sibw', 'sameratio_keepform_largest']
#     dk = ['diffratio_keepform_32', 'diffratio_keepform_sibh', 'diffratio_keepform_sibw', 'diffratio_keepform_libh', 'diffratio_keepform_libw']
#     sd = ['sameratio_diffform_32', 'sameratio_diffform_smallest', 'sameratio_diffform_largest']
#     dd = ['diffratio_diffform_32', 'diffratio_diffform_sibh', 'diffratio_diffform_sibw', 'diffratio_diffform_libh', 'diffratio_diffform_libw']
    sk = ['sk_32_bh', 'sk_32_bw', 'sk_sibh', 'sk_sibw', 'sk_largest']
    dk = ['dk_32', 'dk_sibh', 'dk_sibw', 'dk_libh', 'dk_libw']
    sd = ['sd_32', 'sd_smallest', 'sd_largest']
    dd = ['dd_32', 'dd_sibh', 'dd_sibw', 'dd_libh', 'dd_libw']
    sameratio_keepform = {}
    diffratio_keepform = {}
    sameratio_diffform = {}
    diffratio_diffform = {}
    for x in range(len(sk)):
        sameratio_keepform[sk[x]] = sk_val[x]
    for x in range(len(dk)):
        diffratio_keepform[dk[x]] = dk_val[x]
    for x in range(len(sd)):
        sameratio_diffform[sd[x]] = sd_val[x]
    for x in range(len(dd)):
        diffratio_diffform[dd[x]] = dd_val[x]
    
    return sameratio_keepform, diffratio_keepform, sameratio_diffform, diffratio_diffform

if __name__ == '__main__':
    main()
#     for char_num in sameratio_keepform_32_bh:
#         for variation in range(len(sameratio_keepform_32_bh[char_num])):
#             # cv2.imshow('sk32bh', sameratio_keepform_32_bh[char_num][variation])
#             # cv2.imshow('sk32bw', sameratio_keepform_32_bw[char_num][variation])
#             # cv2.imshow('sksibh', sameratio_keepform_sibh[char_num][variation])
#             # cv2.imshow('sksibw', sameratio_keepform_sibw[char_num][variation])
#             # cv2.imshow('sklibh', sameratio_keepform_largest[char_num][variation])

#             cv2.imshow('dk32', diffratio_keepform_32[char_num][variation])
#             cv2.imshow('dksibh', diffratio_keepform_sibh[char_num][variation])
#             cv2.imshow('dksibw', diffratio_keepform_sibw[char_num][variation])
#             cv2.imshow('dklibh', diffratio_keepform_libh[char_num][variation])
#             cv2.imshow('dklibw', diffratio_keepform_libw[char_num][variation])

#             # cv2.imshow('sd32', sameratio_diffform_32[char_num][variation])
#             # cv2.imshow('sdst', sameratio_diffform_smallest[char_num][variation])
#             # cv2.imshow('sdlt', sameratio_diffform_largest[char_num][variation])

#             # cv2.imshow('dd32', diffratio_diffform_32[char_num][variation])
#             # cv2.imshow('ddsibh', diffratio_diffform_sibh[char_num][variation])
#             # cv2.imshow('ddsibw', diffratio_diffform_sibw[char_num][variation])
#             # cv2.imshow('ddlibh', diffratio_diffform_libh[char_num][variation])
#             # cv2.imshow('ddlibw', diffratio_diffform_libw[char_num][variation])
#             cv2.waitKey(0)
#     cv2.destroyAllWindows()
