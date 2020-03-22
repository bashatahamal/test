import match_oop as mo
import cv2
# import glob
import copy


def eight_conn(image):
    height, width = image.shape
    process = mo.ImageProcessing(original_image=image)
    conn_pack = {}
    reg = 1
    connected = True
    count = 0

    # Doing eight conn on every pixel one by one
    for x in range(width):
        for y in range(height):
            if image[y, x] == 0:
                count += 1
                # conn_pack['region_' + reg].add((x,y))
                x_y = []
                # Left
                if x - 1 > 0:
                    if y + 1 < height:
                        if image[y + 1, x - 1] == 0:
                            x_y.append((y + 1, x - 1))
                            # print('l1')
                    if image[y, x - 1] == 0:
                        x_y.append((y, x - 1))
                        # print('l2')
                    if y - 1 > 0:
                        if image[y - 1, x - 1] == 0:
                            x_y.append((y - 1, x - 1))
                            # print('l3')
                # Middle
                if y + 1 < height:
                    if image[y + 1, x] == 0:
                        x_y.append((y + 1, x))
                    # print('m1')
                x_y.append((y, x))
                # print('m2')
                if y - 1 > 0:
                    if image[y - 1, x] == 0:
                        x_y.append((y - 1, x))
                        # print('m3')
                # Right
                if x + 1 < width:
                    if y + 1 < height:
                        if image[y + 1, x + 1] == 0:
                            x_y.append((y + 1, x + 1))
                            # print('r1')
                    if image[y, x + 1] == 0:
                        x_y.append((y, x + 1))
                        # print('r2')
                    if y - 1 > 0:
                        if image[y - 1, x + 1] == 0:
                            x_y.append((y - 1, x + 1))
                            # print('r3')

                # First region only (Inizialitation)
                if conn_pack == {}:
                    conn_pack['region_1'] = []
                    for x1_join in x_y:
                        if x1_join not in conn_pack['region_1']:
                            conn_pack['region_1'].append(x1_join)
                    # print('inisialitation')

                # Next step is here
                connected = False
                connected_list = []
                if conn_pack != {}:
                    # Check how many region is connected
                    # with detected eight neighbour
                    for x_list in x_y:
                        # if connected:
                        #     break
                        for r in conn_pack.keys():
                            # r += 1
                            # if connected:
                            #     break
                            for val in conn_pack[r]:
                                # if connected:
                                #     break
                                if x_list == val:
                                    if r not in connected_list:
                                        connected_list.append(r)
                                    connected = True
                                    # break
                    # Append eight conn to first detected region
                    if connected_list != []:
                        for x_join in x_y:
                            if x_join not in conn_pack[
                                    connected_list[0]]:
                                conn_pack[connected_list[0]].append(x_join)
                    # print('connected list={}'.format(connected_list))
                    # cv2.waitKey(0)

                    # If eight conn is overlapped (in more than 1 region)
                    # then join every next region to first detected region
                    # and delete who join
                    if len(connected_list) > 1:
                        for c_list in range(len(connected_list) - 1):
                            c_list += 1
                            for x_join in conn_pack[
                                    connected_list[c_list]]:
                                if x_join not in conn_pack[
                                        connected_list[0]]:
                                    conn_pack[
                                        connected_list[0]
                                    ].append(x_join)
                        for c_list in range(len(connected_list) - 1):
                            c_list += 1
                            print('delete {}'.format(
                                connected_list[c_list]
                            ))
                            del(conn_pack[connected_list[c_list]])
                            # print(connected_list[c_list])

                    # if not connected then just create a new region
                    if not connected:
                        reg += 1
                        conn_pack['region_' + str(reg)] = []
                        for x2_join in x_y:
                            if x2_join not in conn_pack['region_' + str(reg)]:
                                conn_pack['region_' + str(reg)].append(x2_join)
    # Get every region length
    temp_max = 0
    for key in conn_pack:
        if len(conn_pack[key]) > temp_max:
            temp_max = len(conn_pack[key])
            key_max = key
    conn_pack_minus_body = copy.deepcopy(conn_pack)
    # The longest region it's the body
    for key in conn_pack_minus_body:
        if key == key_max:
            continue
        else:
            del(conn_pack[key])
    # Get body only and minus body region
    del(conn_pack_minus_body[key_max])
    # Paint body only region
    image_body = image.copy()
    image_body[:] = 255
    for region in conn_pack:
        value = conn_pack[region]
        for x in value:
            image_body[x] = 0

    # cv2.imshow('body', image_body)
    # cv2.waitKey(0)

    image_marker_sum = image.copy()
    image_marker_sum[:] = 255
    print(len(conn_pack_minus_body))
    for region in conn_pack_minus_body:
        value = conn_pack_minus_body[region]
        print(value)
        image_marker = image.copy()
        image_marker[:] = 255
        pixel_count = 0
        for x in value:
            pixel_count += 1
            image_marker[x] = 0
            image_marker_sum[x] = 0
        # cv2.imshow('marker only', image_marker_sum)
        # cv2.waitKey(0)

        process.horizontal_projection(image_marker)
        process.detect_horizontal_line(image_marker.copy(), 0, 0, False)
        one_marker = image_marker[process.start_point_h[0]:
                                  process.start_point_h[1] + 1, :]
        # cv2.imshow('fin', one_marker)
        # cv2.waitKey(0)
        process.vertical_projection(one_marker)
        process.detect_vertical_line(one_marker.copy(), 0, False)
        x1 = process.start_point_v[0]
        x2 = process.start_point_v[1]
        one_marker = one_marker[:, x1:x2+1]
        # print(one_marker)
        # cv2.imshow('fin', one_marker)
        # cv2.waitKey(0)
        # cv2.imshow('marker only', image_marker)
        # cv2.waitKey(0)
        process.horizontal_projection(one_marker)
        process.vertical_projection(one_marker)
        img_h_v_proj = process.v_projection
        img_h_h_proj = process.h_projection

        # Looking for square skeleton
        count_v = 0
        max_v = 0
        for v_sum in img_h_v_proj:
            if v_sum > max_v:
                max_v = v_sum
                max_ord_v = count_v
            count_v += 1
        count_h = 0
        max_h = 0
        for h_sum in img_h_h_proj:
            if h_sum > max_h:
                max_h = h_sum
                max_ord_h = count_h
            count_h += 1
        # start_x, end_x, start_y, end_y skeleton
        height, width = one_marker.shape
        for x_ in range(width):
            if one_marker[max_ord_h, x_] == 0:
                start_x = x_
                break
        end_x = start_x + int(max_h)
        for y_ in range(height):
            if one_marker[y_, max_ord_v] == 0:
                start_y = y_
                break
        end_y = start_y + int(max_v)
        # x1 = start_x, y1 = start_y
        # x2 = end_x, y2 = end_y
        print(start_x, end_x)
        print(start_y, end_y)
        print(max_h, max_v)
        print(max_ord_h, max_ord_v)
        if max_ord_v in range(start_x, end_x):
            squareleton = one_marker[start_y:end_y, start_x:end_x]
            # cv2.imshow('squareleton', squareleton)
            # print(squareleton)
            height, width = squareleton.shape
            scale = 1.55
            if width < scale * height:
                if height < scale * width:
                    print('square')
                    if pixel_count < height * width:
                        for x in range(width):
                            black = False
                            white = False
                            false_dot = False
                            white_val = 0
                            for y in range(height):
                                if squareleton[y, x] == 0:
                                    black = True
                                if black and squareleton[y, x] > 0:
                                    if not white:
                                        white_val = 0
                                    white = True
                                if black and white and squareleton[y, x] == 0:
                                    false_dot = True
                                    print('white hole')
                                    break
                                # If to many whites is not a dot
                                if squareleton[y, x] > 0:
                                    white_val += 1
                                if white_val > round(height/2):
                                    print('to many white')
                                    false_dot = True
                                    break
                            if false_dot:
                                print('NOT a dot')
                                break
                        if not false_dot:
                            print('Its a dot :)')
                            for x in value:
                                image_body[x] = 0
                            #     cv2.imshow('body', image_body)
                            # cv2.waitKey(0)
                    else:
                        print('The square is not enough')
                        print('NOT a dot')
                else:
                    print('portrait image')
                    print('NOT a dot')
            else:
                print('landscape')
                x_l = round(width/2)
                white_l = False
                black_l = False
                dot = False
                for y_l in range(height):
                    if squareleton[y_l, x_l] > 0:
                        white_l = True
                    if white_l and squareleton[y_l, x_l] == 0:
                        black_l = True
                    if white_l and black_l and squareleton[y_l, x_l] > 0:
                        dot = True
                if dot:
                    print('Middle is w -> b -> w')
                    print('Its a dot :)')
                    for x in value:
                        image_body[x] = 0
                else:
                    print('middle is wrong')
                    print('NOT a dot')
            #cv2.waitKey(0)
        else:
            print('Just cannot create a square')
            print('NOT a dot')
            continue

    return image_body


# for imagePath in sorted(glob.glob("test/dot" + "/*.png")):
#     image = cv2.imread(imagePath)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     bin_image = cv2.adaptiveThreshold(gray, 255,
#                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                       cv2.THRESH_BINARY, 11, 2)
#     height, width = gray.shape
#     cv2.imshow('sda', eight_conn(bin_image))
#     cv2.waitKey(0)
    # print('height {} width {}'.format(height, width))
    # print(type(gray))
    # process.vertical_projection(gray)
    # process.horizontal_projection(gray)
    # # plt.figure(0)
    # # plt.subplot(311), plt.imshow(gray)
    # # plt.subplot(312), plt.plot(
    # #     np.arange(0, len(process.v_projection), 1), process.v_projection
    # # )
    # # plt.subplot(313), plt.plot(
    # #     np.arange(0, len(process.h_projection), 1), process.h_projection
    # # )
    # # plt.show()
    # img_v = {}
    # process.detect_vertical_line(gray.copy(), 0)
    # print(process.start_point_v)
    # height, width = gray.shape
    # print('height {} width {}'.format(height, width))
    # for x in range(len(process.start_point_v)):
    #     if x % 2 == 0:
    #         print(x)
    #         img_v[x] = gray[:, process.start_point_v[x]:
    #                         process.start_point_v[x+1] + 1]

    # img_h = {}
    # for x in img_v:
    #     process.horizontal_projection(img_v[x])
    #     process.detect_horizontal_line(img_v[x].copy(), 0, 0)
    #     print(process.start_point_h)
    #     for y in range(len(process.start_point_h)):
    #         # Get individual marker
    #         if y % 2 == 0:
    #             img_h[y] = img_v[x][process.start_point_h[y]:
    #                                 process.start_point_h[y+1], :]
    #             print(img_h[y])
    #             cv2.imshow('ss', img_h[y])
    #             cv2.waitKey(0)

    #             process.vertical_projection(img_h[y])
    #             process.detect_vertical_line(img_h[y].copy(), 0)
    #             x1 = process.start_point_v[0]
    #             x2 = process.start_point_v[1]
    #             print(x1, x2)
    #             marker_box_img = img_h[y][:, x1:x2]
    #             print(marker_box_img)
    #             cv2.imshow('fin', marker_box_img)
    #             cv2.waitKey(0)

    #             process.horizontal_projection(marker_box_img)
    #             process.vertical_projection(marker_box_img)
    #             img_h_v_proj = process.v_projection
    #             img_h_h_proj = process.h_projection

    #             # Looking for square skeleton
    #             count_v = 0
    #             max_v = 0
    #             for v_sum in img_h_v_proj:
    #                 if v_sum > max_v:
    #                     max_v = v_sum
    #                     max_ord_v = count_v
    #                 count_v += 1
    #             count_h = 0
    #             max_h = 0
    #             for h_sum in img_h_h_proj:
    #                 if h_sum > max_h:
    #                     max_h = h_sum
    #                     max_ord_h = count_h
    #                 count_h += 1
    #             # start_x, end_x, start_y, end_y skeleton
    #             height, width = marker_box_img.shape
    #             tracing = False
    #             for x_ in range(width):
    #                 if marker_box_img[max_ord_h, x_] == 0:
    #                     start_x = x_
    #                     break
    #             end_x = start_x + int(max_h)
    #             tracing = False
    #             for y_ in range(height):
    #                 if marker_box_img[y_, max_ord_v] == 0:
    #                     start_y = y_
    #                     break
    #             end_y = start_y + int(max_v)
    #             # x1 = start_x, y1 = start_y
    #             # x2 = end_x, y2 = end_y
    #             print(start_x, end_x)
    #             print(start_y, end_y)
    #             print(max_h, max_v)
    #             print(max_ord_h, max_ord_v)
    #             if max_ord_v in range(start_x, end_x):
    #                 squareleton = marker_box_img[start_y:end_y, start_x:end_x]
    #                 cv2.imshow('squareleton', squareleton)
    #                 print(squareleton)
    #                 cv2.waitKey(0)
    #             else:
    #                 print('Just cannot create a square')
    #                 continue
