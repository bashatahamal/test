# dumpPath = [pathof_crop_ratio_processing_result, pathof_imagelist_horizontal_line_by_eight_conn,
#         pathof_imagelist_template_matching_result, pathof_imagelist_template_scale_visualize,
#             pathof_imagelist_visualize_white_block, pathof_normal_processing_result,
#                 pathof_final_image_result]
import pickle
def display_result(filename):
    dumpPath = pickle.load(open(filename, 'rb'))
    thefile = open('/home/mhbrt/Desktop/Wind/Multiscale/templates/public/temp_result.html', 'r')
    body_file = thefile.read().split('<!--// modified //-->')
    # print(body_file[9])

    button = ''
    final_result = ''
    blok_process = ''
    js_lightgallery = ''
    js_showpage = ''
    for numfile in range(len(dumpPath[6])):
        # print(numfile)
        begin_file = ''
        sv_perfile = ''
        white_block = ''
        wb_perfile = ''
        template_matching = ''
        tm_perfile = ''
        h_line = ''
        hl_perfile = ''
        p_perfile_html = ''
        end_file = ''

        other = ''
        for x in range(len(dumpPath[6])):
            if x == numfile:
                continue
            other += "document.getElementById('file2"+str(x+1)+"').style.display = 'none'" + "\n" \
                + "document.getElementById('file4"+str(x+1)+"').style.display = 'none'" + "\n"\
                + "document.getElementById('file"+str(x+1)+"').className = 'btn btn-primary'" + "\n"
        js_showpage += "function show_file" + str(numfile+1) \
            +"(){document.getElementById('file2"+str(numfile+1)\
                +"').style.display = 'block'"+"\n"\
                +"document.getElementById('file"+str(numfile+1)\
                +"').className = 'btn btn-success'"+"\n"\
                +"document.getElementById('file4"+str(numfile+1)\
                +"').style.display = 'block'"+"\n"+other+"}" + "\n"

        js_lightgallery += "$('#result1"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result2"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result3"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result4"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result5"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result6"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result7"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result8"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result9"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result10"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result11"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result12"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result13"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result14"+ str(numfile+1) +"').lightGallery();" + "\n"\
                            +"$('#result15"+ str(numfile+1) +"').lightGallery();" + "\n"

        button += '<button class="btn btn-primary" id=file' + str(numfile+1) + ' onclick="show_file' + str(numfile+1) \
                + '()">file'+str(numfile+1)+'</button>' +'\n'
        final_result += '<div id="file2'+str(numfile+1)+'" style="display: none;">\
                <ul>\
                    <li>\
                        File'+str(numfile+1)+'\
                    </li>\
                </ul>\
                <div class="demo-gallery">\
                    <ul id=result1'+str(numfile+1)+'>\
                        <li style="display: block;"\
                            data-src="'+dumpPath[6][numfile]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>'+dumpPath[11][numfile]+'</p>">\
                            <a href="">\
                                <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                <div class="demo-gallery-poster">\
                                    <img src="static/light_gallery/img/zoom.png">\
                                </div>\
                            </a>\
                        </li>\
                    </ul>\
                </div>\
            </div>' + '\n'
        # box4
        input_image = '<div id="file4'+str(numfile+1)+'" style="display: none;">\
                <div class="demo-gallery">\
                    <ul id=result13'+str(numfile+1)+'>\
                        <li style="display: block;"\
                            data-src="'+dumpPath[8][numfile][0]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>Original Image</p>">\
                            <a href="">\
                                <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                <div class="demo-gallery-poster">\
                                    <img src="static/light_gallery/img/zoom.png">\
                                </div>\
                            </a>\
                            <h6>Input_Image</h6>\
                        </li>\
                        <li style="display: none;" data-src="'+dumpPath[8][numfile][1]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>Grayscale Image</p>">\
                            <a href="">\
                                <img class="img-responsive" src="'+dumpPath[8][numfile][1]+'" style="max-width: 150px;">\
                            </a>\
                        </li>\
                        <li style="display: none;" data-src="'+dumpPath[8][numfile][2]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>Binary Image</p>">\
                            <a href="">\
                                <img class="img-responsive" src="'+dumpPath[8][numfile][2]+'" style="max-width: 150px;">\
                            </a>\
                        </li>'
        # box4
        begin_file = ' </ul>\
                </div>\
                <div class="demo-gallery">\
                    <ul id=result2'+str(numfile+1)+'>\
                        <li style="display: block;"\
                            data-src="'+dumpPath[3][numfile][0]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                            <a href="">\
                                <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                <div class="demo-gallery-poster">\
                                    <img src="static/light_gallery/img/zoom.png">\
                                </div>\
                            </a>\
                            <h6>Scale_Visualize</h6>\
                        </li>'
        # box4
        sv_perfile = ''
        first = True
        for image in dumpPath[3][numfile]:
            if first:
                first = False
                continue
            sv_perfile += '<li style="display: none;" data-src="'+image+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                            <a href="">\
                                <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                            </a>\
                        </li>'
        # box4
        white_block = ' </ul>\
                </div>\
                    \
                <div class="demo-gallery">\
                    <ul id=result3'+str(numfile+1)+'>\
                        <li style="display: block;"\
                            data-src="'+dumpPath[4][numfile][0]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                            <a href="">\
                                <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                <div class="demo-gallery-poster">\
                                    <img src="static/light_gallery/img/zoom.png">\
                                </div>\
                            </a>\
                            <h6>White_Block</h6>\
                        </li>'
        # box4
        wb_perfile = ''
        first = True
        for image in dumpPath[4][numfile]:
            if first:
                first = False
                continue
            wb_perfile += '<li style="display: none;" data-src="'+image+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                            <a href="">\
                                <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                            </a>\
                        </li>'
        # box4
        font_name = ['AlKareem', 'AlQalam', 'KFGQPC', 'LPMQ', 'PDMS',
                    'amiri', 'meQuran', 'norehidayat', 'norehira', 'norehuda']
        template_matching = ' </ul>\
                </div>\
                    \
                <div class="demo-gallery">\
                    <ul id=result4'+str(numfile+1)+'>\
                        <li style="display: block;"\
                            data-src="'+dumpPath[2][numfile][0]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>'+font_name[0]+'</p>">\
                            <a href="">\
                                <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                <div class="demo-gallery-poster">\
                                    <img src="static/light_gallery/img/zoom.png">\
                                </div>\
                            </a>\
                            <h6>Template_Matching</h6>\
                        </li>'
        # box4
        tm_perfile = ''
        first = True
        count = 0
        for image in dumpPath[2][numfile]:
            if first:
                first = False
                continue
            count += 1
            tm_perfile += '<li style="display: none;" data-src="'+image+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>'+font_name[count]+'</p>">\
                            <a href="">\
                                <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                            </a>\
                        </li>'
        
        # box4
        h_line = ' </ul>\
                </div>\
                    \
                <div class="demo-gallery">\
                    <ul id=result5'+str(numfile+1)+'>\
                        <li style="display: block;"\
                            data-src="'+dumpPath[1][numfile][0]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                            <a href="">\
                                <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                <div class="demo-gallery-poster">\
                                    <img src="static/light_gallery/img/zoom.png">\
                                </div>\
                            </a>\
                            <h6>Horizontal_Line</h6>\
                        </li>'
        # box4
        hl_perfile = ''
        first = True
        for image in dumpPath[1][numfile]:
            if first:
                first = False
                continue
            hl_perfile += '<li style="display: none;" data-src="'+image+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                            <a href="">\
                                <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                            </a>\
                        </li>'
        # box4
        v_check = ' </ul>\
                </div>\
                    \
                <div class="demo-gallery">\
                    <ul id=result14'+str(numfile+1)+'>\
                        <li style="display: block;"\
                            data-src="'+dumpPath[9][numfile][0]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                            <a href="">\
                                <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                <div class="demo-gallery-poster">\
                                    <img src="static/light_gallery/img/zoom.png">\
                                </div>\
                            </a>\
                            <h6>Vertical_Line</h6>\
                        </li>'
        # box4
        vc_perfile = ''
        first = True
        for image in dumpPath[9][numfile]:
            if first:
                first = False
                continue
            vc_perfile += '<li style="display: none;" data-src="'+image+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                            <a href="">\
                                <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                            </a>\
                        </li>'

        if dumpPath[5][numfile] != []:
            np_name = ['Per_Char_Marker', 'Final_Word', 'Final_Segemented_Char',
                    'H_baseline', 'Final_Body', 'Final_Marker', 'H_Image']
            # box4
            p_perfile_list = []
            for x in range(len(dumpPath[5][numfile])):
                # box4
                p_perfile = ''
                if x == 2:
                    p_perfile = ' </ul>\
                            </div>\
                                \
                            <div class="demo-gallery">\
                                <ul id=result'+str(6+x)+str(numfile+1)+'>\
                                    <li style="display: block;"\
                                        data-src="'+dumpPath[5][numfile][x][0][0]+'"\
                                        data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>'+dumpPath[7][numfile][0]+'</p>">\
                                        <a href="">\
                                            <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                            <div class="demo-gallery-poster">\
                                                <img src="static/light_gallery/img/zoom.png">\
                                            </div>\
                                        </a>\
                                        <h6>'+np_name[x]+'</h6>\
                                    </li>'
                else:
                    p_perfile = ' </ul>\
                        </div>\
                            \
                        <div class="demo-gallery">\
                            <ul id=result'+str(6+x)+str(numfile+1)+'>\
                                <li style="display: block;"\
                                    data-src="'+dumpPath[5][numfile][x][0][0]+'"\
                                    data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                                    <a href="">\
                                        <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                        <div class="demo-gallery-poster">\
                                            <img src="static/light_gallery/img/zoom.png">\
                                        </div>\
                                    </a>\
                                    <h6>'+np_name[x]+'</h6>\
                                </li>'

                first = True
                count = 0
                for image in dumpPath[5][numfile][x][0]:
                    if first:
                        first = False
                        continue
                    count += 1
                    if x == 2:
                        p_perfile += '<li style="display: none;" data-src="'+image+'"\
                                    data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>'+dumpPath[7][numfile][count]+'</p>">\
                                    <a href="">\
                                        <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                                    </a>\
                                </li>'
                    else:
                        p_perfile += '<li style="display: none;" data-src="'+image+'"\
                                        data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                                        <a href="">\
                                            <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                                        </a>\
                                    </li>'
                p_perfile_list.append(p_perfile)
            p_perfile_html = p_perfile_list[6] + p_perfile_list[3] + p_perfile_list[4] + p_perfile_list[5] \
                            + p_perfile_list[1] + p_perfile_list[0] + p_perfile_list[2]
        else:
            # box4
            np_name = ['Gray_Image', 'Eight_Conn_on_Base', 'Substract_Image', 'Cutted_Substract', 'Final_Segmented_Char']
            p_perfile_list = []
            for x in range(len(dumpPath[0][numfile])):
                # box4
                p_perfile = ''
                if x == 4:
                    p_perfile = ' </ul>\
                            </div>\
                                \
                            <div class="demo-gallery">\
                                <ul id=result'+str(6+x)+str(numfile+1)+'>\
                                    <li style="display: block;"\
                                        data-src="'+dumpPath[0][numfile][x][0][0]+'"\
                                        data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>'+dumpPath[7][numfile][0]+'</p>">\
                                        <a href="">\
                                            <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                            <div class="demo-gallery-poster">\
                                                <img src="static/light_gallery/img/zoom.png">\
                                            </div>\
                                        </a>\
                                        <h6>'+np_name[x]+'</h6>\
                                    </li>'
                else:
                    p_perfile = ' </ul>\
                            </div>\
                                \
                            <div class="demo-gallery">\
                                <ul id=result'+str(6+x)+str(numfile+1)+'>\
                                    <li style="display: block;"\
                                        data-src="'+dumpPath[0][numfile][x][0][0]+'"\
                                        data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                                        <a href="">\
                                            <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                            <div class="demo-gallery-poster">\
                                                <img src="static/light_gallery/img/zoom.png">\
                                            </div>\
                                        </a>\
                                        <h6>'+np_name[x]+'</h6>\
                                    </li>'

                first = True
                count = 0
                for image in dumpPath[0][numfile][x][0]:
                    if first:
                        first = False
                        continue
                    count += 1
                    if x == 4:
                        p_perfile += '<li style="display: none;" data-src="'+image+'"\
                                        data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>'+dumpPath[7][numfile][count]+'</p>">\
                                        <a href="">\
                                            <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                                        </a>\
                                    </li>'
                    else:
                        p_perfile += '<li style="display: none;" data-src="'+image+'"\
                                        data-sub-html="<h3>File '+str(numfile+1)+'</h3>">\
                                        <a href="">\
                                            <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                                        </a>\
                                    </li>'
                p_perfile_list.append(p_perfile)
            p_perfile_html = p_perfile_list[0] + p_perfile_list[1] + p_perfile_list[2] \
                            + p_perfile_list[3] + p_perfile_list[4]
        
        # box4
        cr_pred = []
        for pred in dumpPath[7][numfile]:
            if pred != 'n/a':
                cr_pred.append(pred)
        char_recog = ' </ul>\
                </div>\
                    \
                <div class="demo-gallery">\
                    <ul id=result15'+str(numfile+1)+'>\
                        <li style="display: block;"\
                            data-src="'+dumpPath[10][numfile][0]+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>'+cr_pred[0]+'</p>">\
                            <a href="">\
                                <img class="img-responsive" src="/static/img/folder1.png" style="max-width: 150px;">\
                                <div class="demo-gallery-poster">\
                                    <img src="static/light_gallery/img/zoom.png">\
                                </div>\
                            </a>\
                            <h6>Char_Recognition</h6>\
                        </li>'
        # box4
        cr_perfile = ''
        first = True
        count = 0
        for image in dumpPath[10][numfile]:
            if first:
                first = False
                continue
            count += 1
            cr_perfile += '<li style="display: none;" data-src="'+image+'"\
                            data-sub-html="<h3>File '+str(numfile+1)+'</h3><p>'+cr_pred[count]+'</p>">\
                            <a href="">\
                                <img class="img-responsive" src="'+image+'" style="max-width: 150px;">\
                            </a>\
                        </li>'

        end_file = '</ul>\
                </div>\
            </div>' + '\n\n'
        
        blok_process += input_image + begin_file + sv_perfile + white_block + wb_perfile + template_matching \
                    + tm_perfile + h_line + hl_perfile + v_check + vc_perfile + p_perfile_html \
                        + char_recog + cr_perfile + end_file


    # print(len(dumpPath[3][2]))


    thefile = open('/home/mhbrt/Desktop/Wind/Multiscale/templates/public/test_result.html', 'w')
    thefile.write(body_file[0] +'<!--// modified //-->\n'+ button +'<!--// modified //-->'
                + body_file[2] +'<!--// modified //-->\n'+ final_result +'<!--// modified //-->'
                + body_file[4] +'<!--// modified //-->\n'+ blok_process +'<!--// modified //-->'
                + body_file[6] +'<!--// modified //-->\n'+ js_lightgallery +'<!--// modified //-->'
                + body_file[8] +'<!--// modified //-->\n'+ js_showpage +'<!--// modified //-->'
                + body_file[10])
