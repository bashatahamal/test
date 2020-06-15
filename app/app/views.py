from flask import Response
import subprocess
import pickle
from flask import jsonify, make_response
import time
from app import app
from flask import render_template, request, redirect, url_for
import os
import cv2
import sys
sys.path.insert(1, '/home/mhbrt/Desktop/Wind/Multiscale/')
import complete_flow as flow

# model_name = '/home/mhbrt/Desktop/Wind/Multiscale/Colab/best_model_DenseNet_DD.pkl'
# model = pickle.load(open(model_name, 'rb'))
# print('_LOAD MODEL DONE_')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
def index():
    return render_template('public/index.html')

# @app.route('/about')
# def about():
#     return "<h1 style = 'color: red'> ABOUT!! </h1>"


@app.route('/stream')
def yieldd():
    def inner():
        proc = subprocess.Popen(
            # call something with a lot of output so we can see it
            ["python", "-u", "count_timer.py"],
            stdout=subprocess.PIPE,
            # universal_newlines=True
        )

        for line in iter(proc.stdout.readline, b''):
        # print(line.decode("utf-8"))
            # yield line.decode("utf-8").rstrip() + '<br/>\n'
            yield line.decode("utf-8").rstrip() + '$'

    # text/html is required for most browsers to show th$
    return Response(inner(), mimetype='text/event-stream')
    # return Response(inner(), mimetype='text/html')

import flask
@app.route('/page')
def get_page():
    return flask.send_file('templates/public/page.html')


@app.route("/guestbook/create-entry", methods=["POST"])
def create_entry():

    req = request.get_json()

    print(req)
    print(request.url)

    # res = make_response(jsonify({"message": "OK"}), 200)
    res = make_response(jsonify(req), 200)

    return res
    # return render_template('public/jinja.html')


@app.route('/jinja', methods=["GET", "POST"])
def jinja():

    req = request.get_json()
    # print(app.config["MARKER"])
    print('hhh', req)

    test_list = ['2323', 'fdfd', '23123']
    if request.method == "POST":
        req = request.form
        # username = request.form.get("username")
        # email = request.form.get("email")
        # password = request.form.get("password")

        # # Alternatively

        # username = request.form["username"]
        # email = request.form["email"]
        # password = request.form["password"]

        print(req)
        print(request.url)

        return redirect(request.url)

    return render_template('public/jinja.html', test_list=test_list)


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            image.save(os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename))

            print("Image saved")

            return redirect(request.url)

    return render_template("public/upload_image.html")


from_sketch_button = False

list_image_files = []

@app.route('/dataset')
def dataset():
    font_folder = ['AlKareem', 'AlQalam', 'KFGQPC', 'LPMQ', 'PDMS',
                   'amiri', 'meQuran', 'norehidayat', 'norehira', 'norehuda']
    # print(app.config['MARKER_FOLDER'][font_folder[1]])
    return render_template('public/dataset.html', marker_folder=app.config['MARKER_FOLDER'])

@app.route('/sketch', methods=["GET", "POST"])
def sketch():
    global from_sketch_button
    global req
    global list_image_files
    if request.method == "POST":
        if request.is_json:
            req = {}
            res = request.get_json()
            print(res)
            from_sketch_button = True

            if res == 'start':
                response = make_response(jsonify('processing'), 200)
                return response

            imagePath = '/home/mhbrt/Desktop/Wind/Multiscale/temp/0v4.jpg'
            markerPath = '/home/mhbrt/Desktop/Wind/Multiscale/marker'
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # font_list = mess.font(imagePath=imagePath, image=gray, setting=res)
            font_list = flow.font_list(
                imagePath=imagePath, image=gray, setting=res, markerPath=markerPath)
            temp_object = []
            # print(font_list)
            # print(font_list[1].marker_location)
            for font_object in font_list:
                font_object.run()
                temp_object.append(font_object.get_object_result())
            print(temp_object)
            flow.big_blok(temp_object, imagePath,
                          font_object, model, font_list)
            response = make_response(jsonify('ok'), 200)
            return response
            # return redirect(request.url)

        if request.files:
            # files = request.files.getlist("files")
            # print(files)

            image = request.files["image"]
            print(image)
            saved_path = os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename)
            list_image_files.append(saved_path)
            image.save(saved_path)
            print("Image saved")
            res = make_response(jsonify(saved_path), 200)
            # return render_template('public/sketch.html', marker_type=app.config['MARKER_TYPE'])
            # return redirect(request.url)
            return res

    print('outside if')

    return render_template('public/sketch.html', marker_type=app.config['MARKER_TYPE'])


# @app.route('/sketch1', methods=["GET", "POST"])
# def sketch():
#     global from_sketch_button
#     return redirect(url_for('sketch_'))

#     return render_template('public/sketch1.html')


req = {}
# from_sketch_button = True


@app.route('/sketch_')
def sketch_():
    global from_sketch_button
    global req

    # from_sketch_button = True
    # for x in range(3):
    req['A message from python'] = 'Initialiation'
    if from_sketch_button:
        # resetting all global variable
        from_sketch_button = True
        req = {}
        req['A message from python'] = 'Doing the number1'
        print('button________________')
        # res = make_response(jsonify(req), 200)
        # time.sleep(6)
        print(request.url)
        return render_template('public/sketch_.html', next='/number1', req=req)

    # return render_template('public/sketch.html')
    return redirect('/sketch')


@app.route('/number1')
def do_something():
    print(from_sketch_button)
    time.sleep(5)
    req['A message from python'] = 'number 1 done and prepare for number 2'
    return render_template('public/sketch_.html', next='/number2', req=req)


@app.route('/number2')
def do_something_again():
    print(from_sketch_button)
    time.sleep(4)
    req['A message from python'] = 'number 2 done and prepare for number 3'
    return render_template('public/sketch_.html', next='/number3', req=req)


@app.route('/number3')
def do_something_and_again():
    print(from_sketch_button)
    time.sleep(4)
    req['A message from python'] = 'number 3 done and prepare for number 4'
    return render_template('public/sketch_.html', next='/number4', req=req)


@app.route('/number4')
def do_something_and_again_and_again_and_again():
    print(from_sketch_button)
    time.sleep(4)
    req['A message from python'] = 'number 4 done and prepare for number 5'
    return render_template('public/sketch_.html', next='/result', req=req)


@app.route('/result')
def do_something_and_again_final():
    print(from_sketch_button)
    time.sleep(3)
    req['A message from python'] = 'number 4 done and back to sketch'
    # return render_template('public/sketch_.html', next='/sketch', req=req)
    return render_template('public/sketch.html', marker_type=app.config['MARKER_TYPE'])

# @app.route('/test')
# def test():
#     test = 'THIS IS JINJA'
#     return render_template('public/test.html', test=test)

# @app.route('/sketch_')
# def sketch_():
#     global from_sketch_button
#     global first
#     global second
#     global end
#     global l
#     global req
#     from_sketch_button = True
#     # for x in range(3):
#     req['A message from python'] = 'Initialiation'
#     if from_sketch_button:
#         from_sketch_button = False
#         first = True
#         req['A message from python'] = 'This is the message'
#         # req[x]=x
#         print('button________________')
#         res = make_response(jsonify(req), 200)
#         # time.sleep(1)
#         print(request.url)
#         # return render_template('public/sketch_.html', req=req)
#         return res
#         # return redirect(request.url)
#     if first:
#         first = False
#         second = True
#         print('first________________________________')
#         req['A message from python'] = 'This is comes from the first section'
#         req[l] = l
#         res = make_response(jsonify(req), 200)
#         time.sleep(1)
#         # return render_template('public/sketch_.html', req=req)
#         return res
#     if second:
#         second = False
#         end = True
#         print('second_________________')
#         req['A message from python'] = 'This is comes from the second section'
#         req[l] = l
#         time.sleep(1)
#         res = make_response(jsonify(req), 200)
#         # return render_template('public/sketch_.html', req=req)
#         return res
#     # if end:
#     #     end = False
#     #     if l > 0:
#     #         l -= 1
#     #         first = True
#     #         time.sleep(1)
#     #         return render_template('public/sketch_.html', req=req)
#     return render_template('public/sketch_.html', req=req)
