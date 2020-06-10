from flask import jsonify, make_response
import time
from app import app
from flask import render_template, request, redirect, url_for
import os



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
    print(app.config["MARKER"])
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

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

            print("Image saved")

            return redirect(request.url)

    return render_template("public/upload_image.html")


from_sketch_button = False

@app.route('/sketch', methods=["GET", "POST"])
def sketch():
    global from_sketch_button
    global l
    global req
    global marker_type
    if request.method == "POST":
        l = 3
        req = {}
        res = request.get_json()
        print(res)
        # res = make_response(jsonify(req), 200)
        # print(type(res))
        # time.sleep(2)
        # return res
        from_sketch_button = True
        # req['add message from python']= 'This is the message'
        # res = make_response(jsonify(req), 200)
        # time.sleep(1)
        return redirect(request.url)
        # return redirect(url_for('sketch_'))
    # json_marker_type = jsonify(marker_type)
    # print(json_marker_type)
    print('outside if')

    return render_template('public/sketch.html', marker_type=marker_type)


# @app.route('/sketch1', methods=["GET", "POST"])
# def sketch():
#     global from_sketch_button
#     return redirect(url_for('sketch_'))

#     return render_template('public/sketch1.html')


first = False
second = False
third = False
end = False
req = {}
l = 5

# from_sketch_button = True
@app.route('/sketch_')
def sketch_():
    global from_sketch_button
    global first
    global second
    global end
    global l
    global req

    # from_sketch_button = True
    # for x in range(3):
    req['A message from python'] = 'Initialiation'
    if from_sketch_button:
        # resetting all global variable
        from_sketch_button = False
        first = False
        second = False
        third = False
        end = False
        req = {}
        l = 5
        first = True
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
    return render_template('public/sketch.html', marker_type=marker_type)

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
