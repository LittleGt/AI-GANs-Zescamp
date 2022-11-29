from flask import Flask,render_template
from flask.globals import request
import cv2
from numpy.lib.type_check import imag
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.python.ops.gen_array_ops import immutable_const_eager_fallback
import Code_LoadModel


app = Flask(__name__) 
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route("/") 
def Home():
    return render_template('web.html')

#modelname = "model/forestAice_generator648.h5"

@app.route("/generateForest")
def generateForestRoute():
    return render_template('gen_page.html'),Code_LoadModel.castpic('model/forestAice_generator648.h5') 

@app.route("/generateMount")
def generateMountRoute():
    return render_template('gen_page.html'),Code_LoadModel.castpic('model/buildMount_generator_Epoch10000.h5') 

@app.route("/generateWaterfall")
def generateWaterfallRoute():
    return render_template('gen_page.html'),Code_LoadModel.castpic('model/waterfall_generator.h5') 

@app.route("/generateDesert")
def generateDesertRoute():
    return render_template('gen_page.html'),Code_LoadModel.castpic('model/desert_generator.h5') 

@app.route("/generateMeadow")
def generateMeadowRoute():
    return render_template('gen_page.html'),Code_LoadModel.castpic('model/meadow_generator.h5') 

@app.route("/Credits")
def CreditsRoute():
    return render_template('cred.html')


if __name__=="__main__":
    app.run(debug=True)


"""
@app.route("/") 
def hello():
    var1 = "zes"
    var2 = "camp"
    return render_template('home.html',data = var1, data2 = var2)

@app.route("/after", methods = ['GET','POST'])
def afterdef():
    file=request.files['thisfile']
    fileคือตัวแปร = ขอไฟล์มา
    file.save('static/uploaded_file/file.jpg')
    model = load_model('DogCat_64.h5')
    image = cv2.imread('static/uploaded_file/file.jpg')
    image = (cv2.resize(image, (64,64)))/255
    image = image.reshape(1,64,64,3)
    prediction = model.predict(image)
    if prediction > 0.5:
        label_map='Dog'
    else:
        label_map='Cat'

    return render_template('after.html',data = label_map)
"""