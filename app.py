import io
import numpy as np
from flask import Flask, jsonify, request 
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils import load_img, img_to_array

app = Flask(__name__)


def vgg16_model_1():
    vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
    for layer in  vgg_model.layers:
        layer.trainable = False 
    
    # Create a new 'top' of the model (i.e. fully-connected layers).
    top_model = vgg_model.output
    top_model = Flatten(name="flatten")(top_model)
    output_layer = Dense(5, activation='softmax')(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    final_model = Model(inputs=vgg_model.input, outputs=output_layer)

    return final_model

def prepare_image(img):
    # Loading the img
    img = load_img(io.BytesIO(img),target_size=(224, 224))
    # Creating a batch of numpy array
    img = np.expand_dims(img_to_array(img), 0)
    
    return img


def predict_result(img):
    model = vgg16_model_1()
    model.load_weights('./vgg16_3.h5')
    pred_results = model.predict(img)

    return np.argmax(pred_results,axis=1)[0]


@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return jsonify(prediction=str(predict_result(img)))
    

@app.route('/', methods=['GET'])
def index():
    return 'Ship Classifier ready'


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')