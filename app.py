from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__, static_folder='/')
model = tf.keras.models.load_model('xceptionnetbalanced.h5')

class_names = ['Actinic Keratoses','Basal Cell Carcinoma','Benign Keratosis','Dermatofibroma','Melanoma','Melanocytic Nevi','Vascular Lesions']

@app.route('/', methods=['GET'])
def skincheck():
    return render_template('MainPage.html')

@app.route('/prediction', methods=['POST'])
def predict():
    image = request.files['skinImage']
    if image:
        image_path = 'skinimages/' + image.filename
        image.save(image_path)

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        class_id = np.argmax(preds[0])
        
        return render_template('EndPage.html', prediction=class_names[class_id], image_url=url_for('static', filename=image_path))


    return render_template('MainPage.html')

if __name__ == '__main__':
    app.run(port=5000, debug=False)