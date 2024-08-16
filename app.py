''' Flask Application for Strongylid Fecal Egg Count '''

from flask import Flask, render_template, request
import os, glob, random, cv2
import numpy as np
from parasite_egg_detection.predict import predict

app = Flask(__name__)


def condition(epg):
    '''
    Identifies infestation state of host animal

    Arguments:
        epg (int): Eggs per gram
    Returns:
        (str): Infestation state/condition
    '''

    if epg > 650: 
        return "Heavy Infestation: Anthelmintic Treatment Necessary"
    elif epg > 350: 
        return "Moderate Infestation: Anthelmintic Treatment Recommended"
    elif epg >= 50:
        return "Light Infestation: Treatment Not Necessary"
    return "No Infestation: Sample is Healthy"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', show=False, fec=-1, epg=-1, condition='', image='')
    elif request.method == 'POST':
        images_names = ['image1', 'image2']
        images = []

        for image_name in images_names:
            if image_name not in request.files:
                return 'No image provided', 404

            image_file = request.files[image_name]
            image_byte = np.frombuffer(image_file.read(), dtype=np.uint8)   # Convert file to bytestring np array
            image = cv2.imdecode(image_byte, cv2.IMREAD_COLOR)     # Read image from bytestring array
            images.append(image)

        labeled_images, fec, epg = predict(
            images, 
            parasite='strongylid',
        )

        images_path = os.path.join(os.path.dirname(__file__), 'static', 'images')
        os.chdir(images_path)

        filenames = []
        for image_name, image in zip(images_names, labeled_images):
            filename = image_name + '_labeled.jpg'
            cv2.imwrite(filename, image)
            filenames.append(filename)

        return render_template('index.html', show=True, fec=fec, epg=epg, condition=condition(epg), image1_path=f'../static/images/{filenames[0]}', image2_path=f'../static/images/{filenames[1]}')

    return 'Bad Request: Must send GET or POST requests only', 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
