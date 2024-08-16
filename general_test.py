''' Test Model Prediction on General Model '''

import os, glob
from parasite_egg_detection.predict import predict
import random
import cv2

images = []

# Random Test Image from Untrained Set (Validation/Test Set)
test_images_path = os.path.join(
    os.path.dirname(__file__), 
    'parasite_egg_detection',
    'data',
    'general_test',
    'images',
)
image_path = os.path.join(
    test_images_path,
    os.listdir(test_images_path)[random.randint(0, 10999)],
)
image = cv2.imread(image_path)
images.append(image)

labeled_images, fec, epg = predict(
    images, 
    parasite='general',
)

predictions_path = os.path.join(os.path.dirname(__file__), 'predictions')

# Create predictions folder if it does not exist
if not os.path.exists(predictions_path):
    os.makedirs(predictions_path)

# Remove all previous images
images = glob.glob(os.path.join(predictions_path, '*'))
for image in images:
    os.remove(image)

os.chdir(predictions_path)

for i, img in enumerate(labeled_images, 1):
    filename = f'{i}.jpg'
    cv2.imwrite(filename, img)

print(f'Average FEC: {fec}, EPG: {epg}')
