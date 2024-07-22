''' Test Model Prediction '''

import os
from parasite_egg_detection.predict import predict
import random

# Test Image
image1_path = os.path.join(os.path.dirname(__file__), 'test.webp')

# Random Test Image from Untrained Set (Validation/Test Set)
images_path = os.path.join(
    os.path.dirname(__file__), 
    'parasite_egg_detection',
    'data',
    'strongylid_dataset',
    'test_images',
)
image2_path = os.path.join(
    images_path,
    os.listdir(images_path)[random.randint(0, 49)],
)
fec, epg = predict(
    [image1_path, image2_path], 
    parasite='strongylid',
)
print(f'Average FEC: {fec}, EPG: {epg}')
