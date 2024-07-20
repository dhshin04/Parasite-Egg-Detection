''' Generate masks from training set images using SAM '''
import os
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'content', 'weights', 'sam_vit_b_01ec64.pth')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = 'vit_b'

# Load SAM Model & Predictor
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)


def generate_mask(image_path, bounding_box):
    '''
    Generate mask for given bounding box in image

    Arguments:
        json_path (str): Path to json file to access annotations. Needed for getting image name.
        image_path (str): Image Path - needed to set SAM Model to predict on image. 
        bounding_box (float[]): Bounding box in image - needed to generate mask.
    
    Returns:
        mask (torch.Tensor): Generated mask based on bounding box in given image.
    '''

    # Read image as BGR -> convert to np array -> convert to RGB np array
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load Image to SAM
    mask_predictor.set_image(image)

    # Predict Mask For Each Bounding Box
    bounding_box = np.array(bounding_box)
    mask = mask_predictor.predict(
        box=bounding_box,       # Bounding Box from which mask should be generated
        multimask_output=False, # One Mask per Bounding Box
    )[0]
    
    return mask
