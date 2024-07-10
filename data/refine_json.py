# Takes raw JSON file and change into JSON annotation usuable by torchvision's model

import os
import json
from mask_generator import generate_mask
from find_image_name import find_image_name_by_id, find_image_name_efficient


def refine_json(in_json_path, out_json_path, dataset_type=None):
    '''
    Refines original json file to annotation format usuable by Mask R-CNN Model

    Arguments:
        in_json_path (str): Path to original json file
        out_json_path (str): Path to new refined json file
    '''

    # Convert Original JSON to HashMap<image_name, AnnotationDict> format
    refined_annotations = {}

    # Load JSON File
    with open(in_json_path, 'r') as json_file:
        json_data = json.load(json_file)
        original_annotations = json_data['annotations']

        count = 0.      # for tracking purposes
        for original_annotation in original_annotations:
            image_id = original_annotation['image_id']       # Image ID of Annotation
            if dataset_type == 'training':      # labels.json
                image_name = find_image_name_efficient(image_id)
                image_folder = 'trainingset'
            else:       # test_labels.json
                image_name = find_image_name_by_id(in_json_path, image_id)
                image_folder = 'testset'
            image_path = os.path.join(os.path.dirname(__file__), image_folder, 'images', image_name)

            # Find bounding box and mask instance for this annotation
            bbox = original_annotation['bbox']
            mask = generate_mask(image_path, bbox)

            if image_id not in refined_annotations:          # If not present, add image_id and append
                refined_annotation = {
                    'image_id': image_id,
                    'boxes': [bbox],
                    'masks': [mask],
                    'labels': [original_annotation['category_id'] + 1],     # Category starts at 1
                    'area': [original_annotation['area']],
                }
                refined_annotations[image_name] = refined_annotation
            else:
                refined_annotation = refined_annotations[image_name]

                refined_annotation['boxes'].append(bbox)
                refined_annotation['masks'].append(mask)
                refined_annotation['labels'].append(original_annotation['category_id'] + 1)
                refined_annotation['area'].append(original_annotation['area'])
            
            count += 1
            if count % 110 == 0:
                print(f'Finished {(count * 100 / 11000):.2f}%')

    with open(out_json_path, 'w') as refined_labels:
        json.dump(refined_annotations, refined_labels)


def main():
    # Create refined labels for training set
    in_json_path = os.path.join(os.path.dirname(__file__), 'trainingset', 'labels.json')
    out_json_path = os.path.join(os.path.dirname(__file__), 'trainingset', 'refined_labels.json')
    refine_json(in_json_path, out_json_path, dataset_type='training')
    
    print()

    # Create refined labels for validation/test sets
    in_json_path = os.path.join(os.path.dirname(__file__), 'testset', 'test_labels.json')
    out_json_path = os.path.join(os.path.dirname(__file__), 'testset', 'refined_test_labels.json')
    refine_json(in_json_path, out_json_path, dataset_type='test')


if __name__ == '__main__':
    main()
