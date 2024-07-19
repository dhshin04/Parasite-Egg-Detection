# Takes raw JSON file and change into JSON annotation usuable by torchvision's model

import os
import json
from find_image_name import find_image_name_by_id, find_image_name_efficient


def refine_bbox(old_bbox):
    '''
    Change bounding box format from COCO to PASCAL VOC

    Arguments:
        old_bbox (tensor): Bounding box in COCO format
    Returns:
        new_bbox (tensor): Bounding box in PASCAL VOC format
    '''

    # old_bbox = [x, y, width, height] where (x,y) is top left corner of box
    # Positive direction is right, down
    new_bbox = []
    new_bbox.append(old_bbox[0])                # x1
    new_bbox.append(old_bbox[1])                # y1
    new_bbox.append(old_bbox[0] + old_bbox[2])  # x2
    new_bbox.append(old_bbox[1] + old_bbox[3])  # y2
    return new_bbox     # [x1, y1, x2, y2]


def refine_json(in_json_path, out_json_path, dataset_type=None):
    '''
    Refines original json file to annotation format usuable by Faster R-CNN Model

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

        for original_annotation in original_annotations:
            image_id = original_annotation['image_id']       # Image ID of Annotation
            if dataset_type == 'test':      
                image_name = find_image_name_efficient(image_id)
            else:       
                image_name = find_image_name_by_id(in_json_path, image_id)

            # Find bounding box and mask instance for this annotation
            bbox = original_annotation['bbox']
            bbox = refine_bbox(bbox)

            if image_name not in refined_annotations:          # If not present, add image_id and append
                refined_annotation = {
                    'image_id': image_id,
                    'boxes': [],
                    'labels': [],
                    'area': []
                }
                refined_annotations[image_name] = refined_annotation

            refined_annotations[image_name]['boxes'].append(bbox)
            refined_annotations[image_name]['labels'].append(original_annotation['category_id'] + 1)     # Category starts at 1
            refined_annotations[image_name]['area'].append(original_annotation['area'])

    with open(out_json_path, 'w') as refined_labels:
        json.dump(refined_annotations, refined_labels)


def main():
    # Create refined labels for data set
    in_json_path = os.path.join(os.path.dirname(__file__), 'dataset', 'labels.json')
    out_json_path = os.path.join(os.path.dirname(__file__), 'dataset', 'refined_labels.json')
    refine_json(in_json_path, out_json_path, dataset_type='dataset')
    
    # Create refined labels for test
    in_json_path = os.path.join(os.path.dirname(__file__), 'test', 'labels.json')
    out_json_path = os.path.join(os.path.dirname(__file__), 'test', 'refined_labels.json')
    refine_json(in_json_path, out_json_path, dataset_type='test')


if __name__ == '__main__':
    main()
