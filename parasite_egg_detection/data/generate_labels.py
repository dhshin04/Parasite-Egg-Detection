''' Used to generate annotations for strongylid egg dataset '''
import os
import json
import random

hookworm_category = 5


def generate_empty_labels(json_path, images_path):
    # Generate a JSON file with empty annotations for each image in images_path

    annotations = {}

    image_list = sorted(os.listdir(images_path))
    image_id = 1
    with open(json_path, 'w') as json_file:
        for image_name in image_list:
            annotations[image_name] = {
                'image_id': image_id,
                'boxes': [],
                'labels': [],
            }
            image_id += 1

        json_file.write(        # Make JSON File with Formatting
            json.dumps(annotations, indent=4)
        )
    return image_id     # return image id for next label, if applicable


def read_xml(xml_name):
    # Read from xml and return image name and bounding boxes for image
    bboxes = []
    image_name = ''     # TODO
    return image_name, bboxes


def import_xml_annotations(json_path, xml_path):
    xml_files = sorted(os.listdir(xml_path))
    
    with open(json_path, 'r') as json_file:
        annotations = json.load(json_file)
        for xml_file in xml_files:
            image_name, bboxes = read_xml(xml_file)
            annotation = annotations[image_name]
            annotation['boxes'] = bboxes
            annotation['labels'] = [1] * len(bboxes)
    
    with open(json_path, 'w') as json_file:
        json_file.write(        # Make JSON File with Formatting
            json.dumps(annotations, indent=4)
        )


def add_label(json_path, image_name, image_id, boxes, labels):
    # Add annotation to provided json file, given image name, id, boxes, and labels

    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

        json_data[image_name] = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
        }
    
    with open(json_path, 'w') as json_file:
        json_file.write(        # Make JSON File with Formatting
            json.dumps(json_data, indent=4)
        )


def import_hookworms(json_path, annotations_path, next_image_id, num_imports, no_egg_imports):
    '''
    Import hookworm annotations from general dataset since hookworm eggs
    are a type of strongylid eggs. 

    Arguments:
        annotations_path (str): Path to json file that contains hookworm annotations
        num_imports (int): Number of hookworm egg image-annotation pair to add to json file
        no_hookworm_imports (int): Number of non-hookworm-egg image-annotation pair to add to json file
    '''

    hookworm_list = []      # List of image names that contain hookworm eggs
    not_hookworm = []       # List of image names that do NOT contain hookworm eggs
    with open(annotations_path, 'r') as annotations_file:
        annotations = json.load(annotations_file)
        for image_name, annotation in annotations.items():
            if len(annotation['labels']) > 0 and annotation['labels'][0] == hookworm_category:
                hookworm_list.append(image_name)
            else:
                not_hookworm.append(image_name)

        # Sample num_imports hookworm image-annotation pair from hookworm_list
        hookworm_sample = random.sample(hookworm_list, num_imports)
        for image_name in hookworm_sample:
            annotation = annotations[image_name]
            new_labels = [1] * len(annotation['labels'])    # Label replaced with 1 (binary class)
            add_label(json_path, image_name, next_image_id, annotation['boxes'], new_labels)
            next_image_id += 1
        
        not_hookworm_sample = random.sample(not_hookworm, no_egg_imports)
        for image_name in not_hookworm_sample:
            annotation = annotations[image_name]
            add_label(json_path, image_name, next_image_id, boxes=[], labels=[])
            next_image_id += 1


if __name__ == '__main__':
    dataset_path = os.path.join(os.path.dirname(__file__), 'strongylid_dataset')
    json_path = os.path.join(dataset_path, 'labels.json')
    images_path = os.path.join(dataset_path, 'images')
    xml_path = os.path.join(dataset_path, 'xml_annotations')

    next_image_id = generate_empty_labels(json_path, images_path)   # likely 120 images
    import_xml_annotations(json_path, xml_path)

    annotations_path = os.path.join(os.path.dirname(__file__), 'general_dataset', 'refined_labels.json')
    num_imports = 80        # 80 hookworm egg images
    no_egg_imports = 50        # 50 images without strongylid eggs
    import_hookworms(json_path, annotations_path, next_image_id, num_imports, no_egg_imports)
