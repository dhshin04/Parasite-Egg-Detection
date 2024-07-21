''' Used to generate annotations for strongylid egg dataset '''
import os
import json
import glob
import random
import xml.etree.ElementTree as ET
import shutil   # To copy image from one path to another

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

    tree = ET.parse(xml_name)
    root = tree.getroot()

    image_name = root.find('filename').text
    
    objects = root.findall('object')
    for object in objects:
        bbox = []
        bbox.append(float(object.find('bndbox/xmin').text))
        bbox.append(float(object.find('bndbox/ymin').text))
        bbox.append(float(object.find('bndbox/xmax').text))
        bbox.append(float(object.find('bndbox/ymax').text))

        bboxes.append(bbox)

    return image_name, bboxes


def import_xml_annotations(json_path, xml_path):
    # Import bounding boxes from xml file and generate corresponding labels as well

    xml_files = sorted(os.listdir(xml_path))
    
    with open(json_path, 'r') as json_file:
        annotations = json.load(json_file)
        for xml_file in xml_files:
            image_name, bboxes = read_xml(
                os.path.join(xml_path, xml_file)
            )
            annotation = annotations[image_name]
            annotation['boxes'] = bboxes
            annotation['labels'] = [1] * len(bboxes)    # Labels are all 1 (binary class)
    
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


def add_image(source_path, destination_path):
    if not os.path.exists(destination_path):
        shutil.copyfile(source_path, destination_path)


def import_hookworms(json_path, old_images_path, new_images_path, annotations_path, next_image_id, num_imports, no_egg_imports):
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
            add_image(
                os.path.join(old_images_path, image_name),
                os.path.join(new_images_path, image_name),
            )
            next_image_id += 1
        
        not_hookworm_sample = random.sample(not_hookworm, no_egg_imports)
        for image_name in not_hookworm_sample:
            annotation = annotations[image_name]
            add_label(json_path, image_name, next_image_id, boxes=[], labels=[])
            add_image(
                os.path.join(old_images_path, image_name),
                os.path.join(new_images_path, image_name),
            )
            next_image_id += 1


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), 'strongylid_dataset')
    json_path = os.path.join(dataset_path, 'labels.json')
    general_images_path = os.path.join(dataset_path, 'general_images')
    images_path = os.path.join(dataset_path, 'images')
    xml_path = os.path.join(dataset_path, 'xml_annotations')

    # Delete existing files in images directory to avoid conflicts
    files = glob.glob(os.path.join(images_path, '*'))
    for file in files:
        try:
            os.remove(file)
        except:
            raise Exception('Failed to delete image in images folder')

    # Import general strongylid egg images
    general_images = sorted(os.listdir(general_images_path))
    for image_name in general_images:
        add_image(
            os.path.join(general_images_path, image_name),
            os.path.join(images_path, image_name),
        )

    # Import general strongylid egg annotations
    next_image_id = generate_empty_labels(json_path, images_path)   # likely 120 images
    import_xml_annotations(json_path, xml_path)

    # Import hookworm egg images and annotations
    general_dataset_path = os.path.join(os.path.dirname(__file__), 'general_dataset')
    hookworm_images_path = os.path.join(general_dataset_path, 'images')
    annotations_path = os.path.join(general_dataset_path, 'refined_labels.json')
    num_imports = 80             # Num Hookworm egg images
    no_egg_imports = 50          # Num Images without strongylid eggs
    import_hookworms(
        json_path=json_path, 
        old_images_path=hookworm_images_path, 
        new_images_path=images_path,
        annotations_path=annotations_path, 
        next_image_id=next_image_id, 
        num_imports=num_imports, 
        no_egg_imports=no_egg_imports,
    )


if __name__ == '__main__':
    main()
