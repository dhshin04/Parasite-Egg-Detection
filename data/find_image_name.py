''' Find image name by ID in JSON file '''
import json


def find_image_name_by_id(json_path, image_id):
    # General way to finding image name by ID

    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
        images = json_data['images']
        for image in images:
            if image['id'] == image_id:
                return image['file_name']
        raise Exception(f"Image with this ID not found: {image_id}")


def find_image_name_efficient(image_id):
    # More efficient way to finding image name - works only for test folder images

    # All image names end with 'ParasiteName_XXXX.jpg'
    category = (image_id - 1) // 1000

    parasite = {
        0: 'Ascaris lumbricoides',
        1: 'Capillaria philippinensis',
        2: 'Enterobius vermicularis',
        3: 'Fasciolopsis buski',
        4: 'Hookworm egg',
        5: 'Hymenolepis diminuta',
        6: 'Hymenolepis nana',
        7: 'Opisthorchis viverrine',
        8: 'Paragonimus spp',
        9: 'Taenia spp. egg',
        10: 'Trichuris trichiura',
    }

    # ID is between 1 and 1000 (inclusive)
    id = image_id % 1000
    if id == 0:
        id = 1000
    id = str(id)
    
    while len(id) < 4:
        id = '0' + id

    image_name = parasite[category] + '_' + id + '.jpg'

    return image_name
