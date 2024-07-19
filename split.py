import json
import random


def even_train_test_split(image_list, annotations, test_size=0.5, random_seed=1):
    '''
    Splits set into training and test sets BUT ensures training set has even amount
    of images from each category.

    Arguments:
        image_list (list): List of image names
        annotations (dictionary): Dictionary of image name and annotation dict pairs
        test_size (float): Ratio of test size
        random_seed (int): Random seed for consistent results
    Returns:
        (tuple): Lists of image names, one for training and other for test sets
    '''

    random.seed(random_seed)

    if test_size > 1 or test_size < 0:
        raise ValueError('Test Size must be between 0 and 1')

    image_into_category = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [],
        11: [],
    }

    for image_name in image_list:
        annotation = annotations[image_name]
        labels = annotation['labels']
        if len(labels) > 0:     # If image has object, put image in appropriate category list
            category = labels[0]
            image_into_category[category].append(image_name)
    
    train_images = []
    test_images = []
    for images in image_into_category.values():     # from each category list
        for image_name in images:                   # retrieve image and put into either training or test sets
            if random.random() >= test_size:
                train_images.append(image_name)
            else:
                test_images.append(image_name)
    
    return train_images, test_images
