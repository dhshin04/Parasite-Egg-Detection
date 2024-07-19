import torchvision.transforms as transforms
import random
import copy


def random_rotate_90(image, annotation):
    # Randomly rotate given image and bounding boxes by 90 degree intervals
    
    # Image is tensor, annotation is a dict
    degree = 90 * random.randint(0, 3)
    if degree == 0:
        return image, annotation
    
    width, height = image.size

    # Rotate image
    random_rotate = transforms.RandomRotation((degree, degree), expand=True)
    image = random_rotate(image)

    # Rotate bounding box
    bboxes = annotation['boxes']
    rotated_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = tuple(bbox)

        if degree == 90:
            rotated_bboxes.append([y1, width - x2, y2, width - x1])
        elif degree == 180:
            rotated_bboxes.append([width - x2, height - y2, width - x1, height - y1])
        elif degree == 270:
            rotated_bboxes.append([height - y2, x1, height - y1, x2])
        else:
            raise ValueError('Degree is not in 90 degree intervals')
    
    # To not alter original annotation dictionary
    annotation_copy = copy.deepcopy(annotation)
    annotation_copy['boxes'] = rotated_bboxes

    return image, annotation_copy


def random_horizontal_flip(image, annotation, p=0.5):
    # Horizontally flip image and bounding box with probability of p

    if random.random() < p:
        width, _ = image.size

        # Flip Image
        flip = transforms.RandomHorizontalFlip(p=1.0)
        image = flip(image)

        # Flip bounding box
        bboxes = annotation['boxes']
        flipped_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = tuple(bbox)
            flipped_bboxes.append([width - x2, y1, width - x1, y2])
        
        # To not alter original annotation dictionary
        annotation_copy = copy.deepcopy(annotation)
        annotation_copy['boxes'] = flipped_bboxes

        return image, annotation_copy
    
    return image, annotation


def transform(image, annotation):
    # Apply transformations on image

    # Rotate and flip
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    image = to_pil(image)

    image, annotation = random_rotate_90(image, annotation)
    image, annotation = random_horizontal_flip(image, annotation)

    image = to_tensor(image)

    # Color Jitter - omitted for now due to increase in train time
    # color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1)
    # image = color_jitter(image)

    return image, annotation
