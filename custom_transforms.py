import torchvision.transforms as transforms
import random
import copy


def random_rotate_90(image, annotation):
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
    # Rotate and flip
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    image = to_pil(image)

    image, annotation = random_rotate_90(image, annotation)
    image, annotation = random_horizontal_flip(image, annotation)

    image = to_tensor(image)

    # Color Jitter
    # color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1)
    # image = color_jitter(image)

    return image, annotation


if __name__ == '__main__':
    import os
    import json
    from data.fecal_egg_dataset import FecalEggDataset
    from split import even_train_test_split
    import time
    import random

    testset_path = os.path.join(os.path.dirname(__file__), 'data', 'testset')
    test_images_path = os.path.join(testset_path, 'images')
    test_labels_path = os.path.join(testset_path, 'refined_test_labels.json')
    with open(test_labels_path, 'r') as refined_test_labels:
        test_annotations = json.load(refined_test_labels)
    
    train_images, test_images = even_train_test_split(
        sorted(os.listdir(test_images_path)),
        annotations=test_annotations,
        test_size=0.4,
        random_seed=10,
    )

    train_dataset = FecalEggDataset(test_images_path, train_images, test_annotations, transforms=transform)

    print('Start Count!')
    start = time.time()
    for i in range(1351):
        if i % 135 == 0:
            print(f'{i / 1350 * 100:.2f}% done')
        image, target = train_dataset.__getitem__(random.randint(0, 1000))
    end = time.time()

    print(f'Elapsed Time: {end - start:.1f}s')
