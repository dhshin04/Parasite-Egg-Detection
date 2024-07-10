import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class FecalEggDataset(Dataset):

    def __init__(self, root, images, annotations, transforms=None, device='cpu'):
        self.root = root
        self.images = images
        self.annotations = annotations
        self.transforms = transforms
        self.device = device

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.images[idx])
        image = read_image(image_path)
        
        image_name = self.images[idx]
        annotation = self.annotations[image_name]   # Annotation for Image

        # None of the masks are crowds (only contain single instances)
        num_bbox = len(annotation['boxes'])
        iscrowd = torch.zeros((num_bbox), dtype=torch.int64).to(self.device)

        # Input Image
        image = tv_tensors.Image(image).to(self.device)

        # Target Annotations with Preprocessing
        target = {}
        target['boxes'] = tv_tensors.BoundingBoxes(     # Convert bounding boxes to tensor
            annotation['boxes'], 
            format='XYXY',
            canvas_size=F.get_size(image)
        )
        target['masks'] = tv_tensors.Mask(annotation['masks']).to(self.device)
        target['labels'] = torch.tensor(annotation['labels'], dtype=torch.int64).to(self.device)
        target['image_id'] = annotation['image_id']       # NOT tensor!
        target['area'] = torch.tensor(annotation['area'], dtype=torch.float32).to(self.device)
        target['iscrowd'] = iscrowd

        # Apply Additional Transformations to Image and Target (if necessary, such as Data Augmentation)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def __len__(self):
        return len(self.images)
