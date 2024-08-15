import os
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class for COCO
class CocoDataset(Dataset):
    # Initialization
    def __init__(self, root_dir, set='train2017', transform=None):
        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform
        # Load COCO annotations
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        # Load class names and labels
        self.load_classes()
    # Load and sort class categories
    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        # Map class names to labels
        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)
        # Map labels to class names
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
    # Return number of images
    def __len__(self):
        return len(self.image_ids)
    # Get a sample from the dataset
    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample
    # Load image
    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.
    # Load annotations
    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        if len(annotations_ids) == 0:
            return annotations
        # Parse and filter annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)
        # Convert bbox format
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        return annotations

# Collate function for batching
def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    imgs = imgs.permute(0, 3, 1, 2)
    return {'img': imgs, 'annot': annot_padded, 'scale': scales}

# Resizer class for images
class Resizer(object):
    # Initialization
    def __init__(self, img_size=512):
        self.img_size = img_size
    # Resize image and annotations
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        annots[:, :4] *= scale
        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}

# Augmenter class for image transformations
class Augmenter(object):
    # Apply horizontal flip
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]
            rows, cols, channels = image.shape
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            x_tmp = x1.copy()
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
            sample = {'img': image, 'annot': annots}
        return sample

# Normalizer class for image normalization
class Normalizer(object):
    # Initialization with mean and std
    def __init__(self, mean=[0.1428785, 0.28567852, 0.24536263], std=[0.17634014, 0.28190098, 0.26452454]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])
    # Normalize image
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}