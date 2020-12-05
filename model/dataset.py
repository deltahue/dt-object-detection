import os
import numpy as np
import torch
import transforms as T
from PIL import Image


class DuckietownDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all npz files, sorting them to ensure that they are aligned
        self.archives =  list(filter(lambda x: "npz" in x, os.listdir(root)))
        self.archives.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    def __getitem__(self, idx):
        # load image, boxes and labels
        with np.load(f'{self.root}{self.archives[idx]}') as data:
            img_array, boxes, classes = tuple([data[f"arr_{i}"] for i in range(3)])
        # convert image to PIL image
        img = Image.fromarray(img_array)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(classes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.archives)

def get_transform(self, train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)