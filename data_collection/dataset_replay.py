import os
import numpy as np
import cv2
import time
# Replay dataset
#dataset_folder = "./eval/dataset/"
dataset_folder = "./data_collection/dataset/"
os.listdir(dataset_folder)
dataset_files = list(filter(lambda x: "npz" in x, os.listdir(dataset_folder)))
dataset_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for file in dataset_files:
    with np.load(f'{dataset_folder}{file}') as data:
        img, boxes, classes = tuple([data[f"arr_{i}"] for i in range(3)])
        for i, box in enumerate(boxes):
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [int(classes[i]*50), int(255 - classes[i]), int(0)])

        cv2.imshow('dataset replay', img)
        cv2.waitKey(20)
    time.sleep(0.2)
