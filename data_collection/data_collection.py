import numpy as np
import cv2

from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs("./data_collection/dataset"):
        np.savez(f"./data_collection/dataset/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1


def get_mask_of_color(img, color):
    """
    veryslow but does the job
    """
    n, m, q  = img.shape
    
    mask = np.zeros((n,m), dtype=np.uint8)

    color_array = color * np.ones((n,m,3), dtype=np.uint8)
    mask = np.equal(img, color_array).all(2)
    mask = mask.astype(np.uint8)
    
    return mask

def clean_segmented_image(seg_img):
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    # TODO: do we need to return the cleaned up masks
    # TODO: mulple instances get classified as one instance, is this a problem?

    # morphology params
    kernel = np.ones((3,3),np.uint8)
    it = 1

    boxes = []
    classes = []

    # color schemes for the classes
    class_color_def = {
        'duckie': [100, 117, 226],
        'bus': [ 216, 171, 15],
        'cone': [226, 111, 101],
        'truck': [116, 114, 117]
    }

    class_num_def = {
        'duckie': 1,
        'bus': 4,
        'cone': 2,
        'truck': 3
    }


    for key in class_color_def:
        mask = get_mask_of_color(seg_img, class_color_def[key])
        
        # clean "snow"
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations = it)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations = it+2)

        # get contour
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # get boxes
        for contour in contours:
            # TODO: maybe check size
            x,y,w,h = cv2.boundingRect(contour)

            boxes.append([x, y, x+w, y+h])
            classes.append(class_num_def[key])


    # draw
    for i, box in enumerate(boxes):
        cv2.rectangle(seg_img, (box[0], box[1]), (box[2], box[3]), [classes[i]*50, 255 - classes[i], 0])

    cv2.imshow('1', seg_img)

    cv2.waitKey(50) 
    return boxes, classes

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 2000

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        obs = cv2.resize(obs, (224, 224))
        segmented_obs = cv2.resize(segmented_obs, (224, 224))

        boxes, classes = clean_segmented_image(segmented_obs)
        if len(boxes) > 0:
            save_npz(obs, boxes, classes)

            nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
