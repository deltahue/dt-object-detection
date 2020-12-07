# Object Detection Exercise


*Authors: David Helm, Frederic Letsch, Silvan Löw*

## Set up dependencies
1. Clone this repository to your workspace and make sure you followed the [setup instructions](https://docs.duckietown.org/daffy/duckietown-learning-robotics/out/lra_object_detection.html).
1. If you want to train on your GPU, make sure to have the [nvidia driver installed](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux).
1. If you also want to evaluate on the GPU using docker, you additionally need the [nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker).

## Making a dataset
In order to generate new .npz image data, please run the data_collection.py script, e.g. with
```bash
python3 data_collection/data_collection.py
```

## Training the model
1. The training dataset is assumed to be in `data_collection/dataset`.
1. Training can be started with
    ```bash 
    python3 model/train.py
    ``` 
    The weights will be saved in `model/weights`

## Running and validating the model
1. Either download the docker image form dockerhub or build one using `docker build` in the root of this repository.
1. Run a container from this image once to make sure the model is downloaded.
1. Now `cd eval` and run the evaluation with 
    ```bash
    `make eval-gpu SUB=<submission dockerfile>` or `make eval-cpu SUB=<submission dockerfile>`
    ```