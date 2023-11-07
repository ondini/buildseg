# RGBT Fusion

Codebase for Computer Vision and DeepLearning approaches to RGB and thermal sensor fusion for maritime environment, what is a topic of my SEA.AI internship and Master Thesis at CTU.

## Training instructions 
Main entrypoint is _train.py_ file, which traines the network provided in _config.json_. This config file allows for setting of a lot of other training aspects allowing for high modularity of the code. 

Important modules are 
    - models, where are defined architectures among which can be chosen in th config file,
    - scoring, where are defined metric and loss functions,
    - trainer, where is defined the data propagation in one epoch, as this is also dependent on experiment (such as semantic segmentation vs object detection vs classification).

Let's suppose you have fiftyone dataset export prepared in /your/dataset/path, then to start training, you should run

    python prepare_fo_dataset.py -d /your/dataset/path 

to create live fiftyone dataset with views corresponding to train and test split. This dataset can be viewed in browser using fiftyone.

Then  simply run

    python train.py -c /path/to/your/config.json

You can view the results of your training in the browser after running

    tensorboar --logdir=/outpath/you/have/in/confing --host 0.0.0.0



## Docker instructions
This codebase contains _Dockerfile_, so that all experiments can be reproduced in the same environment. 

In order to run the container, first build it using

    docker build -t seaai_pytorch

and the you can run it with following command

    docker run -it --ipc=host --gpus all -p 6006:6006 -p 5151:5151 -v /your/dataset/path:/app/data/ -v codes/path:/app/codes --name rgbt_dev seaai_pytorch 
    
which allows for exposed __tensorboard__ and __fiftyone__ ports for the interface. 