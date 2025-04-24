# Sensitivity-Informed Augmentation for Robust Segmentation 

If you have any questions, please contact Laura Zheng at ```lyzheng@umd.edu```. Thank you!

## Getting Started 

### Environment Setup (Conda) 

First, create a conda environment with Python version 3.10:

```conda create --name bp-38 python=3.10```

Then, install the library dependencies via pip: 

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In case dependencies need to be installed manually, we use the following versions: 

- PyTorch 2.0.0 
- Latest version of MMSegmentation

## Setting the Right Paths 
This repo contains references to many local paths for purposes of config, datasets, and output directories. It is important to set these paths up prior to running experiments.

Here are the following files which need to be modified to your own system: 

- [SEG_CONFIG.py](SEG_CONFIG.py)
- [All convenience scripts in job_scripts folder](job_scripts)

The code reads the paths from these files throughout training and testing. 

## Setting up Supported Datasets 

Currently, this repo supports all backbones provided by MMSegmentation and additionally the datasets in [SEG_CONFIG.py](SEG_CONFIG.py):

Training datasets:
- Cityscapes
- ADE20K 
- PASCAL VOC 2012
- LoveDA
- POTSDAM
- Synapse
- A2I2Haze

Testing datasets: 
- ACDC
- Dark Zurich 
- Nighttime Driving 
- IDD

Each dataset has different setup instructions. You can find the setup instructions for most of them (all but a2i2haze) on this [MMSeg tutorial link](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html).

Additionally, you can also contact me if you would like a zip file of the post-processed data for convenience: ```lyzheng@umd.edu```.

## Training a Model 
To train a model, you can either call the Python training file [```train.py```](train.py) directly or use one of the convenience bash scripts provided in ```job_scripts```. 

There are many command-line arguments in the train.py script, which you can list with ```python train.py --help```. The convenience scripts like [```job_scripts/train_generic.sh```](job_scripts/train_generic.sh) help make training simple and reproducible. 

To train with the convenience script, simply run 

```bash job_scripts/train_generic.sh [aug] [model] [dataset]```

or, if you want to submit to a GPU cluster with a SLURM scheduler, you can simply run the same but with sbatch:

```sbatch job_scripts/train_generic.sh [aug] [model] [dataset]```

[aug] options: 'none', 'ours', 'default', 'autoaugment', 'augmix', 'randaugment', 'trivialaugment'

[model] options: any model name from subfolders of [```custom_configs/mmseg```](sensaug/custom_configs/mmseg). example: 'pspnet', 'segformer', 'vit', 'swin'. 

[dataset] options: any key from [SEG_CONFIG.py](SEG_CONFIG.py):

```
DATA_ROOT_LOOKUP = {
    "cityscapes" : f"{DATA_ROOT}/cityscapes",
    "ade20k" : f"{DATA_ROOT}/ade/ADEChallengeData2016",
    "pascal_voc12" : f"{DATA_ROOT}/VOCdevkit/VOC2012",
    "loveda" : f"{DATA_ROOT}/loveDA",
    "potsdam" : f"{DATA_ROOT}/potsdam",
    "synapse" : f"{DATA_ROOT}/synapse",
    "a2i2haze" : f"{DATA_ROOT}/a2i2haze"
}
```

NOTE: Our repo supports Tensorboard! You can launch Tensorboard while a model is training like so:
 
```tensorboard --logdir [work dir here] --host 0.0.0.0```

## Implementing a New Dataset 

If you would like to implement your own custom dataset, there are a few steps involved. 

### Step 1: Implement a new dataset MMSeg-style
Implement the dataset class in [```sensaug/dataset/datasets.py```](sensaug/dataset/datasets.py).
The existing custom datasets are short implementations because they subclass Cityscapes. For a more sophisticated implementation, you can check the official MMSeg tutorial: https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_datasets.html 

### Step 2: Create a training config for the dataset 
MMSegmentation uses separate training configs for each dataset; it makes things easier for fine-tuning and whatnot. 

Our training script is set up to adapt any existing config for a dataset to any supported model, so only one new config is needed to support all models. 

In the past, we just create a new training config for the dataset in the path: [```custom_configs/mmseg/pspnet```](sensaug/custom_configs/mmseg/pspnet). This is purely because PSPNet already had many dataset implemented, and it is a light(er) model to test.

The [PSPNet config for A2I2Haze](sensaug/custom_configs/mmseg/pspnet/pspnet_r18-d8_4xb2-80k_a2i2haze.py) is entirely custom, so it may be easiest to make a copy of that file and swap out paths. 

Make sure the config file follows the same naming convention as all other configs, even if the naming convention is obscure. If you decide to make a copy of the A2I2Haze config, you can name the file like so: 
```pspnet_r18-d8_4xb2-80k_DATASETNAME.py``` 
Make note of the dataset name for the next step. 

### Step 3: Modify SEG Config 

Remember that SEG_CONFIG.py file we keep referencing? Well, it's time to modify that file now. Here's the link for convenience: [SEG_CONFIG.py](SEG_CONFIG.py)

Simply add the name of the new dataset to the dictionary of supported datasets. Make sure the key matches that of ```DATASETNAME``` that you chose in the last step in the config naming. 

The training script should automatically pull from this config. If all steps go smoothly, then you should be able to run the convenience script with the new dataset, with the ```DATASETNAME``` from the config you chose as the dataset argument. 
