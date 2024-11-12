# Fluo-Cast-Bright

This is the code repository for the Fluo-Cast-Bright pipeline in the paper "Fluo-Cast-Bright: A Deep Learning Pipeline for the Non-Invasive Prediction of Chromatin Structure and Developmental Potential in Live Oocytes" published on Communications Biology.

## 1. Installation

The environment required for running Fluo-Cast-Bright pipeline can be installed using Anaconda/Miniconda:

```bash
conda env create -f Environment.yml
conda activate fluocastbright_env
```

## 2. Application

Training a fluorescence prediction network with training and validation images in separate folders. The image format is (2,Z,X,Y). The first dimension represents the bright-field image and fluorescence map.

```bash
python FluoCastBright_Pipeline.py --Option FPM_Train --FPMTrain_Device_To_Use <cuda:0 or cpu> --FPMTrain_TrainImage_Directory <path to training images> --FPMTrain_ValidImage_Directory <path to validation images> --FPMTrain_LogFolder_Directory <path to log folder> --FPMTrain_ModelFolder_Directory <path to model ouptut folder>
```

Utilizing an existing fluorescence prediction network to analyze a group of brightfield images. The image format is (Z,X,Y).

```bash
python FluoCastBright_Pipeline.py --Option FPM_Predict --FPMPredict_Device_To_Use <cuda:0 or cpu> --FPMPredict_Image_Directory <path to brightfield images> --FPMPredict_PFM_Directory <path to output predicted fluorescence maps> -- 
```

