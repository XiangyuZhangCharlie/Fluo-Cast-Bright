# Fluo-Cast-Bright

This is the code repository for the Fluo-Cast-Bright pipeline in the paper "Fluo-Cast-Bright: A Deep Learning Pipeline for the Non-Invasive Prediction of Chromatin Structure and Developmental Potential in Live Oocytes" published on Communications Biology.

## 1. Installation

The environment required for running Fluo-Cast-Bright pipeline can be installed using Anaconda/Miniconda:

```bash
conda env create -f Environment.yml
conda activate fluocastbright_env
```

## 2. Application

Training a fluorescence prediction model with training and validation images in separate folders. The image format is (2,Z,X,Y). The first dimension represents the bright-field image and fluorescence map.

```bash
python FluoCastBright_Pipeline.py --Option FPM_Train --FPMTrain_Device_To_Use <cuda:0 or cpu> --FPMTrain_TrainImage_Directory <path to training images> --FPMTrain_ValidImage_Directory <path to validation images> --FPMTrain_LogFolder_Directory <path to log folder> --FPMTrain_ModelFolder_Directory <path to model ouptut folder>
```

Utilizing an existing fluorescence prediction model to analyze a group of new brightfield images. The image format is (Z,X,Y).

```bash
python FluoCastBright_Pipeline.py --Option FPM_Predict --FPMPredict_Device_To_Use <cuda:0 or cpu> --FPMPredict_Image_Directory <path to brightfield images> --FPMPredict_PFM_Directory <path to output predicted fluorescence maps> --FPMPredict_ModelFolder_Directory <path to model> --FPMPredict_FPMModel_Name <model name>
```

Training classification models with a group of predicted fluorescence maps. The image format is (Z,X,Y) and the correct classification labels are in the file names.

```bash
python FluoCastBright_Pipeline.py --Option CM_Train --CMTrain_Device_To_Use <cuda:0 or cpu> --CMTrain_PFM_Directory <path to predicted fluorescence maps for training> --CMTrain_Models_Directory <path to output classification models>
```

Testing the accuracy of classification models on their respective validation set images.

```bash
python FluoCastBright_Pipeline.py --Option CM_Test --CMTest_Device_To_Use <cuda:0 or cpu>
```

**Thanks for your interest in our work!**
