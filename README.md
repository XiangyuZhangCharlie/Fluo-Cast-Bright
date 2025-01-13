# Fluo-Cast-Bright

![FluoCastBright](https://github.com/user-attachments/assets/31bac245-363d-49e9-bbda-4b9349364b7e)

Code repository for the Fluo-Cast-Bright pipeline "Xiangyu Zhang, Claudia Baumann, Rabindranath De La Fuente: Fluo-Cast-Bright: A Deep Learning Pipeline for the Non-Invasive Prediction of Chromatin Structure and Developmental Potential in Live Oocytes" published in Communications Biology.

This code is in active development and adaptation to other cellular applications. We are not maintaining this code and are simply releasing the code to the community AS IS. Active response is not guaranteed.

## Installation

The environment required for running Fluo-Cast-Bright pipeline can be installed using Anaconda/Miniconda on Windows or Linux:

```bash
conda env create -f <Environment_Windows.yml or Environment_Linux.yml>
conda activate fluocastbright_env
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```


## Application

Training a fluorescence prediction model with training and validation images in separate folders. The image format is (2,Z,X,Y). The first dimension represents the bright-field image and the corresponding fluorescence map.

Make sure to end every directory and path with /

```bash
python FluoCastBright_Pipeline.py --Option FPM_Train --FPMTrain_TrainImage_Directory <path to training images> --FPMTrain_ValidImage_Directory <path to validation images> --FPMTrain_LogFolder_Directory <path to log folder> --FPMTrain_ModelFolder_Directory <path to model ouptut folder>
```

If GPU limits, reached reduce batch size and starting channel number.

Utilizing an existing fluorescence prediction model to analyze a group of new brightfield images. The image format is (Z,X,Y).

```bash
python FluoCastBright_Pipeline.py --Option FPM_Predict --FPMPredict_Image_Directory <path to brightfield images> --FPMPredict_PFM_Directory <path to output predicted fluorescence maps> --FPMPredict_ModelFolder_Directory <path to model> --FPMPredict_FPMModel_Name <model name>
```

Training classification models with a group of predicted fluorescence maps. The image format is (Z,X,Y), and the correct classification labels are in the file names.

```bash
python FluoCastBright_Pipeline.py --Option CM_Train --CMTrain_PFM_Directory <path to predicted fluorescence maps for training> --CMTrain_Models_Directory <path to output classification models>
```

Testing the accuracy of classification models generated in the previous step on their respective validation set images.

```bash
python FluoCastBright_Pipeline.py --Option CM_Test --CMTest_PFM_Directory <path to predicted fluorescence maps used for training> --CMTest_Models_Directory <path to classification models>
```

Apply existing classification models to classify fluorescence prediction maps generated from new brightfield images. For this step the images can be with or without true classification in file names. The image format is (Z,X,Y).

```bash
python FluoCastBright_Pipeline.py --Option CM_Predict --CMPredict_PFM_Directory <path to predicted fluorescence maps> --CMPredict_Models_Directory <path to classification models>
```

**Thanks for your interest in our work!**
