##################################################
# Fluo-Cast-Bright Pipeline
##################################################

import os
import sys
import argparse
from Train_FPM import FPM_Train_Main
from Predict_FPM import FPM_Predict_Main
from Train_CM import CM_Train_Main
from TestValid_CM import CM_TestValid_Main
from PredictNew_CM import CM_Predict_New_Images_Main

Parser = argparse.ArgumentParser()
Parser.add_argument('--Option', help="Which function to run", type=str)

# FPM Training Parameters
Parser.add_argument('--FPMTrain_Device_To_Use', help="On which device to run program", default="cuda:0", type=str)
Parser.add_argument('--FPMTrain_TrainImage_Directory', help="Select the training set image directory")
Parser.add_argument('--FPMTrain_ValidImage_Directory', help="Select the validation set image directory")
Parser.add_argument('--FPMTrain_LogFolder_Directory', help="Select the output log folder directory")
Parser.add_argument('--FPMTrain_ModelFolder_Directory', help="Select the output model folder directory")
Parser.add_argument('--FPMTrain_Max_Epochs', help="Max epoch number", default=300, type=int)
Parser.add_argument('--FPMTrain_Steps_In_Each_Epoch', help="Step number in each epoch", default=50, type=int)
Parser.add_argument('--FPMTrain_Batch_Size', help="Batch size", default=16, type=int)
Parser.add_argument('--FPMTrain_Learning_Rate', help="Learning rate", default=0.001, type=float)
Parser.add_argument('--FPMTrain_Starting_Channel_Number', help="Starting Channel Number of UNet++", default=32, type=int)
Parser.add_argument('--FPMTrain_Buffer_Size', help="Buffer size", default=16, type=int)
Parser.add_argument('--FPMTrain_Buffer_Exchange_Interval', help="Buffer exchange interval", default=500, type=int)
Parser.add_argument('--FPMTrain_Patch_Shape', help="Patch shape of randomly generated image patches", type=str, default="16,64,64")
Parser.add_argument('--FPMTrain_Deep_Supervision', help="Deep supervision or not in UNet++")

# FPM Prediction Parameters
Parser.add_argument('--FPMPredict_Device_To_Use', help="On which device to run program", default="cuda:0", type=str)
Parser.add_argument('--FPMPredict_Image_Directory', help="Select the image directory")
Parser.add_argument('--FPMPredict_PFM_Directory', help="Select the predicted fluorescence map output directory")
Parser.add_argument('--FPMPredict_ModelFolder_Directory', help="Select the directory to the FPM to use")
Parser.add_argument('--FPMPredict_FPMModel_Name', help="Name of FPM to use", default="NoDS_lr0001_BS16_PS166464_SC32_BufS16_BSI500_SIEE50.pth", type=str)
Parser.add_argument('--FPMPredict_Starting_Channel_Number', help="Starting Channel Number used to train the FPM", default=32, type=int)
Parser.add_argument('--FPMPredict_Patch_Shape', help="Patch shape used to train the FPM", type=str, default="16,64,64")
Parser.add_argument('--FPMPredict_Deep_Supervision', help="Deep supervision or not in UNet++")

# CM Training Parameters
Parser.add_argument('--CMTrain_Device_To_Use', help="On which device to run program", default="cuda:0", type=str)
Parser.add_argument('--CMTrain_PFM_Directory', help="Select the directory of predicted fluorescence map for training")
Parser.add_argument('--CMTrain_Models_Directory', help="Select the directory to store output classification models")
Parser.add_argument('--CMTrain_Batch_Size', help="Batch size", default=32, type=int)
Parser.add_argument('--CMTrain_Max_Epochs', help="Max epoch number", default=120, type=int)
Parser.add_argument('--CMTrain_Learning_Rate', help="Learning rate", default=0.0005, type=float)
Parser.add_argument('--CMTrain_ExpectedZ', help="ow many Z stacks in each 3D PFM to use for CM training", default=14, type=int)
Parser.add_argument('--CMTrain_Class_Names', help="Class names for classfication (needs to be in file name)", default="SN,NSN", type=str)
Parser.add_argument('--CMTrain_Min_Acc', help="Minimum accuracy allowed for CM training", default=0.79, type=float)
Parser.add_argument('--CMTrain_Train_Single_CM', help="Train a single CM")
Parser.add_argument('--CMTrain_Valid_Which_Part', help="Which part of data to use as valiation set for this CM training", default=3, type=int)

# CM Testing Parameters
Parser.add_argument('--CMTest_Device_To_Use', help="On which device to run program", default="cuda:0", type=str)
Parser.add_argument('--CMTest_PFM_Directory', help="Select the directory of predicted fluorescence map for testing")
Parser.add_argument('--CMTest_Models_Directory', help="Select the directory the classification models are stored")
Parser.add_argument('--CMTest_ExpectedZ', help="How many Z stacks in each 3D PFM to use for CM testing", default=15, type=int)
Parser.add_argument('--CMTest_Class_Names', help="Class names for classfication (needs to be in file name)", default="SN,NSN", type=str)
Parser.add_argument('--CMTest_Train_Single_CM', help="Test a single CM")
Parser.add_argument('--CMTest_Valid_Which_Part', help="Which part of data to use as valiation set for this CM testing", default=3, type=int)

# CM Prediction Parameters
Parser.add_argument('--CMPredict_Device_To_Use', help="On which device to run program", default="cuda:0", type=str)
Parser.add_argument('--CMPredict_PFM_Directory', help="Select the directory of predicted fluorescence map for prediction")
Parser.add_argument('--CMPredict_Models_Directory', help="Select the directory of CMs to use for prediction")
Parser.add_argument('--CMPredict_ExpectedZ', help="How many Z stacks in each 3D PFM to use for CM prediction", default=15, type=int)
Parser.add_argument('--CMPredict_Class_Names', help="Class names for classfication (can be in your file name or absent)", default="SN,NSN", type=str)
Parser.add_argument('--CMPredict_Label_In_Name', help="If the class names are in file names the accuracy will be calculated, otherwise only classfication results")

args = Parser.parse_args()

# Run Pipeline
if args.Option == 'FPM_Train':
	FPM_Train_Main(Device_To_Use=args.FPMTrain_Device_To_Use, TrainImage_Directory=args.FPMTrain_TrainImage_Directory, ValidImage_Directory=args.FPMTrain_ValidImage_Directory, LogFolder_Directory=args.FPMTrain_LogFolder_Directory, ModelFolder_Directory=args.FPMTrain_ModelFolder_Directory, Max_Epochs=args.FPMTrain_Max_Epochs, Steps_In_Each_Epoch=args.FPMTrain_Steps_In_Each_Epoch, Batch_Size=args.FPMTrain_Batch_Size, Learning_Rate=args.FPMTrain_Learning_Rate, Starting_Channel_Number=args.FPMTrain_Starting_Channel_Number, Deep_Supervision=args.FPMTrain_Deep_Supervision, Patch_Shape=(int(args.FPMTrain_Patch_Shape.split(",")[0]),int(args.FPMTrain_Patch_Shape.split(",")[1]),int(args.FPMTrain_Patch_Shape.split(",")[2])), Buffer_Size=args.FPMTrain_Buffer_Size, Buffer_Exchange_Interval=args.FPMTrain_Buffer_Exchange_Interval)
elif args.Option == 'FPM_Predict':
	FPM_Predict_Main(Device_To_Use=args.FPMPredict_Device_To_Use, Image_Directory=args.FPMPredict_Image_Directory, PFM_Directory=args.FPMPredict_PFM_Directory, FPM_Model_State=args.FPMPredict_ModelFolder_Directory+args.FPMPredict_FPMModel_Name, Starting_Channel_Number=args.FPMPredict_Starting_Channel_Number, Deep_Supervision=args.FPMPredict_Deep_Supervision, Patch_Shape=(int(args.FPMPredict_Patch_Shape.split(",")[0]),int(args.FPMPredict_Patch_Shape.split(",")[1]),int(args.FPMPredict_Patch_Shape.split(",")[2])))
elif args.Option == 'CM_Train':
	if args.CMTrain_Train_Single_CM:
		Final_Best_Valid_Acc = CM_Train_Main(Device_To_Use=args.CMTrain_Device_To_Use, PFM_Directory=args.CMTrain_PFM_Directory, Models_Directory=args.CMTrain_Models_Directory, Batch_Size=args.CMTrain_Batch_Size, Max_Epochs=args.CMTrain_Max_Epochs, Learning_Rate=args.CMTrain_Learning_Rate, ExpectedZ=args.CMTrain_ExpectedZ, Class_Name_List=args.CMTrain_Class_Names.split(","), Valid_Which_Part=args.CMTrain_Valid_Which_Part)
		while Final_Best_Valid_Acc < args.CMTrain_Min_Acc:
			Final_Best_Valid_Acc = CM_Train_Main(Device_To_Use=args.CMTrain_Device_To_Use, PFM_Directory=args.CMTrain_PFM_Directory, Models_Directory=args.CMTrain_Models_Directory, Batch_Size=args.CMTrain_Batch_Size, Max_Epochs=args.CMTrain_Max_Epochs, Learning_Rate=args.CMTrain_Learning_Rate, ExpectedZ=args.CMTrain_ExpectedZ, Class_Name_List=args.CMTrain_Class_Names.split(","), Valid_Which_Part=args.CMTrain_Valid_Which_Part)
	else:
		Final_Best_Valid_Acc = CM_Train_Main(Device_To_Use=args.CMTrain_Device_To_Use, PFM_Directory=args.CMTrain_PFM_Directory, Models_Directory=args.CMTrain_Models_Directory, Batch_Size=args.CMTrain_Batch_Size, Max_Epochs=args.CMTrain_Max_Epochs, Learning_Rate=args.CMTrain_Learning_Rate, ExpectedZ=args.CMTrain_ExpectedZ, Class_Name_List=args.CMTrain_Class_Names.split(","), Valid_Which_Part=1)
		while Final_Best_Valid_Acc < args.CMTrain_Min_Acc:
			Final_Best_Valid_Acc = CM_Train_Main(Device_To_Use=args.CMTrain_Device_To_Use, PFM_Directory=args.CMTrain_PFM_Directory, Models_Directory=args.CMTrain_Models_Directory, Batch_Size=args.CMTrain_Batch_Size, Max_Epochs=args.CMTrain_Max_Epochs, Learning_Rate=args.CMTrain_Learning_Rate, ExpectedZ=args.CMTrain_ExpectedZ, Class_Name_List=args.CMTrain_Class_Names.split(","), Valid_Which_Part=1)
		Final_Best_Valid_Acc = CM_Train_Main(Device_To_Use=args.CMTrain_Device_To_Use, PFM_Directory=args.CMTrain_PFM_Directory, Models_Directory=args.CMTrain_Models_Directory, Batch_Size=args.CMTrain_Batch_Size, Max_Epochs=args.CMTrain_Max_Epochs, Learning_Rate=args.CMTrain_Learning_Rate, ExpectedZ=args.CMTrain_ExpectedZ, Class_Name_List=args.CMTrain_Class_Names.split(","), Valid_Which_Part=2)
		while Final_Best_Valid_Acc < args.CMTrain_Min_Acc:
			Final_Best_Valid_Acc = CM_Train_Main(Device_To_Use=args.CMTrain_Device_To_Use, PFM_Directory=args.CMTrain_PFM_Directory, Models_Directory=args.CMTrain_Models_Directory, Batch_Size=args.CMTrain_Batch_Size, Max_Epochs=args.CMTrain_Max_Epochs, Learning_Rate=args.CMTrain_Learning_Rate, ExpectedZ=args.CMTrain_ExpectedZ, Class_Name_List=args.CMTrain_Class_Names.split(","), Valid_Which_Part=2)
		Final_Best_Valid_Acc = CM_Train_Main(Device_To_Use=args.CMTrain_Device_To_Use, PFM_Directory=args.CMTrain_PFM_Directory, Models_Directory=args.CMTrain_Models_Directory, Batch_Size=args.CMTrain_Batch_Size, Max_Epochs=args.CMTrain_Max_Epochs, Learning_Rate=args.CMTrain_Learning_Rate, ExpectedZ=args.CMTrain_ExpectedZ, Class_Name_List=args.CMTrain_Class_Names.split(","), Valid_Which_Part=3)
		while Final_Best_Valid_Acc < args.CMTrain_Min_Acc:
			Final_Best_Valid_Acc = CM_Train_Main(Device_To_Use=args.CMTrain_Device_To_Use, PFM_Directory=args.CMTrain_PFM_Directory, Models_Directory=args.CMTrain_Models_Directory, Batch_Size=args.CMTrain_Batch_Size, Max_Epochs=args.CMTrain_Max_Epochs, Learning_Rate=args.CMTrain_Learning_Rate, ExpectedZ=args.CMTrain_ExpectedZ, Class_Name_List=args.CMTrain_Class_Names.split(","), Valid_Which_Part=3)
elif args.Option == 'CM_Test':
	if args.CMTest_Train_Single_CM:
		CM_TestValid_Main(Device_To_Use=args.CMTest_Device_To_Use, PFM_Directory=args.CMTest_PFM_Directory, Models_Directory=args.CMTest_Models_Directory, ExpectedZ=args.CMTest_ExpectedZ, Class_Name_List=args.CMTest_Class_Names.split(","), Valid_Which_Part=args.CMTest_Valid_Which_Part)
	else:
		CM_TestValid_Main(Device_To_Use=args.CMTest_Device_To_Use, PFM_Directory=args.CMTest_PFM_Directory, Models_Directory=args.CMTest_Models_Directory, ExpectedZ=args.CMTest_ExpectedZ, Class_Name_List=args.CMTest_Class_Names.split(","), Valid_Which_Part=1)
		CM_TestValid_Main(Device_To_Use=args.CMTest_Device_To_Use, PFM_Directory=args.CMTest_PFM_Directory, Models_Directory=args.CMTest_Models_Directory, ExpectedZ=args.CMTest_ExpectedZ, Class_Name_List=args.CMTest_Class_Names.split(","), Valid_Which_Part=2)
		CM_TestValid_Main(Device_To_Use=args.CMTest_Device_To_Use, PFM_Directory=args.CMTest_PFM_Directory, Models_Directory=args.CMTest_Models_Directory, ExpectedZ=args.CMTest_ExpectedZ, Class_Name_List=args.CMTest_Class_Names.split(","), Valid_Which_Part=3)
elif args.Option == 'CM_Predict':
	CM_Predict_New_Images_Main(Device_To_Use=args.CMPredict_Device_To_Use, PFM_Directory=args.CMPredict_PFM_Directory, Models_Directory=args.CMPredict_Models_Directory, ExpectedZ=args.CMPredict_ExpectedZ, Class_Name_List=args.CMPredict_Class_Names.split(","), Label_In_Name=args.CMPredict_Label_In_Name)
