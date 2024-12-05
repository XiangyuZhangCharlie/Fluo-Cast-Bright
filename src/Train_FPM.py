##################################################
# Train florescence prediction network
##################################################

import os
import math
import torch
import argparse
import numpy as np
from Networks_FPM import Nested_UNet_3D
from PredictFunctions_FPM import Predict_With_Patch_Batch
from Dataset_FPM import Cell_Image_Dataset, Buffer_Patch_Dataset, Image_Dataset_For_Valid

def FPM_Train_Main(Device_To_Use, TrainImage_Directory, ValidImage_Directory, LogFolder_Directory, ModelFolder_Directory, Max_Epochs, Steps_In_Each_Epoch, Batch_Size, Learning_Rate, Starting_Channel_Number, Deep_Supervision, Patch_Shape, Buffer_Size, Buffer_Exchange_Interval):

	Device = torch.device(Device_To_Use)

	# Load generators

	Train_Image_List = os.listdir(TrainImage_Directory)
	for Train_Image_Number in range(0,len(Train_Image_List)):
		Train_Image_List[Train_Image_Number] = TrainImage_Directory+Train_Image_List[Train_Image_Number]
	Valid_Image_List = os.listdir(ValidImage_Directory)
	for Valid_Image_Number in range(0,len(Valid_Image_List)):
		Valid_Image_List[Valid_Image_Number] = ValidImage_Directory+Valid_Image_List[Valid_Image_Number]
	Train_Set = Cell_Image_Dataset(Train_Image_List)
	Train_Generator = Buffer_Patch_Dataset(Original_Dataset=Train_Set, Patch_Shape=Patch_Shape, Buffer_Size=Buffer_Size, Buffer_Exchange_Interval=Buffer_Exchange_Interval, Shuffle=True)
	Valid_Set = Image_Dataset_For_Valid(Valid_Image_List)
	Valid_Generator = torch.utils.data.DataLoader(Valid_Set, batch_size=Batch_Size, shuffle=True)

	# Compile and run model

	Nested_UNet_3D_Model = Nested_UNet_3D(Starting_Channel=Starting_Channel_Number, Deep_Supervision_Switch=Deep_Supervision)
	Optimizer = torch.optim.Adam(Nested_UNet_3D_Model.parameters(),lr=Learning_Rate)
	Error_Calculation = torch.nn.MSELoss()
	Nested_UNet_3D_Model.to(device=Device)

	# Epoch Loop

	Best_Valid_Loss = 10000
	if Deep_Supervision:
		Model_Name = "DS_lr" + str(Learning_Rate).replace(".","") + "_BS" + str(Batch_Size) + "_PS" + str(Patch_Shape).replace(",","").replace("(","").replace(")","").replace(" ","") + "_SC" + str(Starting_Channel_Number) + "_BufS" + str(Buffer_Size) + "_BSI" + str(Buffer_Exchange_Interval) + "_SIEE" + str(Steps_In_Each_Epoch)
	else:
		Model_Name = "NoDS_lr" + str(Learning_Rate).replace(".","") + "_BS" + str(Batch_Size) + "_PS" + str(Patch_Shape).replace(",","").replace("(","").replace(")","").replace(" ","") + "_SC" + str(Starting_Channel_Number) + "_BufS" + str(Buffer_Size) + "_BSI" + str(Buffer_Exchange_Interval) + "_SIEE" + str(Steps_In_Each_Epoch)
	Log_File = open(LogFolder_Directory+Model_Name+"_Training_Log.txt","w")
	Log_File.write("Train with parameters: " + Model_Name)
	Log_File.close()

	for Epoch_Count in range(Max_Epochs):
		Epoch_Count += 1
		Total_Valid_Loss = 0.0
		Total_Valid_Batch_Number = 0.0
		Log_File = open(LogFolder_Directory+Model_Name+"_Training_Log.txt","a")

		Nested_UNet_3D_Model.train()
		for Step_Count in range(Steps_In_Each_Epoch):
			Step_Count += 1
			Current_Batch_Images, Current_Batch_Labels = Train_Generator.Get_Patch_Batch(Batch_Size)
			Current_Batch_Images, Current_Batch_Labels = Current_Batch_Images.to(device=Device, dtype=torch.float32), Current_Batch_Labels.to(device=Device, dtype=torch.float32)
			Optimizer.zero_grad()
			Current_Batch_Predicted_Labels = Nested_UNet_3D_Model(Current_Batch_Images)
			if Deep_Supervision:
				Loss_1 = Error_Calculation(Current_Batch_Predicted_Labels[0],Current_Batch_Labels)
				Loss_2 = Error_Calculation(Current_Batch_Predicted_Labels[1],Current_Batch_Labels)
				Loss_3 = Error_Calculation(Current_Batch_Predicted_Labels[2],Current_Batch_Labels)
				Loss_4 = Error_Calculation(Current_Batch_Predicted_Labels[3],Current_Batch_Labels)
				Loss = Loss_1 + Loss_2 + Loss_3 + Loss_4
				Current_Step_Loss = Loss.item() / 4.0
			else:
				Loss = Error_Calculation(Current_Batch_Predicted_Labels,Current_Batch_Labels)
				Current_Step_Loss = Loss.item()
			Loss.backward()
			Optimizer.step()
			Log_File.write("\nEpoch: " + str(Epoch_Count) + "\tStep: " + str(Step_Count) + "\tTraining loss: " + str(Current_Step_Loss))
			print("Epoch: " + str(Epoch_Count) + "\tStep: " + str(Step_Count) + "\tTraining loss: " + str(Current_Step_Loss))

		Nested_UNet_3D_Model.eval()
		with torch.no_grad():
			for Current_Batch_Images, Current_Batch_Labels in Valid_Generator:
				Current_Batch_Images, Current_Batch_Labels = Current_Batch_Images.to(device=Device, dtype=torch.float32), Current_Batch_Labels.to(device=Device, dtype=torch.float32)
				Current_Batch_Predicted_Labels = Predict_With_Patch_Batch(Nested_UNet_3D_Model,Current_Batch_Images,Device)
				Loss = Error_Calculation(Current_Batch_Predicted_Labels,Current_Batch_Labels)
				Total_Valid_Loss += Loss.item()
				Total_Valid_Batch_Number += 1.0

		Average_Valid_Loss = Total_Valid_Loss / Total_Valid_Batch_Number
		Log_File.write("\nEpoch: " + str(Epoch_Count) + "\tValidation loss: " + str(Average_Valid_Loss))
		print("Epoch: " + str(Epoch_Count) + "\tValidation loss: " + str(Average_Valid_Loss))

		if Average_Valid_Loss < Best_Valid_Loss:
			Log_File.write("\nModel saved due to better validation loss at epoch " + str(Epoch_Count))
			print("Model saved due to better validation loss at epoch " + str(Epoch_Count))
			torch.save(Nested_UNet_3D_Model.state_dict(), ModelFolder_Directory + Model_Name + ".pth")
			Best_Valid_Loss = Average_Valid_Loss

		Log_File.close()
