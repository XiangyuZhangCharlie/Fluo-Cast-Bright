##################################################
# Train classification network
##################################################

import os
import torch
import argparse
import numpy as np
from Dataset_CM import Image_Set_Generator, Obtain_Image_Label_List_TrainTest
from NetworkUtils_CM import Full_Classification_Network, Calculate_Classification_Accuracy

def CM_Train_Main(Device_To_Use, PFM_Directory, Models_Directory, Batch_Size, Max_Epochs, Learning_Rate, ExpectedZ, Class_Name_List, Valid_Which_Part):

	Device = torch.device(Device_To_Use)
	if not os.path.exists(Models_Directory + "CM_TrainValidSplit_"+str(Valid_Which_Part)+"_Logs"):
		os.makedirs(Models_Directory + "CM_TrainValidSplit_"+str(Valid_Which_Part)+"_Logs")
	if not os.path.exists(Models_Directory + "CM_TrainValidSplit_"+str(Valid_Which_Part)+"_Model"):
		os.makedirs(Models_Directory + "CM_TrainValidSplit_"+str(Valid_Which_Part)+"_Model")

	# Load data

	Train_Image_Stacks, Valid_Image_Stacks, Labels = Obtain_Image_Label_List_TrainTest(PFM_Directory, Models_Directory, ExpectedZ, Class_Name_List, Valid_Which_Part)
	Train_Set = Image_Set_Generator(Train_Image_Stacks, Labels, Number_Of_Classes=len(Class_Name_List), Random_Rotate=True)
	Train_Generator = torch.utils.data.DataLoader(Train_Set, batch_size=Batch_Size, shuffle=True)
	Valid_Set = Image_Set_Generator(Valid_Image_Stacks, Labels, Number_Of_Classes=len(Class_Name_List), Random_Rotate=False)
	Valid_Generator = torch.utils.data.DataLoader(Valid_Set, batch_size=Batch_Size, shuffle=True)

	# Compile and run model

	Classification_Network = Full_Classification_Network(Number_Of_Classes=len(Class_Name_List))
	Optimizer = torch.optim.Adam(Classification_Network.parameters(),lr=Learning_Rate)
	Error_Calculation = torch.nn.BCELoss()
	Classification_Network.to(device=Device)

	# Epoch loop

	Best_Valid_Acc = 0.0
	Log_File = open(Models_Directory + "CM_TrainValidSplit_"+str(Valid_Which_Part)+"_Logs/Training_Log.txt","w")
	Log_File.write("Model Training Output")
	Log_File.close()

	for Epoch_Count in range(Max_Epochs):
		Epoch_Count += 1
		Total_Valid_Loss = 0.0
		Total_Valid_Acc = 0.0
		Total_Valid_Batch_Number = 0.0
		Log_File = open(Models_Directory + "CM_TrainValidSplit_"+str(Valid_Which_Part)+"_Logs/Training_Log.txt","a")

		Classification_Network.train()
		Current_Step = 0
		for Current_Batch_Images, Current_Batch_Labels in Train_Generator:
			Current_Batch_Images, Current_Batch_Labels = Current_Batch_Images.to(device=Device, dtype=torch.float), Current_Batch_Labels.to(device=Device, dtype=torch.float)
			Optimizer.zero_grad()
			Current_Batch_Predicted_Labels = Classification_Network(Current_Batch_Images)
			Loss = Error_Calculation(Current_Batch_Predicted_Labels,Current_Batch_Labels)
			Acc = Calculate_Classification_Accuracy(Current_Batch_Predicted_Labels,Current_Batch_Labels,Class_Name_List)
			Loss.backward()
			Optimizer.step()
			Current_Step += 1
			Log_File.write("\nEpoch: " + str(Epoch_Count) + "\tStep: " + str(Current_Step) + "\tTraining Loss: " + str(Loss.item()) + "\tTraining Acc: " + str(Acc))
			print("Epoch: " + str(Epoch_Count) + "\tStep: " + str(Current_Step) + "\tTraining Loss: " + str(Loss.item()) + "\tTraining Acc: " + str(Acc))

		Classification_Network.eval()
		with torch.no_grad():
			for Current_Batch_Images, Current_Batch_Labels in Valid_Generator:
				Current_Batch_Images, Current_Batch_Labels = Current_Batch_Images.to(device=Device, dtype=torch.float), Current_Batch_Labels.to(device=Device, dtype=torch.float)
				Current_Batch_Predicted_Labels = Classification_Network(Current_Batch_Images)
				Loss = Error_Calculation(Current_Batch_Predicted_Labels,Current_Batch_Labels)
				Acc = Calculate_Classification_Accuracy(Current_Batch_Predicted_Labels,Current_Batch_Labels,Class_Name_List)
				Total_Valid_Loss += Loss.item()
				Total_Valid_Acc += Acc
				Total_Valid_Batch_Number += 1.0

		Average_Valid_Loss = Total_Valid_Loss / Total_Valid_Batch_Number
		Average_Valid_Acc = Total_Valid_Acc / Total_Valid_Batch_Number
		Log_File.write("\nEpoch: " + str(Epoch_Count) + "\tValidation loss: " + str(Average_Valid_Loss) + "\tValidation Acc: " + str(Average_Valid_Acc))
		print("Epoch: " + str(Epoch_Count) + "\tValidation loss: " + str(Average_Valid_Loss) + "\tValidation Acc: " + str(Average_Valid_Acc))

		if Average_Valid_Acc > Best_Valid_Acc:
			Log_File.write("\nModel saved due to better validation loss at epoch " + str(Epoch_Count))
			print("Model saved due to better validation loss at epoch " + str(Epoch_Count))
			torch.save(Classification_Network.state_dict(), Models_Directory + "CM_TrainValidSplit_" + str(Valid_Which_Part) + "_Model/PFM_CM.pth")
			Best_Valid_Acc = Average_Valid_Acc

		Log_File.close()
		
	return Best_Valid_Acc
