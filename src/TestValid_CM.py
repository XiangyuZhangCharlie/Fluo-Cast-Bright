##################################################
# Test with classification network on valid
##################################################

import os
import torch
import argparse
import numpy as np
from skimage import io
from Dataset_CM import Image_Set_Generator, Obtain_Image_Label_List_TrainTest
from NetworkUtils_CM import Full_Classification_Network, Calculate_Classification_Accuracy

def CM_TestValid_Main(Device_To_Use, PFM_Directory, Models_Directory, ExpectedZ, Class_Name_List, Valid_Which_Part):

	Device = torch.device(Device_To_Use)

	# Load data

	Train_Image_Stacks, Valid_Image_Stacks, Labels = Obtain_Image_Label_List_TrainTest(PFM_Directory, Models_Directory, ExpectedZ, Class_Name_List, Valid_Which_Part)

	# Load model

	Saved_CN_Model = Full_Classification_Network(Number_Of_Classes=len(Class_Name_List))
	Saved_CN_Model.load_state_dict(torch.load(Models_Directory + "CM_TrainValidSplit_" + str(Valid_Which_Part) + "_Model/PFM_CM.pth", map_location=Device))
	Saved_CN_Model.to(device=Device)
	Saved_CN_Model.eval()

	# Predict 2D images in each 3D stack

	Image_Stack_Classification_Score_Dict = {}
	for Valid_Image_Name in Valid_Image_Stacks:
		Valid_Image_Name_As_List = Valid_Image_Name.split("_")
		Z_Number_In_Valid_Image_Name = int(Valid_Image_Name_As_List[-1])
		Ori_Name_In_Valid_Image_Name = "_".join(Valid_Image_Name_As_List[:-1])
		if Ori_Name_In_Valid_Image_Name not in Image_Stack_Classification_Score_Dict:
			Image_Stack_Classification_Score_Dict[Ori_Name_In_Valid_Image_Name] = {}
			for Class_Name in Class_Name_List:
				Image_Stack_Classification_Score_Dict[Ori_Name_In_Valid_Image_Name][Class_Name] = 0
		Image_Stack = io.imread(Ori_Name_In_Valid_Image_Name)
		Current_Image_2D_Section = Image_Stack[Z_Number_In_Valid_Image_Name,:,:]
		Current_Image_2D_Section = np.expand_dims(Current_Image_2D_Section, axis=0)
		Current_Image_2D_Section = np.expand_dims(Current_Image_2D_Section, axis=0)
		Current_Image_2D_Section_As3Channel = np.concatenate((Current_Image_2D_Section,Current_Image_2D_Section,Current_Image_2D_Section),axis=1)
		Current_Image_2D_Section_As3Channel = torch.tensor(Current_Image_2D_Section_As3Channel, device=torch.device(Device))
		with torch.no_grad():
			Current_Image_2D_Section_Prediction = Saved_CN_Model(Current_Image_2D_Section_As3Channel)
		for Class_Name_Index in range(0,len(Class_Name_List)):
			if float(max(Current_Image_2D_Section_Prediction[0])) == float(Current_Image_2D_Section_Prediction[0][Class_Name_Index]):
				Image_Stack_Classification_Score_Dict[Ori_Name_In_Valid_Image_Name][Class_Name_List[Class_Name_Index]] += 1
				continue

	Image_Stack_Classification_Answer_Dict = {}
	Image_Stack_Classification_Correct_Number = 0
	Image_Stack_Total_Number = len(Image_Stack_Classification_Score_Dict)
	for Image_Stack_Name, Class_Score_Dict in Image_Stack_Classification_Score_Dict.items():
		Score_List_For_Score_Comparison = []
		for Class_Score in Class_Score_Dict.values():
			Score_List_For_Score_Comparison.append(Class_Score)
		Final_Classification = Class_Name_List[Score_List_For_Score_Comparison.index(max(Score_List_For_Score_Comparison))]
		Image_Stack_Classification_Answer_Dict[Image_Stack_Name] = Final_Classification
		if ("_" + Final_Classification + "_") in Image_Stack_Name:
			Image_Stack_Classification_Correct_Number += 1

	Result_Output_File = open(Models_Directory + "CM_TrainValidSplit_" + str(Valid_Which_Part) + "_Logs/Valid_Results.txt", "w")
	Result_Output_File.write(str(Image_Stack_Classification_Answer_Dict)+"\n")
	Result_Output_File.write("Classification Accuracy: " + str(float(Image_Stack_Classification_Correct_Number) / float(Image_Stack_Total_Number)))
	Result_Output_File.close()
