##################################################
# Predict new with classification network
##################################################

import os
import torch
import argparse
import numpy as np
from skimage import io
from NetworkUtils_CM import Full_Classification_Network, Calculate_Classification_Accuracy

def CM_Predict_New_Images_Main(Device_To_Use, PFM_Directory, Models_Directory, ExpectedZ, Class_Name_List, Label_In_Name):

	Device = torch.device(Device_To_Use)

	# Load data

	NewFPM_List = os.listdir(PFM_Directory)
	for NewFPM_Number in range(0, len(NewFPM_List)):
		if "Prediction_Results" not in NewFPM_List[NewFPM_Number]:
			NewFPM_List[NewFPM_Number] = PFM_Directory + NewFPM_List[NewFPM_Number]

	# Load model

	Saved_CN_Model_1 = Full_Classification_Network(Number_Of_Classes=len(Class_Name_List))
	Saved_CN_Model_1.load_state_dict(torch.load(Models_Directory + "CM_TrainValidSplit_1_Model/PFM_CM.pth", map_location=Device))
	Saved_CN_Model_1.to(device=Device)
	Saved_CN_Model_2 = Full_Classification_Network(Number_Of_Classes=len(Class_Name_List))
	Saved_CN_Model_2.load_state_dict(torch.load(Models_Directory + "CM_TrainValidSplit_2_Model/PFM_CM.pth", map_location=Device))
	Saved_CN_Model_2.to(device=Device)
	Saved_CN_Model_3 = Full_Classification_Network(Number_Of_Classes=len(Class_Name_List))
	Saved_CN_Model_3.load_state_dict(torch.load(Models_Directory + "CM_TrainValidSplit_3_Model/PFM_CM.pth", map_location=Device))
	Saved_CN_Model_3.to(device=Device)
	Saved_CN_Model_1.eval()
	Saved_CN_Model_2.eval()
	Saved_CN_Model_3.eval()

	# Predict 2D images in each 3D stack

	Image_Stack_Classification_Score_Dict = {}
	for NewFPM_Image_Name in NewFPM_List:
		if NewFPM_Image_Name not in Image_Stack_Classification_Score_Dict:
			Image_Stack_Classification_Score_Dict[NewFPM_Image_Name] = {}
			for Class_Name in Class_Name_List:
				Image_Stack_Classification_Score_Dict[NewFPM_Image_Name][Class_Name] = 0
		Image_Stack = io.imread(NewFPM_Image_Name)
		for Z_Number_In_NewFPM in range(0,Image_Stack.shape[0]):
			Current_Image_2D_Section = Image_Stack[Z_Number_In_NewFPM,:,:]
			Current_Image_2D_Section = np.expand_dims(Current_Image_2D_Section, axis=0)
			Current_Image_2D_Section = np.expand_dims(Current_Image_2D_Section, axis=0)
			Current_Image_2D_Section_As3Channel = np.concatenate((Current_Image_2D_Section,Current_Image_2D_Section,Current_Image_2D_Section),axis=1)
			Current_Image_2D_Section_As3Channel = torch.tensor(Current_Image_2D_Section_As3Channel, device=torch.device(Device))
			with torch.no_grad():
				Current_Image_2D_Section_Prediction_1 = Saved_CN_Model_1(Current_Image_2D_Section_As3Channel)
			with torch.no_grad():
				Current_Image_2D_Section_Prediction_2 = Saved_CN_Model_2(Current_Image_2D_Section_As3Channel)
			with torch.no_grad():
				Current_Image_2D_Section_Prediction_3 = Saved_CN_Model_3(Current_Image_2D_Section_As3Channel)
			Current_Image_2D_Section_Prediction_Average = []
			for Class_Name_Index in range(0,len(Class_Name_List)):
				Current_Image_2D_Section_Prediction_Average.append((float(Current_Image_2D_Section_Prediction_1[0][Class_Name_Index]) + float(Current_Image_2D_Section_Prediction_2[0][Class_Name_Index]) + float(Current_Image_2D_Section_Prediction_3[0][Class_Name_Index])) / 3)
			for Class_Name_Index in range(0,len(Class_Name_List)):
				if float(max(Current_Image_2D_Section_Prediction_Average)) == float(Current_Image_2D_Section_Prediction_Average[Class_Name_Index]):
					Image_Stack_Classification_Score_Dict[NewFPM_Image_Name][Class_Name_List[Class_Name_Index]] += 1
					continue

	Image_Stack_Classification_Answer_Dict = {}
	if Label_In_Name:
		Image_Stack_Classification_Correct_Number = 0
		Image_Stack_Total_Number = len(Image_Stack_Classification_Score_Dict)
	for Image_Stack_Name, Class_Score_Dict in Image_Stack_Classification_Score_Dict.items():
		Score_List_For_Score_Comparison = []
		for Class_Score in Class_Score_Dict.values():
			Score_List_For_Score_Comparison.append(Class_Score)
		Final_Classification = Class_Name_List[Score_List_For_Score_Comparison.index(max(Score_List_For_Score_Comparison))]
		Image_Stack_Classification_Answer_Dict[Image_Stack_Name] = Final_Classification
		if Label_In_Name:
			if ("_" + Final_Classification + "_") in Image_Stack_Name:
				Image_Stack_Classification_Correct_Number += 1

	Result_Output_File = open(PFM_Directory + "Prediction_Results.txt", "w")
	Result_Output_File.write(str(Image_Stack_Classification_Answer_Dict)+"\n")
	if Label_In_Name:
		Result_Output_File.write("Classification Accuracy: " + str(float(Image_Stack_Classification_Correct_Number) / float(Image_Stack_Total_Number)))
	Result_Output_File.close()
