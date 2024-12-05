##################################################
# Dataset to provide data label for training
##################################################

import os
import math
import torch
import numpy as np
from skimage import io
from random import random

class Image_Set_Generator(torch.utils.data.Dataset):
	def __init__(self, Image_Name_List, Labels, Number_Of_Classes, Random_Rotate):
		# Initialization
		self.List_Of_Images = Image_Name_List
		self.Labels = Labels
		self.Number_Of_Classes = Number_Of_Classes
		self.Random_Rotate = Random_Rotate

	def __len__(self):
		# Total Number of Images
		return len(self.List_Of_Images)

	def __getitem__(self, index):
		ID = self.List_Of_Images[index]

		# Store sample
		ID_As_List = ID.split("_")
		Z_Number_In_ID = int(ID_As_List[-1])
		Ori_Name_In_ID = "_".join(ID_As_List[:-1])
		Image_Stack = io.imread(Ori_Name_In_ID)
		Current_Image_2D_Section = Image_Stack[Z_Number_In_ID,:,:]
		if self.Random_Rotate:
			Random_Number_For_Rotation = random()
			if Random_Number_For_Rotation <= 0.25:
				Current_Image_2D_Section = np.rot90(Current_Image_2D_Section,3)
			elif Random_Number_For_Rotation > 0.25 and Random_Number_For_Rotation <= 0.5:
				Current_Image_2D_Section = np.rot90(Current_Image_2D_Section,2)
			elif Random_Number_For_Rotation > 0.5 and Random_Number_For_Rotation <= 0.75:
				Current_Image_2D_Section = np.rot90(Current_Image_2D_Section,1)
		Current_Image_2D_Section = np.expand_dims(Current_Image_2D_Section, axis=0)
		Current_Image_2D_Section_As3Channel = np.concatenate((Current_Image_2D_Section,Current_Image_2D_Section,Current_Image_2D_Section),axis=0)
		X = Current_Image_2D_Section_As3Channel.copy()

		# Store class
		y = np.eye(self.Number_Of_Classes, dtype='uint8')[self.Labels[ID]]

		return X, y

def Obtain_Image_Label_List_TrainTest(PFM_Directory, Models_Directory, ExpectedZ, Class_Name_List, Valid_Which_Part):
	CM_PFM_Directory = PFM_Directory
	Image_List = os.listdir(CM_PFM_Directory)
	for Image_Number in range(0,len(Image_List)):
		Image_List[Image_Number] = CM_PFM_Directory + Image_List[Image_Number]
	Image_List = sorted(Image_List)

	Image_Dict_ByClass = {}
	for Class_Name in Class_Name_List:
		Image_Dict_ByClass[Class_Name] = []
	Label_Dict = {}

	for Image_Name in Image_List:
		for Class_Name in Class_Name_List:
			if ("_" + Class_Name + "_") in Image_Name:
				Image_Dict_ByClass[Class_Name].append(Image_Name)

	All_Train_3D_Images = []
	All_Valid_3D_Images = []
	for Class_Name, Images_In_Class in Image_Dict_ByClass.items():
		Total_Number_Images_In_Class = len(Images_In_Class)
		Number_Images_One_Third = int(Total_Number_Images_In_Class/3)
		Valid_Images_In_Class_List = Images_In_Class[Number_Images_One_Third*(Valid_Which_Part-1):Number_Images_One_Third*Valid_Which_Part]
		Train_Images_In_Class_List = []
		for Single_Image_Name in Images_In_Class:
			if Single_Image_Name not in Valid_Images_In_Class_List:
				Train_Images_In_Class_List.append(Single_Image_Name)
		All_Train_3D_Images += Train_Images_In_Class_List
		All_Valid_3D_Images += Valid_Images_In_Class_List

	All_Train_2D_Images = []
	All_Valid_2D_Images = []
	for Single_Train_3D_Image_Name in All_Train_3D_Images:
		Image_Stack = io.imread(Single_Train_3D_Image_Name)
		Total_Z_Length = int(Image_Stack.shape[0])
		Difference_To_ExpectedZ = Total_Z_Length - ExpectedZ
		Start_Location = int(Difference_To_ExpectedZ/2)
		for Current_Image_2D_Section_Count in range(0,ExpectedZ):
			Current_Image_2D_Section_Number = Start_Location + Current_Image_2D_Section_Count
			All_Train_2D_Images.append(Single_Train_3D_Image_Name + "_" + str(Current_Image_2D_Section_Number))
			for Class_Name in Class_Name_List:
				if ("_" + Class_Name + "_") in Single_Train_3D_Image_Name:
					Label_Dict[Single_Train_3D_Image_Name + "_" + str(Current_Image_2D_Section_Number)] = Class_Name_List.index(Class_Name)
	for Single_Valid_3D_Image_Name in All_Valid_3D_Images:
		Image_Stack = io.imread(Single_Valid_3D_Image_Name)
		Total_Z_Length = int(Image_Stack.shape[0])
		Difference_To_ExpectedZ = Total_Z_Length - ExpectedZ
		Start_Location = int(Difference_To_ExpectedZ/2)
		for Current_Image_2D_Section_Count in range(0,ExpectedZ):
			Current_Image_2D_Section_Number = Start_Location + Current_Image_2D_Section_Count
			All_Valid_2D_Images.append(Single_Valid_3D_Image_Name + "_" + str(Current_Image_2D_Section_Number))
			for Class_Name in Class_Name_List:
				if ("_" + Class_Name + "_") in Single_Valid_3D_Image_Name:
					Label_Dict[Single_Valid_3D_Image_Name + "_" + str(Current_Image_2D_Section_Number)] = Class_Name_List.index(Class_Name)

	Data_Used_Log_File = open(Models_Directory+"CM_TrainValidSplit_" + str(Valid_Which_Part) + "_Logs/Data_Used_Log.txt","w")
	Data_Used_Log_File.write("3D images in training set:\n")
	Data_Used_Log_File.write(str(All_Train_3D_Images)+str(len(All_Train_3D_Images))+"\n\n\n\n\n")
	Data_Used_Log_File.write("3D images in validation set:\n")
	Data_Used_Log_File.write(str(All_Valid_3D_Images)+str(len(All_Valid_3D_Images))+"\n\n\n\n\n")
	Data_Used_Log_File.write("2D images in training set:\n")
	Data_Used_Log_File.write(str(All_Train_2D_Images)+str(len(All_Train_2D_Images))+"\n\n\n\n\n")
	Data_Used_Log_File.write("2D images in validation set:\n")
	Data_Used_Log_File.write(str(All_Valid_2D_Images)+str(len(All_Valid_2D_Images))+"\n\n\n\n\n")
	Data_Used_Log_File.write("Label Dictionary:\n")
	Data_Used_Log_File.write(str(Label_Dict))

	return All_Train_2D_Images, All_Valid_2D_Images, Label_Dict
