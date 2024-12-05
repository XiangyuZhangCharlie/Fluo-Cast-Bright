##################################################
# Predict with florescence prediction network torch
##################################################

import os
import math
import torch
import argparse
import numpy as np
from skimage import io
from Networks_FPM import Nested_UNet_3D
from PredictFunctions_FPM import Predict_With_Patch_Batch

# Load arguments

def FPM_Predict_Main(Device_To_Use, Image_Directory, PFM_Directory, FPM_Model_State, Starting_Channel_Number, Deep_Supervision, Patch_Shape):

	Device = torch.device(Device_To_Use)

	# Load model

	Nested_UNet_3D_Model = Nested_UNet_3D(Starting_Channel=Starting_Channel_Number, Deep_Supervision_Switch=Deep_Supervision)
	Nested_UNet_3D_Model.load_state_dict(torch.load(FPM_Model_State,map_location=Device))
	Nested_UNet_3D_Model.to(device=Device)

	# Get image list

	Original_Image_List = os.listdir(Image_Directory)
	for Original_Image_Number in range(0,len(Original_Image_List)):
		Original_Image_List[Original_Image_Number] = Image_Directory + Original_Image_List[Original_Image_Number]

	# Generate FPMs

	Nested_UNet_3D_Model.eval()
	with torch.no_grad():
		for Image_Count, Current_Original_Image in enumerate(Original_Image_List):
			Current_Image_Loaded = io.imread(Current_Original_Image)
			if len(Current_Image_Loaded.shape) == 4:
				Current_Image_Loaded = Current_Image_Loaded[0,]
			Preprocessed_Image = np.expand_dims(np.expand_dims(Current_Image_Loaded,axis=0),axis=0)
			Half_Difference_To_16 = int((Preprocessed_Image.shape[2] - 16) / 2)
			Preprocessed_Image = torch.from_numpy(Preprocessed_Image[:,:,Half_Difference_To_16:Half_Difference_To_16+16,:,:])
			Preprocessed_Image = Preprocessed_Image.to(device=Device, dtype=torch.float)
			Predicted_Fluorescence_Map = Predict_With_Patch_Batch(Nested_UNet_3D_Model,Preprocessed_Image,Device)
			if Deep_Supervision:
				Fluorescence_Map_ToSave = Predicted_Fluorescence_Map[3].to(device="cpu").numpy()
			else:
				Fluorescence_Map_ToSave = Predicted_Fluorescence_Map.to(device="cpu").numpy()
			io.imsave(".".join(Current_Original_Image.split(".")[:-1]).replace(Image_Directory,PFM_Directory) + "_PFM.tiff", Fluorescence_Map_ToSave[0,0,:,:,:])
			print("Predicted " + ".".join(Current_Original_Image.split(".")[:-1]).replace(Image_Directory,"")+"_PFM.tiff")
