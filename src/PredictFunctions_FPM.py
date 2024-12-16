##################################################
# Predict functions
##################################################

import torch
import numpy as np
try:
	from scipy.signal import triang
except:
	from scipy.signal.windows import triang

def Flip_Y(Input_Array):
	return np.flip(Input_Array, axis=-2)

def Flip_X(Input_Array):
	return np.flip(Input_Array, axis=-1)

def Predict_Single_With_TTA(Predictor_Model,Input_Current_Image_AsArray, Original_Device):
	Input_Current_Image_Tensor_Unsqueezed = torch.unsqueeze(torch.tensor(Input_Current_Image_AsArray), 0)
	Input_Current_Image_AsArray_Unsqueezed = Input_Current_Image_Tensor_Unsqueezed.numpy().astype(np.float32)
	Augmentations = [None, [Flip_Y], [Flip_X], [Flip_Y, Flip_X]]
	Output_Current_Label_AsArray_Unsqueezed_Aug_Mean = None

	for Augmentation in Augmentations:
		# Apply Augmentation
		Input_Current_Image_AsArray_Unsqueezed_Aug = Input_Current_Image_AsArray_Unsqueezed.copy()
		if Augmentation is not None:
			for Transform in Augmentation:
				Input_Current_Image_AsArray_Unsqueezed_Aug = Transform(Input_Current_Image_AsArray_Unsqueezed_Aug)

		# Do prediction
		Input_Current_Image_AsArray_Unsqueezed_Aug = torch.tensor(Input_Current_Image_AsArray_Unsqueezed_Aug.copy(), dtype=torch.float32, device=Original_Device)
		Predictor_Model.eval()
		with torch.no_grad():
			Output_Current_Label_AsArray_Unsqueezed_Aug = Predictor_Model(Input_Current_Image_AsArray_Unsqueezed_Aug)
		if type(Output_Current_Label_AsArray_Unsqueezed_Aug) == tuple:
			Output_Current_Label_AsArray_Unsqueezed_Aug = Output_Current_Label_AsArray_Unsqueezed_Aug[-1]
		Output_Current_Label_AsArray_Unsqueezed_Aug = Output_Current_Label_AsArray_Unsqueezed_Aug.cpu().numpy()

		# Put image back to original state
		if Augmentation is not None:
			for Transform in Augmentation:
				Output_Current_Label_AsArray_Unsqueezed_Aug = Transform(Output_Current_Label_AsArray_Unsqueezed_Aug)
		if Output_Current_Label_AsArray_Unsqueezed_Aug_Mean is None:
			Output_Current_Label_AsArray_Unsqueezed_Aug_Mean = np.zeros(Output_Current_Label_AsArray_Unsqueezed_Aug.shape, dtype=np.float32)

		Output_Current_Label_AsArray_Unsqueezed_Aug_Mean += Output_Current_Label_AsArray_Unsqueezed_Aug

	Output_Current_Label_AsArray_Unsqueezed_Aug_Mean /= len(Augmentations)
	Final_Current_Label_Output_AsArray = torch.tensor(Output_Current_Label_AsArray_Unsqueezed_Aug_Mean, dtype=torch.float32, device=torch.device("cpu")).squeeze(0)
	return Final_Current_Label_Output_AsArray

def Predict_With_Patch_Batch(Predictor_Model, Input_Batch_Images_Tensor, Original_Device):
	Input_Batch_Images_AsArray = Input_Batch_Images_Tensor.cpu().numpy().astype(np.float32)
	Output_Batch_Labels_AsArray = np.zeros(Input_Batch_Images_AsArray.shape)

	for Input_Current_Image_Number in range(0,Input_Batch_Images_AsArray.shape[0]):
		Input_Current_Image_AsArray = Input_Batch_Images_AsArray[Input_Current_Image_Number]
		# Predict single with test time augmentation
		Output_Current_Label_AsArray = Predict_Single_With_TTA(Predictor_Model, Input_Current_Image_AsArray, Original_Device)
		Output_Current_Label_AsArray = Output_Current_Label_AsArray.numpy().astype(np.float32)

		# Get Weights_AsArray
		Array_Shape = Output_Current_Label_AsArray.shape[1:]
		Array_Weights = 1
		for i in range(len(Array_Shape)):
			Current_Slice = [np.newaxis] * len(Array_Shape)
			Current_Slice[i] = slice(None)
			Current_Size = Array_Shape[i]
			Array_Weights = Array_Weights * triang(Current_Size)[tuple(Current_Slice)]
		Output_Current_Label_Weights_AsArray = np.broadcast_to(Array_Weights, Output_Current_Label_AsArray.shape).astype(np.float32)

		Output_Current_Label_AsArray = Output_Current_Label_AsArray * Output_Current_Label_Weights_AsArray
		Weight_Mask = Output_Current_Label_Weights_AsArray > 0.0
		Output_Current_Label_AsArray[Weight_Mask] = Output_Current_Label_AsArray[Weight_Mask] / Output_Current_Label_Weights_AsArray[Weight_Mask]

		Output_Batch_Labels_AsArray[Input_Current_Image_Number] = Output_Current_Label_AsArray

	Output_Batch_Labels_Tensor = torch.tensor(Output_Batch_Labels_AsArray).to(device=Original_Device, dtype=torch.float32)
	return Output_Batch_Labels_Tensor

