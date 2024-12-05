##################################################
# Data Generator
##################################################

import torch
import numpy as np
from tqdm import tqdm
from skimage import io
from collections import deque

def Random_Crop(Image_Stack,Label_Stack,Crop_Shape):
	X_Difference = Image_Stack.shape[-2] - Crop_Shape[-2]
	Y_Difference = Image_Stack.shape[-1] - Crop_Shape[-1]
	Z_Difference = Image_Stack.shape[-3] - Crop_Shape[-3]

	X_Random = np.random.randint(max(X_Difference-1,1))
	Y_Random = np.random.randint(max(Y_Difference-1,1))
	Z_Mid = int(Z_Difference/2)

	Cropped_Image = Image_Stack[Z_Mid:Z_Mid+Crop_Shape[-3],X_Random:X_Random+Crop_Shape[-2],Y_Random:Y_Random+Crop_Shape[-1]]
	Cropped_Label = Label_Stack[Z_Mid:Z_Mid+Crop_Shape[-3],X_Random:X_Random+Crop_Shape[-2],Y_Random:Y_Random+Crop_Shape[-1]]
	Cropped_Image = np.expand_dims(Cropped_Image,axis=0)
	Cropped_Label = np.expand_dims(Cropped_Label,axis=0)
	return Cropped_Image, Cropped_Label

class Cell_Image_Dataset(torch.utils.data.Dataset):
	def __init__(self, Image_Name_List):
		# Initialization
		self.List_Of_Images = Image_Name_List

	def __len__(self):
		# Total Number of Images
		return len(self.List_Of_Images)

	def __getitem__(self, index):
		Image_Name = self.List_Of_Images[index]
		Image_Stack = io.imread(Image_Name)
		X = Image_Stack[0,]
		y = Image_Stack[1,]
		return X, y

class Buffer_Patch_Dataset(torch.utils.data.Dataset):
	def __init__(self, Original_Dataset, Patch_Shape, Buffer_Size, Buffer_Exchange_Interval, Shuffle):
		# Initialization
		self.Original_Dataset = Original_Dataset
		self.Patch_Shape = Patch_Shape
		self.Buffer_Size = min(len(self.Original_Dataset), Buffer_Size)
		self.Buffer_Exchange_Interval = Buffer_Exchange_Interval
		self.Shuffle_Switch = Shuffle
		self.BEI_Counter = 0
		self.Buffer = deque()
		self.Not_Yet_In_Buffer = deque()
		for Buffer_Index in tqdm(range(self.Buffer_Size), desc="Loading initial image patches into buffer..."):
			self.Buffer_Insert_New()

	def __iter__(self):
		return self

	def __next__(self):
		Image_Patch = self.Get_Random_Patch()
		self.BEI_Counter += 1
		if (self.Buffer_Exchange_Interval > 0) and (self.BEI_Counter % self.Buffer_Exchange_Interval == 0):
			self.Buffer_Insert_New()
		return Image_Patch

	def Buffer_Insert_New(self):
		if len(self.Not_Yet_In_Buffer) == 0:
			self.Not_Yet_In_Buffer = deque(range(len(self.Original_Dataset)))
			if self.Shuffle_Switch:
				np.random.shuffle(self.Not_Yet_In_Buffer)
		if len(self.Buffer) >= self.Buffer_Size:
			self.Buffer.popleft()
		New_Datum_Index = self.Not_Yet_In_Buffer.popleft()
		self.Buffer.append(self.Original_Dataset[New_Datum_Index])
		print("Added image patch " + str(New_Datum_Index) + " into buffer")

	def Get_Random_Patch(self):
		Buffer_Index = np.random.randint(len(self.Buffer))
		Random_Image_From_Buffer, Random_Label_From_Buffer = self.Buffer[Buffer_Index]
		Randomly_Cropped_Image, Randomly_Cropped_Label = Random_Crop(Image_Stack=Random_Image_From_Buffer,Label_Stack=Random_Label_From_Buffer,Crop_Shape=self.Patch_Shape)
		return Randomly_Cropped_Image, Randomly_Cropped_Label

	def Get_Patch_Batch(self, Batch_Size):
		return tuple(torch.tensor(np.stack(Batch_Part)) for Batch_Part in zip(*[next(self) for Batch_Index in range(Batch_Size)]))

class Image_Dataset_For_Valid(torch.utils.data.Dataset):
	def __init__(self, Image_Name_List):
		# Initialization
		self.List_Of_Images = Image_Name_List

	def __len__(self):
		# Total Number of Images
		return len(self.List_Of_Images)

	def __getitem__(self, index):
		Image_Name = self.List_Of_Images[index]
		X, y = Random_Crop(Image_Stack=io.imread(Image_Name)[0,],Label_Stack=io.imread(Image_Name)[1,],Crop_Shape=(16,128,128))
		return X, y
