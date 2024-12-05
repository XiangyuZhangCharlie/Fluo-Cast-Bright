##################################################
# Network and Utils of classification network
##################################################

import torch
import torchvision

# Full Classification Network

class Full_Classification_Network(torch.nn.Module):
	def __init__(self, Number_Of_Classes):
		super().__init__()
		self.Resnext101_Main = torchvision.models.resnext101_32x8d(pretrained=False,num_classes=Number_Of_Classes)
		self.Softmax_Layer = torch.nn.Softmax(dim=1)

	def forward(self,x):
		x = self.Resnext101_Main(x)
		xFinal = self.Softmax_Layer(x)
		return xFinal

# Accuracy calculation

def Calculate_Classification_Accuracy(Predicted_Label, True_Label, Class_Name_List):
	Total_Item_Number = float(Predicted_Label.shape[0])
	Correct_Item_Number = 0.0
	for Current_Item in range(0,int(Total_Item_Number)):
		for Class_Name_Index in range(0,len(Class_Name_List)):
			if float(max(Predicted_Label[Current_Item])) == float(Predicted_Label[Current_Item][Class_Name_Index]) and float(True_Label[Current_Item][Class_Name_Index]) == 1.0:
				Correct_Item_Number += 1.0
				continue
	Final_Accuracy = Correct_Item_Number / Total_Item_Number
	return Final_Accuracy
