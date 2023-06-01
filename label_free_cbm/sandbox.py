import torch
import pytorchcv
from pytorchcv.model_provider import get_model as ptcv_get_model

# Load a pre-trained model from pytorchcv
model = ptcv_get_model('resnet18', pretrained=True)

# Save the model to a file
torch.save(model.state_dict(), 'resnet18.pth')

# Load the model from the file
model.load_state_dict(torch.load('resnet18.pth'))