import torch
import torch.nn as nn
import numpy as np
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(f'{base_dir}/weights', exist_ok=True)
os.makedirs(f'{base_dir}/inputs', exist_ok=True)
os.makedirs(f'{base_dir}/outputs', exist_ok=True)

def save_tensor_to_txt(tensor, filepath):
    np_array = tensor.detach().cpu().numpy().flatten()
    np.savetxt(filepath, np_array, fmt='%f')

class TinyYOLOLite(nn.Module):
    def __init__(self, num_classes=10, num_anchors=2):
        super(TinyYOLOLite, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 0, bias=True)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, 3, 1, 0, bias=True)
        self.relu2 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, 3, 1, 0, bias=True)
        self.relu3 = nn.ReLU()

        self.detection_head = nn.Conv2d(32, num_anchors * (5 + num_classes), 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        output = self.detection_head(x)
        return output
    
if __name__ == "__main__":
    model = TinyYOLOLite(num_classes=10, num_anchors=2)
    model.eval()

    # EXTRACT WEIGHTS AND BIASES
    for name, param in model.named_parameters():
        clean_name = name.replace('.', '_')
        save_tensor_to_txt(param, f'{base_dir}/weights/{clean_name}.txt')

    # EXTRACT INTERMEDIATE FEATURE MAPS
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model.conv1.register_forward_hook(get_activation('conv1_out'))
    model.conv2.register_forward_hook(get_activation('conv2_out'))
    model.conv3.register_forward_hook(get_activation('conv3_out'))
    model.relu1.register_forward_hook(get_activation('relu1_out'))
    model.relu2.register_forward_hook(get_activation('relu2_out'))
    model.relu3.register_forward_hook(get_activation('relu3_out'))
    model.maxpool1.register_forward_hook(get_activation('maxpool1_out'))
    model.maxpool2.register_forward_hook(get_activation('maxpool2_out'))
    model.detection_head.register_forward_hook(get_activation('detection_head_out'))

    # DUMMY INPUT
    dummy_input = torch.randn(2, 3, 64, 64)
    save_tensor_to_txt(dummy_input, f'{base_dir}/inputs/dummy_input.txt')

    # FORWARD PASS
    output = model(dummy_input)

    # EXTRACT OUTPUT
    save_tensor_to_txt(output, f'{base_dir}/outputs/dummy_output.txt')

    # EXTRACT INTERMEDIATE FEATURE MAPS
    for key, value in activation.items():
        save_tensor_to_txt(value, f'{base_dir}/outputs/{key}.txt')