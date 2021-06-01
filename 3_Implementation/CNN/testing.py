import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from random import randint
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
model_file = 'veggie-classifier.pt'

def resize_image(src_img, size=(128,128), bg_color="white"): 
    from PIL import Image
    src_img.thumbnail(size, Image.ANTIALIAS)
    new_image = Image.new("RGB", size, bg_color)
    new_image.paste(src_img, (int((size[0] - src_img.size[0]) / 2), int((size[1] - src_img.size[1]) / 2)))
    return new_image
def predict_image(classifier, image_array):
    classifier.eval()
    class_names = ['RoadWay', 'ShortHerbs', 'Wood', 'Acacia', 'Cork oak', 'dirt', 'misc vegetation', 'other_yellow', 'pine tree']
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    image_tensor = torch.stack([transformation(image).float() for image in image_array])
    input_features = image_tensor
    predictions = classifier(input_features)
    predicted_classes = []
    for prediction in predictions.data.numpy():
        class_idx = np.argmax(prediction)
        predicted_classes.append(class_names[class_idx])
    return np.array(predicted_classes)
class Net(nn.Module):
    def __init__(self, num_classes=9):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)        
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=16 * 16 * 48, out_features=num_classes)
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool(self.conv3(x)))
        x = x.view(-1, 16 * 16 * 48)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
model = Net(num_classes=9)
model.load_state_dict(torch.load(model_file))
test_folder = '../data/Vegetation/testing'
test_image_files = os.listdir(test_folder)
image_arrays = []
size = (128,128)
background_color = "white"
fig = plt.figure(figsize=(12, 8))
for file_idx in range(len(test_image_files)):
    img = Image.open(os.path.join(test_folder, test_image_files[file_idx]))
    resized_img = np.array(resize_image(img, size, background_color))
    image_arrays.append(resized_img)
predictions = predict_image(model, np.array(image_arrays))
print(predictions)
for idx in range(len(predictions)):
    a=fig.add_subplot(3,len(predictions)/3,idx+1)
    plt.imshow(image_arrays[idx])
    a.set_title(predictions[idx])
plt.show()






