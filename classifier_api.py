from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import time
import copy
import os
import sys
import argparse
import csv
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from PIL import Image
import glob
import cv2
import datetime
import time

def micros(t1, t2):
    delta = (t2-t1).microseconds
    return delta
class XrayClassifier:
    def __init__(self,input_size_=1000,mean_=[0.485, 0.456, 0.406],std_=[0.229, 0.224, 0.225],
                class_num_=2,model_name = 'alexnet',device_id=0,with_gray=False):

        self.input_size = input_size_
        self.mean = mean_
        self.std = std_
        if with_gray:
            self.test_transform = transforms.Compose([
                    transforms.Resize((self.input_size,self.input_size)),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
        else:
            self.test_transform = transforms.Compose([
                    transforms.Resize((self.input_size,self.input_size)),
                    #transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
        self.class_num = class_num_
        self.device = torch.device("cuda:"+str(device_id))

        if model_name == "alexnet":
            """ Alexnet
            """
            self.model = models.alexnet()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224

        elif model_name == "vgg11_bn":
            """ VGG11_bn
            """
            self.model = models.vgg11_bn()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg11":
            """ VGG11
            """
            self.model = models.vgg11()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg13_bn":
            """ VGG13_bn
            """
            self.model = models.vgg13_bn()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg13":
            """ VGG13
            """
            self.model = models.vgg13()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg16_bn":
            """ VGG16_bn
            """
            self.model = models.vgg16_bn()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg16":
            """ VGG16
            """
            self.model = models.vgg16()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg19_bn":
            """ VGG19_bn
            """
            self.model = models.vgg19_bn()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg19":
            """ VGG19
            """
            self.model = models.vgg19()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "squeezenet1_0":
            """ squeezenet1_0
            """
            self.model = models.squeezenet1_0()
            self.model.classifier[1] = nn.Conv2d(512, self.class_num, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = self.class_num
            input_size = 224
        elif model_name == "squeezenet1_1":
            """ squeezenet1_1
            """
            self.model = models.squeezenet1_1()
            self.model.classifier[1] = nn.Conv2d(512, self.class_num, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = self.class_num
            input_size = 224
        elif model_name == "resnet18":
            """ Resnet18
            """
            self.model = models.resnet18()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name == "resnet34":
            """ Resnet34
            """
            self.model = models.resnet34()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name == "resnet50":
            """ Resnet50
            """
            self.model = models.resnet50()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name == "resnet101":
            """ Resnet101
            """
            self.model = models.resnet101()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name == "resnet152":
            """ Resnet152
            """
            self.model = models.resnet152()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224
        elif model_name=='inception3':
            self.model = models.inception_v3(pretrained=False)
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.class_num)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.class_num)
        elif model_name == "densenet121":

            self.model = models.densenet121()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num) 
        elif model_name == "densenet161":

            self.model = models.densenet161()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num)
        elif model_name == "densenet169":

            self.model = models.densenet169()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num) 
        elif model_name == "densenet201":

            self.model = models.densenet201()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num)  
        elif model_name == "shufflenetv2_x0_5":

            self.model = models.shufflenetv2_x0_5()
            num_ftrs = self.model.fc.in_features
            self.model = nn.Linear(num_ftrs, self.class_num)
        elif model_name == "mobilenetv2":
            self.model = models.mobilenet_v2()
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs,self.class_num)
        elif model_name == "shufflenetv2_x1_0":   
            self.model = models.shufflenetv2_x1_0()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,  self.class_num)
            input_size = 224
        else:
            assert False,"No model with the name of "+model_name
                    
    def softmax(self,x):

        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def ini_model(self,model_dir):

        checkpoint = torch.load(model_dir)

        self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)

        cudnn.benchmark = True

        self.model.eval()

    def predict(self,img):
        t1 = datetime.datetime.now()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        image = self.test_transform(image)
        
        inputs = image
        inputs = Variable(inputs)
        inputs = inputs.to(self.device)
        inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2)) 
        
        
        outputs = self.model(inputs)
        t2 = datetime.datetime.now()
        #print(micros(t1,t2)/1000)
        softmax_res = self.softmax(outputs.data.cpu().numpy()[0])
        probilities = []
        for probility in softmax_res:
            probilities.append(probility)
        return probilities.index(max(probilities))

    def predict_imgdir(self,img_dir):

        assert os.path.exists(img_dir),"No such image file:"+img_dir
        
        img = cv2.imread(img_dir)

        return self.predict(img)

if __name__ == "__main__":
    classifier = XrayClassifier(224,model_name="mobilenetv2")
    classifier.ini_model("modelstobeused/unbalanced/mobilenet_v2.model")
    img = cv2.imread("testImages/C0000004-3F_091020084933562.jpg")
    class_label = classifier.predict(img)
    print(class_label)