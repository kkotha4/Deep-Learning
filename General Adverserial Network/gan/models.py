import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 4, stride=2,padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2,padding=1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256,512,4,stride=2,padding=1)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4,stride=2,padding=1)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, 4,stride=1,padding=0)
        self.relu=nn.LeakyReLU(0.2)
        
        
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        
        ##########       END      ##########
        x = self.relu(self.conv1(x))
        
       
        #print(x.shape)
        x = self.relu(self.conv2_bn(self.conv2(x)))
        
        x = self.relu(self.conv3_bn(self.conv3(x)))
        #print(x.shape)
        x = self.relu(self.conv4_bn(self.conv4(x)))
        #print(x.size())
        #print("kashish")
        x = self.conv5(x)
        
  
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
            
        self.conv1 = nn.ConvTranspose2d(self.noise_dim,1024,4, stride=1,padding=0)
        self.conv1_bn = nn.BatchNorm2d(1024)
        self.conv2 =nn.ConvTranspose2d(1024,512,4, stride=2,padding=1)
        self.conv2_bn = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512,256,4,stride=2,padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, 4,stride=2,padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 3, 4,stride=2, padding=1)
        self.relu=nn.LeakyReLU(0.2)
        self.tanh=nn.Tanh()
        
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = x.reshape(-1,self.noise_dim,1,1)
        x = self.relu(self.conv1_bn(self.conv1(x)))
        
       
        #print(x.shape)
        x = self.relu(self.conv2_bn(self.conv2(x)))
        
        x = self.relu(self.conv3_bn(self.conv3(x)))
        #print(x.shape)
        x = self.relu(self.conv4_bn(self.conv4(x)))
    
        x = self.tanh(self.conv5(x))
        #x = x.view(-1,3,64,64)
        #print(x.size())
        
        
        
        ##########       END      ##########
        
        return x
    

