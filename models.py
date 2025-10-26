"""
Model architectures for CIFAR-100 classification.

This file contains the exact model definitions used in the training notebook,
ensuring that saved checkpoints can be loaded correctly.
"""

import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    """Two-layer fully connected neural network."""
    
    def __init__(self, input_size=3*32*32, hidden_size=512, num_classes=100):
        """
        Initializes the two-layer neural network model.
        
        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden layer.
            num_classes (int): The number of classes in the dataset.
        """
        super(TwoLayerNet, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = x.reshape(x.size(0), -1)
        return self.layers(x)


class ConvNet(nn.Module):
    """LeNet-style convolutional neural network."""
    
    def __init__(self):
        """
        Initializes the convolutional neural network model.
        """
        super(ConvNet, self).__init__()
        
        # Feature extractor (three convolutional layers + activations + pooling)
        self.features = nn.Sequential(
            # C1: First conv block: 3->6 channels with 5x5 kernels (handles RGB input), Tanh activation, then 2x2 average pooling
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # C2: Second conv block: 6->16 channels with 5x5 kernels, Tanh activation, then 2x2 average pooling
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # C3: Third conv block: 16->120 channels with 5x5 kernels, Tanh activation
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),  # F6 layer
            nn.Tanh(),  # Tanh activation
            nn.Linear(84, 100)  # Output layer for 100 CIFAR-100 classes
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the convolutional network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


class ExtendedTwoLayerNet(nn.Module):
    """Deep MLP: input -> h1 -> h2 -> h3 -> out with dropout."""
    
    def __init__(self, input_size=3*32*32, h1=768, h2=512, h3=256, num_classes=100, dropout=0.25):
        """
        Initializes the extended two-layer network with multiple hidden layers.
        
        Args:
            input_size (int): Size of input features
            h1 (int): Size of first hidden layer
            h2 (int): Size of second hidden layer
            h3 (int): Size of third hidden layer
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, h1), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(h3, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.seq(x.reshape(x.size(0), -1))


class ExtendedConvNet(nn.Module):
    """Extended ConvNet with deeper architecture and batch normalization."""
    
    def __init__(self, in_channels=3, fc_mid=256, dropout=0.30, c1_out=32, c2_out=64, 
                 c3_out=128, c4_out=256, c5_out=512, dropout2d=0.2, num_classes=100):
        """
        Initializes the extended convolutional network.
        
        Args:
            in_channels (int): Number of input channels
            fc_mid (int): Size of middle fully connected layer
            dropout (float): Dropout probability for FC layers
            c1_out to c5_out (int): Output channels for each conv layer
            dropout2d (float): 2D dropout probability for conv layers
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1_out, 5), nn.BatchNorm2d(c1_out), nn.SiLU(), nn.AvgPool2d(2, 2),
            nn.Conv2d(c1_out, c2_out, 5), nn.BatchNorm2d(c2_out), nn.SiLU(), nn.Dropout2d(dropout2d),
            nn.Conv2d(c2_out, c3_out, 3, padding=1), nn.BatchNorm2d(c3_out), nn.SiLU(), nn.Dropout2d(dropout2d),
            nn.Conv2d(c3_out, c4_out, 3, padding=1), nn.BatchNorm2d(c4_out), nn.SiLU(), nn.Dropout2d(dropout2d), nn.AvgPool2d(2, 2),
            nn.Conv2d(c4_out, c5_out, 5), nn.BatchNorm2d(c5_out), nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c5_out, fc_mid), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(fc_mid, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # (batch, c5_out, 1, 1) -> (batch, c5_out)
        x = self.classifier(x)
        return x


def build_model_from_params(model_name, params):
    """
    Build a model instance given the model name and hyperparameters.
    
    Args:
        model_name (str): Name of the model ('TwoLayerNet', 'ConvNet', etc.)
        params (dict): Dictionary of hyperparameters
    
    Returns:
        nn.Module: Instantiated model
    """
    if model_name == 'TwoLayerNet':
        return TwoLayerNet(
            input_size=3*32*32,
            hidden_size=params['hidden_size'],
            num_classes=100
        )
    elif model_name == 'ConvNet':
        return ConvNet()
    elif model_name == 'ExtendedTwoLayerNet':
        return ExtendedTwoLayerNet(
            input_size=3*32*32,
            h1=params['h1'],
            h2=params['h2'],
            h3=params['h3'],
            num_classes=100,
            dropout=params['dropout']
        )
    elif model_name == 'ExtendedConvNet':
        return ExtendedConvNet(
            in_channels=3,
            fc_mid=params['fc_mid'],
            dropout=params['dropout'],
            c1_out=32,
            c2_out=64,
            c3_out=128,
            c4_out=256,
            c5_out=512,
            dropout2d=params.get('dropout2d', 0.2),
            num_classes=100
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

