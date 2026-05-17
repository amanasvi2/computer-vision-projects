import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
import time

from vision.part1 import positional_encoding, Model2d

def test_positional_encoding():   
    THRESH = 1e-2
    TEST_FILES = [os.path.join("tests", "data", "positional_encoding_test_0.npz"), 
                  os.path.join("tests", "data", "positional_encoding_test_1.npz")]
    for test_file in TEST_FILES:
        test_data = np.load(test_file)
        coordinates = torch.from_numpy(test_data['coordinates'])
        embedded_coordinates = positional_encoding(coordinates, num_frequencies=int(test_data['num_frequencies']))
        error_coordinates = np.mean((test_data['embedded_coordinates'] - embedded_coordinates.numpy())**2)
        assert error_coordinates < THRESH

def test_model_2d_shape():
    model = Model2d()
    inp = torch.randn(180, 2)
    out = model(inp)
    assert out.shape == torch.Size([180, 3]), "the input output shape is not 3!"
    