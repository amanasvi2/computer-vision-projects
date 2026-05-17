import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from vision.part1 import positional_encoding
from vision.part2 import get_rays, sample_points_from_rays, compute_compositing_weights, NerfModel, render_image_nerf

def test_get_rays():
    THRESH = 1e-2
    TEST_FILES = [os.path.join("tests", "data", "get_rays_test_0.npz")]
    for test_file in TEST_FILES:    
        test_data = np.load(test_file)
        h, w = int(test_data['h']), int(test_data['w'])
        intrinsics = test_data['intrinsics']
        pose = test_data['pose']
        ray_o_predict, ray_d_predict = get_rays(h, w, torch.from_numpy(intrinsics), torch.from_numpy(pose))
        error_ray_o = np.mean((test_data['ray_o'] - ray_o_predict.numpy())**2)
        error_ray_d = np.mean((test_data['ray_d'] - ray_d_predict.numpy())**2)

        assert (error_ray_o < THRESH) and (error_ray_d < THRESH)

def test_sample_points():    
    THRESH = 1e-2
    TEST_FILES = [os.path.join("tests", "data", "sample_points_from_rays_test_0.npz")]
    for test_file in TEST_FILES:    
        test_data = np.load(test_file)
        near_thresh, far_thresh = float(test_data['near_thresh']), float(test_data['far_thresh'])
        num_samples = int(test_data['num_samples'])
        ray_origins = torch.from_numpy(test_data['ray_origins'])
        ray_directions = torch.from_numpy(test_data['ray_directions'])
        points, depth = sample_points_from_rays(ray_origins=ray_origins,
        ray_directions=ray_directions, near_thresh=near_thresh,
        far_thresh=far_thresh, num_samples=num_samples, randomize = False)

        error_points = np.mean((test_data['points'] - points.numpy())**2)
        error_depth = np.mean((test_data['depth'] - depth.numpy())**2)

        assert (error_points < THRESH) and (error_depth < THRESH)        

def test_compute_compositing_weights():  
    THRESH = 1e-2
    TEST_FILES = [os.path.join("tests", "data", "compute_composit_weights_test_0.npz")] 
    for test_file in TEST_FILES:     
        test_data = np.load(test_file)
        sigma = torch.from_numpy(test_data['sigma'])
        depth_values = torch.from_numpy(test_data['depth_values'])
        weights = compute_compositing_weights(sigma, depth_values)
        error_weights = np.mean((test_data['weights'] - weights.numpy())**2)
        assert error_weights < THRESH

def render_image_test(device='cpu'):
    ckpt = torch.load(os.path.join('tests', 'data', 'render_image_test_nerf.pt'), map_location='cpu', weights_only=True)
    encode = lambda x: positional_encoding(x, num_frequencies=6)
    model_test5 = NerfModel(in_channels=39)
    model_test5.load_state_dict(ckpt['model_state_dict'])
    model_test5.to(device)
    test_intrinsics = ckpt['intrinsics'].to(device)
    tform_cam2world = ckpt['tform_cam2world'].to(device)
    groundtruth = ckpt['gt'].to(device)

    with torch.no_grad():
        rgb_predicted, depth_predicted = render_image_nerf(
                                        100, 100, test_intrinsics,
                                        tform_cam2world, 0.667,
                                        2.0, 64,
                                        encode, model_test5)

    # Test according to MSE criterion
    assert F.mse_loss(rgb_predicted, groundtruth) < 0.003
    print("Passed!")

    plt.figure(figsize=(8, 4))
    plt.subplot(131)
    plt.imshow(rgb_predicted.detach().cpu().numpy())
    plt.title(f"Output")
    plt.subplot(132)
    plt.imshow(groundtruth.cpu().numpy())
    plt.title("Expected")
    plt.subplot(133)
    plt.imshow(depth_predicted.detach().cpu().numpy())  
    plt.title("Depth")  
    plt.show()