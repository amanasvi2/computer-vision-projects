from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision.part2_baseline import Baseline
from vision.part3_pointnet import PointNet
from vision.part5_tnet import PointNetTNet


def get_critical_indices(model: Union[PointNet, PointNetTNet], pts: torch.Tensor) -> np.ndarray:
    '''
    Finds the indices of the critical points in the given point cloud. A
    critical point is a point that contributes to the global feature (i.e
    a point whose calculated feature has a maximal value in at least one 
    of its dimensions)
    
    Hint:
    1) Use the encodings returned by your model
    2) Make sure you aren't double-counting points since points may
       contribute to the global feature in more than one dimension

    Inputs:
        model: The trained model
        pts: (model.pad_size, 3) tensor point cloud representing an object

    Returns:
        crit_indices: (N,) numpy array, where N is the number of critical pts

    '''
    crit_indices = None

    ############################################################################
    # Student code begin
    ############################################################################

    model.eval()
    with torch.no_grad():
        x = pts.unsqueeze(0)
        output = model(x)
        
        if isinstance(output, tuple):
            _, encodings = output
        else:
            encodings = output

        if encodings is None:
            if hasattr(model, 'point_net'):
                x = model.apply_tnet(x)
                enc = model.point_net
            else:
                enc = model

            x_enc = enc.relu(enc.bn1(enc.enc1(x).permute(0, 2, 1)).permute(0, 2, 1))
            x_enc = enc.relu(enc.bn2(enc.enc2(x_enc).permute(0, 2, 1)).permute(0, 2, 1))
            x_enc = enc.relu(enc.bn3(enc.enc3(x_enc).permute(0, 2, 1)).permute(0, 2, 1))
            encodings = enc.enc4(x_enc)
    encodings = encodings.squeeze(0)
    max_indices = torch.argmax(encodings, dim=0)
    crit_indices = torch.unique(max_indices).cpu().numpy()

    ############################################################################
    # Student code end
    ############################################################################

    return crit_indices

    
def get_confusion_matrix(
    model: Union[Baseline, PointNet, PointNetTNet], 
    loader: DataLoader, 
    num_classes: int,
    normalize: bool=True, 
    device='cpu'
) -> np.ndarray:
    '''
    Builds a confusion matrix for the given models predictions
    on the given dataset. 
    
    Recall that each ground truth label corresponds to a row in
    the matrix and each predicted value corresponds to a column.

    A confusion matrix can be normalized by dividing entries for
    each ground truch prior by the number of actual isntances the
    ground truth appears in the dataset. (Think about what this means
    in terms of rows and columns in the matrix) 

    Hint:
    1) Generate list of prediction, ground-truth pairs
    2) For each pair, increment the correct cell in the matrix
    3) Keep track of how many instances you see of each ground truth label
       as you go and use this to normalize 

    Args: 
    -   model: The model to use to generate predictions
    -   loader: The dataset to use when generating predictions
    -   num_classes: The number of classes in the dataset
    -   normalize: Whether or not to normalize the matrix
    -   device: If 'cuda' then run on GPU. Run on CPU by default

    Output:
    -   confusion_matrix: a numpy array with shape (num_classes, num_classes)
                          representing the confusion matrix
    '''

    model.eval()
    confusion_matrix = None

    ############################################################################
    # Student code begin
    ############################################################################

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    gt_counts = np.zeros(num_classes, dtype=np.float32)
    with torch.no_grad():
        for pts, labels in loader:
            pts = pts.to(device)
            labels = labels.to(device)
            outputs, _ = model(pts)
            preds = torch.argmax(outputs, dim=1)
            for gt, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion_matrix[gt, pred] += 1
                gt_counts[gt] += 1

    if normalize:
        for i in range(num_classes):
            if gt_counts[i] > 0:
                confusion_matrix[i] /= gt_counts[i]

    ############################################################################
    # Student code end
    ############################################################################

    model.train()

    return confusion_matrix
