import math

import numpy as np
import cv2


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: int
) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    if not (0.0 < prob_success < 1.0):
        raise ValueError("prob_success have to be in the range (0, 1)")
    if sample_size <= 0:
        raise ValueError("sample_size have to be a positive integer")
    if not (0.0 < ind_prob_correct < 1.0):
        raise ValueError("ind_prob_correct have to be in the range (0, 1)")
    
    w = ind_prob_correct
    s = sample_size
    p = prob_success
    if w == 0:
        return int(1e9)
    if w == 1:
        return 1
    ws = w ** s
    denom = math.log(1 - ws)
    if denom == 0:
        return 1
    N = math.log(1 - p) / denom
    num_samples = int(math.ceil(N)) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)

def ransac_homography(
    points_a: np.ndarray, points_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses the RANSAC algorithm to robustly estimate a homography matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) of points from image A.
    -   points_b: A numpy array of shape (N, 2) of corresponding points from image B.

    Returns:
    -   best_H: The best homography matrix of shape (3, 3).
    -   inliers_a: The subset of points_a that are inliers (M, 2).
    -   inliers_b: The subset of points_b that are inliers (M, 2).
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    #                                                                         #
    # HINT: You are allowed to use the `cv2.findHomography` function to       #
    # compute the homography from a sample of points. To compute a direct     #
    # solution without OpenCV's built-in RANSAC, use it like this:            #
    #   H, _ = cv2.findHomography(sample_a, sample_b, 0)                      #
    # The `0` flag ensures it computes a direct least-squares solution.       #
    ###########################################################################
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    if points_a.shape != points_b.shape or points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError("points_a and points_b have to be shape (N,2) and match")

    N = points_a.shape[0]
    if N < 4:
        raise ValueError("Need at least 4 correspondences for homography")
    num_iters = 5000
    thresh = 3.0
    sample_size = 4
    best_H = None
    best_inliers = None
    best_count = -1
    best_mean_err = np.inf
    A_h = np.hstack([points_a, np.ones((N, 1), dtype=np.float64)])
    for _ in range(num_iters):
        idx = np.random.choice(N, size=sample_size, replace=False)
        sample_a = points_a[idx]
        sample_b = points_b[idx]
        H, _ = cv2.findHomography(sample_a, sample_b, 0)
        if H is None:
            continue

        proj = (H @ A_h.T).T
        w = proj[:, 2:3]
        valid = np.abs(w[:, 0]) > 1e-12
        proj_xy = np.zeros((N, 2), dtype=np.float64)
        proj_xy[valid] = proj[valid, :2] / w[valid]
        err = np.linalg.norm(proj_xy - points_b, axis=1)
        err[~valid] = np.inf

        inliers = err < thresh
        count = int(np.sum(inliers))
        if count == 0:
            continue
        mean_err = float(np.mean(err[inliers]))
        if (count > best_count) or (count == best_count and mean_err < best_mean_err):
            best_count = count
            best_mean_err = mean_err
            best_H = H
            best_inliers = inliers

    if best_H is None or best_inliers is None:
        best_H, _ = cv2.findHomography(points_a, points_b, 0)
        best_inliers = np.ones((N,), dtype=bool)

    inliers_a = points_a[best_inliers]
    inliers_b = points_b[best_inliers]
    if inliers_a.shape[0] >= 4:
        H_refined, _ = cv2.findHomography(inliers_a, inliers_b, 0)
        if H_refined is not None:
            best_H = H_refined
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_H, inliers_a, inliers_b
