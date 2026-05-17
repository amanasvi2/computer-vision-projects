"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points has to be shape (N,2)")

    mu = np.mean(points, axis=0)
    sigma = np.std(points, axis=0, ddof=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    points_normalized = (points - mu) / sigma

    T = np.array([
        [1.0 / sigma[0], 0.0, -mu[0] / sigma[0]],
        [0.0, 1.0 / sigma[1], -mu[1] / sigma[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_norm = np.asarray(F_norm, dtype=np.float64)
    T_a = np.asarray(T_a, dtype=np.float64)
    T_b = np.asarray(T_b, dtype=np.float64)
    if F_norm.shape != (3, 3):
        raise ValueError("F_norm has to be shape (3, 3)")
    if T_a.shape != (3, 3) or T_b.shape != (3, 3):
        raise ValueError("T_a and T_b have to be shape (3, 3)")
    F_orig = T_b.T @ F_norm @ T_a

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def make_singular(F_norm: np.array) -> np.ndarray:
    """
    Force F to be singular by zeroing the smallest of its singular values.
    This is done because F is not supposed to be full rank, but an inaccurate
    solution may end up as rank 3.

    Args:
    - F_norm: A numpy array of shape (3,3) representing the normalized fundamental matrix.

    Returns:
    - F_norm_s: A numpy array of shape (3, 3) representing the normalized fundamental matrix
                with only rank 2.
    """
    U, D, Vt = np.linalg.svd(F_norm)
    D[-1] = 0
    F_norm_s = np.dot(np.dot(U, np.diag(D)), Vt)

    return F_norm_s


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here. Equation (9) in the documentation indicates
    one equation of a linear system in which you'll want to solve for f_{i, j}.

    Since the matrix is defined up to a scale, many solutions exist. To constrain
    your solution, use can either use SVD and use the last Vt vector as your
    solution, or you can fix f_{3, 3} to be 1 and solve with least squares.

    Be sure to reduce the rank of your estimate - it should be rank 2. The
    make_singular() function can do this for you.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)   
    if points_a.shape != points_b.shape or points_a.shape[1] != 2 or points_a.ndim != 2: 
        raise ValueError("points_a and points_b have to have the same shape (N, 2)")
    N = points_a.shape[0]
    if N < 8:
        raise ValueError("At least 8 point pairs are needed to solve for the fundamental matrix")
    points_a_norm, T_a = normalize_points(points_a)
    points_b_norm, T_b = normalize_points(points_b)
    xa, ya = points_a_norm[:, 0], points_a_norm[:, 1]
    xb, yb = points_b_norm[:, 0], points_b_norm[:, 1]
    A = np.column_stack((xb * xa, xb * ya, xb, yb * xa, yb * ya, yb, xa, ya, np.ones(N, dtype=np.float64)))
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1, :].reshape(3, 3)
    F_norm = make_singular(F_norm)
    F = unnormalize_F(F_norm, T_a, T_b)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
