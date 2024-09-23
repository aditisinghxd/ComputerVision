import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute Idx and Idy with cv2.Sobel
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3) #Derivative in X
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3) #Derivative in Y

    # Step 2: Ixx Iyy Ixy from Idx and Idy
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur
    # Use sdev = 1 and kernelSize = (3, 3) in cv2.GaussianBlur
    A = cv2.GaussianBlur(Ixx, (3, 3), sigmaX=1)
    B = cv2.GaussianBlur(Iyy, (3, 3), sigmaX=1)
    C = cv2.GaussianBlur(Ixy, (3, 3), sigmaX=1)

    # Step 4: Compute the harris response with the determinant and the trace of T
    det_T = A * B - C ** 2
    trace_T = A + B
    R = det_T - k * (trace_T ** 2)

    return R, A, B, C, Ix, Iy


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R = np.pad(R, pad_width=1, mode='constant', constant_values=0)


    # Step 2 (recommended): Create one image for every offset in the 3x3 neighborhood
    local_max = np.zeros_like(R, dtype=bool)


    # Step 3 (recommended): Compute the greatest neighbor of every pixel
    for x in range(1, padded_R.shape[0] - 1):
        for y in range(1, padded_R.shape[1] - 1):
            # Extract the 3x3 neighborhood
            neighborhood = padded_R[x-1:x+2, y-1:y+2]
            if R[x-1, y-1] == np.max(neighborhood):
                local_max[x-1, y-1] = True

    # Step 4 (recommended): Compute a boolean image with only all key-points set to True
    significant = (R > threshold)
    keypoints = significant & local_max


    # Step 5 (recommended): Use np.nonzero to compute the locations of the key-points from the boolean image
    y_coords, x_coords = np.nonzero(keypoints)

    return x_coords, y_coords


def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R = np.pad(R, pad_width=1, mode='constant', constant_values=0)


    # Step 2 (recommended): Calculate significant response pixels
    significant = (R <= edge_threshold)


    # Step 3 (recommended): Create two images with the smaller x-axis and y-axis neighbors respectively
    local_min_x = np.zeros_like(R, dtype=bool)
    local_min_y = np.zeros_like(R, dtype=bool)

    for x in range(1, padded_R.shape[0] - 1):
        for y in range(1, padded_R.shape[1] - 1):
            # Check x-axis neighbors for local minimum
            if (padded_R[x, y] < padded_R[x, y-1]) and (padded_R[x, y] < padded_R[x, y+1]):
                local_min_x[x-1, y-1] = True
            
            # Check y-axis neighbors for local minimum
            if (padded_R[x, y] < padded_R[x-1, y]) and (padded_R[x, y] < padded_R[x+1, y]):
                local_min_y[x-1, y-1] = True


    # Step 4 (recommended): Calculate pixels that are lower than either their x-axis or y-axis neighbors
    local_min = local_min_x | local_min_y

    # Step 5 (recommended): Calculate valid edge pixels by combining significant and axis_minimal pixels
    edges = significant & local_min

    return edges
