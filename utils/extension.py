from skimage.ndimage import gaussian_filter, measurements
import numpy as np

def segment_objects(image, threshold=0.5):
    # Threshold the image to obtain initial seeds
    seeds = image > threshold

    # Initialize the labels and boundaries arrays
    labels, num_labels = measurements.label(seeds)
    boundaries = np.zeros_like(image)

    # Iterate over each seed
    for label in range(1, num_labels + 1):
        # Initialize the current seed and boundary masks
        current_seed = labels == label
        current_boundary = np.zeros_like(image)

        # Iterate until the boundary reaches its local maximum
        while True:
            # Find the maximum probability boundary pixel
            max_boundary = np.argmax(image * (1 - current_boundary) * current_seed)

            # Convert the 1D index to 2D coordinates
            max_boundary_coords = np.unravel_index(max_boundary, image.shape)

            # Check if the maximum boundary pixel is at the edge of the image
            if (max_boundary_coords[0] == 0 or max_boundary_coords[0] == image.shape[0] - 1 or
                max_boundary_coords[1] == 0 or max_boundary_coords[1] == image.shape[1] - 1):
                break

            # Update the current boundary mask
            current_boundary[max_boundary_coords] = 1

            # Check if the current boundary overlaps with any other labels
            overlap = np.logical_and(labels != label, current_boundary)

            # If there is no overlap, update the current seed and continue iterating
            if not overlap.any():
                current_seed = np.logical_or(current_seed, current_boundary)
            # If there is overlap, stop iterating and discard the current boundary
            else:
                break

        # Add the current seed to the final output
        boundaries += current_seed.astype(int)

    # Return the union of the seeds and boundaries as the final segmentation
    return np.logical_or(seeds, boundaries).astype(int)