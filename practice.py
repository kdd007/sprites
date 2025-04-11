import os
import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

def analyze_image_grouped_colors(image_path, n_clusters=5):
    """
    Reads a PNG image, gets its pixel colors, groups similar colors using
    K-Means clustering, and calculates the percentage of each color group.

    Args:
        image_path (str): The path to the PNG image file.
        n_clusters (int): The number of color groups to create.

    Returns:
        dict: A dictionary where keys are the representative color tuples (BGR)
              of each group and values are the percentage of pixels belonging
              to that color group in the image. Returns None if the image
              cannot be read.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        if img.shape[2] == 4:
            mask = img[:, :, 3] > 0
            b, g, r, a = cv2.split(img)
            pixels = np.vstack((b[mask].flatten(), g[mask].flatten(), r[mask].flatten())).T
        else:
            pixels = img.reshape((-1, 3))

        total_pixels = pixels.shape[0]
        if total_pixels == 0:
            return {}

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(pixels)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_.astype(int)

        cluster_counts = Counter(labels)
        color_percentages = {}
        for i in range(n_clusters):
            count = cluster_counts[i]
            percentage = (count / total_pixels) * 100
            color = tuple(cluster_centers[i].tolist())  # BGR
            color_percentages[color] = percentage

        return color_percentages

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        return None

def analyze_folder_grouped_colors(folder_path, n_clusters=5):
    """
    Reads all PNG files in a folder, groups similar colors in each image
    using K-Means, and calculates the total percentage of these color groups
    across all images.

    Args:
        folder_path (str): The path to the folder containing the PNG images.
        n_clusters (int): The number of color groups to create for each image.

    Returns:
        dict: A dictionary where keys are the representative color tuples (BGR)
              of the groups and values are the total percentage of pixels
              belonging to that color group across all analyzed images.
              Returns an empty dictionary if no valid PNG images are found.
    """
    all_cluster_counts = Counter()
    total_pixels_across_images = 0
    all_representative_colors = {} # Keep track of representative colors from each image

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            grouped_percentages = analyze_image_grouped_colors(image_path, n_clusters)

            if grouped_percentages:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    if image.shape[2] == 4:
                        mask = image[:, :, 3] > 0
                        total_pixels = np.sum(mask)
                    else:
                        total_pixels = image.shape[0] * image.shape[1]
                    total_pixels_across_images += total_pixels

                    # Accumulate color counts based on the grouped percentages
                    for color, percentage in grouped_percentages.items():
                        count = (percentage / 100) * total_pixels
                        all_cluster_counts[color] += count

    if total_pixels_across_images == 0:
        return {}

    overall_color_percentages = {}
    for color, total_count in all_cluster_counts.items():
        overall_color_percentages[color] = (total_count / total_pixels_across_images) * 100

    return overall_color_percentages

if __name__ == "__main__":
    folder_to_analyze = input("Enter the path to the folder containing the PNG images: ")
    num_color_groups = int(input("Enter the number of color groups to use: "))

    if not os.path.isdir(folder_to_analyze):
        print(f"Error: Folder '{folder_to_analyze}' not found.")
    else:
        overall_grouped_percentages = analyze_folder_grouped_colors(folder_to_analyze, num_color_groups)

        if overall_grouped_percentages:
            print(f"\nTotal Color Percentages (Grouped into {num_color_groups} colors) Across All Images:")
            for color, percentage in sorted(overall_grouped_percentages.items(), key=lambda item: item[1], reverse=True):
                print(f"BGR Group: {color}, Percentage: {percentage:.4f}%")
        else:
            print("No valid PNG images found in the specified folder.")