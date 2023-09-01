# This program is to Detect the edges in a images and convert the pixels on the edges to world coordinate points

#############################################################################################################
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import time
import pandas as pd
# Import Delaunay triangulation from scipy.spatial
from scipy.spatial import Delaunay

#############################################################################################################

# Resize Image


def resize_image(image):
    # Resize to specific size
    dim = (595, 842)

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized


#############################################################################################################

# Function to detect the edges in a image


def detect_edges(image):
    # Resize image
    image = resize_image(image)

    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to the gray scale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 5,5 is the kernel size

    # Apply Canny edge detection to the blurred image
    # 50,150 are the thresholds for the hysteresis procedure
    canny = cv2.Canny(blur, 50, 150)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)

    return canny


#############################################################################################################

# Convert canny edge detected image pixel i,j to equivalent world coordinate x,y points

def pixels_to_points(canny):
    # Invert the image
    # canny = cv2.bitwise_not(canny)

    # Get the shape of the image
    height, width = canny.shape

    # Get the number of pixels
    num_pixels = height * width

    # Get the number of points
    num_points = 0

    # Get the pixel values
    pixel_values = canny.reshape(num_pixels)

    # Get the x,y points
    points = []

    # Convert the pixel values to points
    for i in range(num_pixels):
        if pixel_values[i] == 255:
            x = i % width
            y = math.floor(i / width)
            points.append([x, y])
            num_points += 1

    # Reflect the points by 180 degrees
    for i in range(num_points):
        points[i][0] = width - points[i][0]
        points[i][1] = height - points[i][1]

    # Plot the points using matplotlib
    # create a figure
    plt.figure()
    # plot the points
    plt.scatter(*zip(*points), color="red", marker=".",
                s=0.1)  # Set the marker size using s=0.1
    # set the title
    plt.title("Points")
    # set the x-axis label
    plt.xlabel("x")
    # set the y-axis label
    plt.ylabel("y")
    # Set the x axis limit
    plt.xlim(0, width)
    # Set the y axis limit
    plt.ylim(0, height)
    # display scatter plot
    plt.show()

    return points, num_points


#############################################################################################################

# Convert the points to a csv file
def points_to_csv(filename, points, num_points):
    # Create a dataframe
    df = pd.DataFrame(points, columns=["x", "y"])

    filename = filename.split(".")[0]
    filename = "points_" + filename + ".csv"

    # Save the dataframe to a csv file
    df.to_csv(filename, index=False)

    return

#############################################################################################################

# Function for Delaunay Triangulation from a set of points


def delaunay_triangulation(points, num_points):
    # Triangulate the points using scipy Delaunay triangulation
    triangles = Delaunay(points)

    # Get the number of triangles
    num_triangles = triangles.nsimplex

    # Get the indices of the triangles
    indices = triangles.simplices

    # Get the coordinates of the triangles
    coordinates = triangles.points

    # Get the edges of the triangles
    edges = triangles.convex_hull

    # Plot the triangles using matplotlib
    # create a figure
    plt.figure()
    # plot the points
    plt.triplot(coordinates[:, 0], coordinates[:, 1], indices, color="blue")
    # set the title
    plt.title("Triangles")
    # set the x-axis label
    plt.xlabel("x")
    # set the y-axis label
    plt.ylabel("y")
    # display scatter plot
    plt.show()

    return triangles, edges

#############################################################################################################

# Main Function


def main():
    # Read the image
    image_path = ".\Dataset\Image (1).jpg"
    file_name = os.path.basename(image_path)
    if not os.path.exists(image_path):
        print("Image does not exist")
        sys.exit(0)

    image = cv2.imread(image_path)

    edge_detected_image = detect_edges(image)
    points, num_points = pixels_to_points(edge_detected_image)
    points_to_csv(file_name, points, num_points)
    return

#############################################################################################################


# Call the main function
if __name__ == "__main__":
    main()
