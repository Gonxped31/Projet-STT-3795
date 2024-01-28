import Datasets 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##FETCH DATAS
#Instantiate Datasets object and get k cifar10 images
d = Datasets.Datasets()
#Array of queried images
images = d.getCifar10Images(1)

#GENERATE POINT CLOUD
# Load the depth image as a grayscale image
depth_image = cv2.imread('src/data/dataset-tools/image_735.jpg', cv2.IMREAD_GRAYSCALE)

# Convert depth to float and scale if necessary
depth_in_meters = depth_image.astype(np.float32) / 1000.0

# Get the shape of the image
height, width = depth_image.shape

# Generate meshgrid for pixel coordinates
xx, yy = np.meshgrid(np.arange(width), np.arange(height))

fx, fy = 525, 525
cx, cy = width / 2, height / 2

# Convert pixel coordinates to camera coordinates
x_3d = (xx - cx) * depth_in_meters / fx
y_3d = (yy - cy) * depth_in_meters / fy
z_3d = depth_in_meters

# Stack to get 3D coordinates
points = np.dstack((x_3d, y_3d, z_3d)).reshape(-1, 3)

# filter out points with zero depth
#points = points[depth_in_meters.flatten() > 0]


sampled_points = points[::10]

# Create a new matplotlib figure and axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=1)

# Labeling axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Set title
ax.set_title('3D Point Cloud')

# Show the plot
plt.show()
