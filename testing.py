from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = "durian_dataset/test/D13/20200815_143926.jpg"  # Replace with your image path
image = Image.open(image_path)
image_np = np.array(image)

# Step 1: Separate the image into its RGB channels
r_channel, g_channel, b_channel = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]

# Step 2: Detect the object in the image using thresholding or edge detection
gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
_, threshold = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Step 3: Segment the object using contours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmented_image = cv2.drawContours(np.zeros_like(gray_image), contours, -1, (255), thickness=cv2.FILLED)

# Step 4: Find the dimension of the image in pixels
height, width = image_np.shape[:2]

# Step 5: Measure the size of the detected object
object_sizes = [cv2.contourArea(contour) for contour in contours]

# Display the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(2, 3, 2)
plt.title("Red Channel")
plt.imshow(r_channel, cmap='Reds')

plt.subplot(2, 3, 3)
plt.title("Green Channel")
plt.imshow(g_channel, cmap='Greens')

plt.subplot(2, 3, 4)
plt.title("Blue Channel")
plt.imshow(b_channel, cmap='Blues')

plt.subplot(2, 3, 5)
plt.title("Thresholded Image")
plt.imshow(threshold, cmap='gray')

plt.subplot(2, 3, 6)
plt.title("Segmented Object")
plt.imshow(segmented_image, cmap='gray')

plt.tight_layout()
plt.show()

(height, width, object_sizes)