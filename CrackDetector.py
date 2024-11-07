import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


input_folder = "imagest02"

image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]


for image_file in image_files:
    # Load the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error loading image {image_file}")
        continue

    # Split into color channels
    b, g, r = cv2.split(image)

    gray_red = r  
    gray_green = g  

    # Apply Gaussian Blur to the red channel
    blurred_red = cv2.GaussianBlur(gray_red, (11, 11), 0)

    # Threshold the blurred red channel to create a binary mask for the egg
    _, binary_image = cv2.threshold(blurred_red, 150, 255, cv2.THRESH_BINARY)

    # Detect edges in the green channel 
    edges = cv2.Canny(gray_green, 50, 250)

    # Apply morphological closing to the edges to close small gaps
    kernel = np.ones((3, 3), np.uint8)
    morph_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    #final_image = cv2.subtract(binary_image, morph_closed)

    masked_cracks_inside = cv2.bitwise_and(morph_closed, morph_closed, mask=binary_image)

    # Find contours of cracks inside the egg
    contours_inside, _ = cv2.findContours(masked_cracks_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    min_area = 500  
    crack_contours_inside = [c for c in contours_inside if cv2.contourArea(c) > min_area]

   
    crack_image_inside = image.copy()

    # Draw the detected cracks 
    if crack_contours_inside:
        cv2.drawContours(crack_image_inside, crack_contours_inside, -1, (0, 255, 0), 3)  # Draw inside cracks in red
        crack_detected = True
    else:
        crack_detected = False
    

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image: {image_file}")
    plt.axis('off')

    if crack_detected:
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(crack_image_inside, cv2.COLOR_BGR2RGB))
        plt.title("Detected Cracks In Egg")
        plt.axis('off')
    else:
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, 'No Crack Detected In Egg', fontsize=12, ha='center', va='center')
        plt.title("Crack Detection Result")
        plt.axis('off')

    plt.show()

