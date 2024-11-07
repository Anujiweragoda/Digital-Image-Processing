
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_egg_subimage(image_path):
    img = cv2.imread(image_path)
   
    # Convert to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold 
    _, binary_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours 
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
       
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract the sub-image (ROI) from the original color image
        egg_subimage = img[y:y + h, x:x + w]

        return egg_subimage
    else:
        print("No egg found in the image.")
        return None

def enhance_contrast(image):
    # Apply CLAHE 
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def remove_noise(image):
    # Apply median filter to remove noise
    denoised_image = cv2.medianBlur(image, ksize=3)

    # Apply Gaussian smoothing for further noise reduction
    smoothed_image = cv2.GaussianBlur(denoised_image, (5, 5), sigmaX=0)

    return smoothed_image

def otsu_threshold(smoothed_image):
    _, binary_image = cv2.threshold(smoothed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def is_fertilized(thresh,image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Analyze contours
    for contour in contours:
        # Create a mask for the current contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Compute the mean color inside the contour area
        mean_value = cv2.mean(gray, mask=mask)
       
        
        # The mean value should be close to 0 for a fertilized egg (black area)
        if mean_value[0] < 40: 
            return True
    
    return False

def process_multiple_images(image_paths):
    for idx, image_path in enumerate(image_paths):
        isolated_egg = extract_egg_subimage(image_path)
        if isolated_egg is None:
            continue 
        
        enhanced_egg_subimage = enhance_contrast(isolated_egg)
        final_egg_subimage = remove_noise(enhanced_egg_subimage)
        binary_result = otsu_threshold(final_egg_subimage)

        is_fertilized_result = is_fertilized(binary_result, cv2.imread(image_path))
        fertilized_status = "Fertilized" if is_fertilized_result else "Not Fertilized"

  
        img = cv2.imread(image_path)

        # Plot the results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(isolated_egg, cv2.COLOR_BGR2RGB))
        plt.title("Isolated Egg Subimage")

        plt.subplot(1, 3, 3)
        plt.imshow(binary_result, cmap='gray')
        plt.title(f"Thresholded Image\n({fertilized_status})")

        plt.suptitle(f"Egg Analysis {idx+1} - {fertilized_status}", fontsize=16)
        plt.show()


image_folder = 'imageSet01' 
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png','.webp'))]

image_paths = [os.path.join(image_folder, file) for file in image_files]

process_multiple_images(image_paths)




# Plot the rStepls
"""
image_path = "WhatsApp Image 2024-05-07 at 00.25.23_78be7af1.jpg" 
 
isolated_egg = extract_egg_subimage(image_path)
enhanced_egg_subimage = enhance_contrast(isolated_egg)
final_egg_subimage = remove_noise(enhanced_egg_subimage)
binary_result = otsu_threshold(final_egg_subimage)

# Check if the egg is fertilized or not
is_fertilized_result = is_fertilized(binary_result, cv2.imread(image_path))
fertilized_status = "Fertilized" if is_fertilized_result else "Not Fertilized"


img = cv2.imread(image_path)


plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("original image")

plt.subplot(2,3,2)
plt.imshow(cv2.cvtColor(isolated_egg,cv2.COLOR_BGR2RGB))
plt.title("isolated egg")

plt.subplot(2,3,3)
plt.imshow(enhanced_egg_subimage,cmap='gray')
plt.title("Enhanced Egg image")

plt.subplot(2,3,4)
plt.imshow(final_egg_subimage,cmap='gray')
plt.title(" Egg noice remove")

plt.subplot(2,3,5)
plt.imshow(binary_result,cmap='gray')
plt.title("Binary Image (Otsu)")


plt.show()
"""

