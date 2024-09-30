import numpy as np
import cv2  # Only for loading and saving images

# Grayscale Conversion
def to_grayscale(image):
    grayscale = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return grayscale

# Blurring using a basic 3x3 kernel
def blur_image(image):
    kernel = np.ones((3, 3)) / 9.0
    blurred = convolve2d(image, kernel)
    return blurred

# Sobel Edge Detection
def edge_detection(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = convolve2d(image, sobel_x)
    grad_y = convolve2d(image, sobel_y)
    
    edges = np.sqrt(grad_x**2 + grad_y**2)
    edges = np.clip(edges, 0, 255)
    return edges

# Convolution function for applying filters
def convolve2d(image, kernel):
    if len(image.shape) == 2:  # Grayscale image
        image_height, image_width = image.shape
        channels = 1
    else:  # Color image
        image_height, image_width, channels = image.shape

    output = np.zeros_like(image)

    padded_image = np.pad(image, ((kernel.shape[0] // 2, kernel.shape[0] // 2),
                                   (kernel.shape[1] // 2, kernel.shape[1] // 2),
                                   (0, 0)), mode='constant')

    for i in range(image_height):
        for j in range(image_width):
            for c in range(channels):
                output[i, j, c] = np.sum(padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1], c] * kernel)

    return output

# Command-line interface to choose filters
def apply_filter(image, choice):
    if choice == '1':
        print("Applying Grayscale Filter")
        return to_grayscale(image)
    elif choice == '2':
        print("Applying Blur Filter")
        return blur_image(image)
    elif choice == '3':
        print("Applying Edge Detection Filter")
        return edge_detection(image)
    else:
        print("Invalid Choice! Exiting.")
        return None

# Load the image
def main():
    image_path = input("Enter the path of the image file: ")
    image = cv2.imread(image_path)

    if image is None:
        print(" Unable to load image. Please check the file path.")
        return

    image = image.astype(np.float32)
    # image = cv2.resize(image, (512, 512))

    print("Choose a filter to apply:")
    print("1. Grayscale")
    print("2. Blur")
    print("3. Edge Detection")

    choice = input("Enter your choice (1/2/3): ")

    output_image = apply_filter(image, choice)

    if output_image is not None:
        output_image = np.uint8(output_image)
        cv2.imwrite("output_image.png", output_image)
        print("Output saved as 'output_image.png'")

if __name__ == "__main__":
    main()
