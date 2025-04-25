import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cv2

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing GUI")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f5f5f5")

        # Main Frame for Layout
        self.main_frame = tk.Frame(root, bg="#e3f2fd")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Image Display Frame
        self.image_frame = tk.Frame(self.main_frame, bg="lightgray", bd=2, relief="solid")
        self.image_frame.pack(side=tk.LEFT, padx=20, pady=10, fill=tk.BOTH, expand=True)

        # Controls Frame
        self.control_frame = tk.Frame(self.main_frame, bg="#ffffff", bd=2, relief="solid")
        self.control_frame.pack(side=tk.RIGHT, padx=20, pady=10, fill=tk.BOTH, expand=True)

        # Placeholder for Images
        self.original_image = None
        self.processed_image = None
        self.display_label = None

        # Header Label
        header_label = tk.Label(self.control_frame, text="Image Processing Operators", font=("Helvetica", 20), bg="#4caf50", fg="white", padx=10, pady=5)
        header_label.pack(fill="x", pady=10)

        # Upload Button
        self.upload_button = ttk.Button(self.control_frame, text="Upload Image", command=self.upload_image, style="TButton")
        self.upload_button.pack(pady=10, ipadx=5)

        # Separator for aesthetics
        ttk.Separator(self.control_frame, orient="horizontal").pack(fill="x", pady=5)

        # Operation Buttons Frame
        self.operation_frame = tk.Frame(self.control_frame, bg="#ffffff")
        self.operation_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create operation buttons
        self.create_operation_buttons()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.original_image = Image.open(file_path)
            self.processed_image = self.original_image
            self.display_image(self.original_image)

    def display_image(self, img):
        img = img.resize((400, 400))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        if self.display_label:
            self.display_label.destroy()
        self.display_label = tk.Label(self.image_frame, image=img_tk, bd=0, bg="lightgray")
        self.display_label.image = img_tk
        self.display_label.pack()

    def create_operation_buttons(self):
        # Customize button colors and sizes for better aesthetics
        button_style = ttk.Style()
        button_style.configure("TButton", font=("Arial", 16), padding=2, background="#4caf50", foreground="black")
        button_style.map("TButton",
                         background=[("active", "#45a049")],  # Darker green on hover
                         foreground=[("active", "white")])

        operations = [
            ("Convert to Grayscale", self.convert_to_grayscale),
            ("Threshold", self.apply_threshold),
            ("Halftone (Simple)", self.apply_halftone_simple),
            ("Halftone (Error Diffusion)", self.apply_halftone_advanced),
            ("Histogram", self.show_histogram),
            ("Histogram Equalization", self.apply_histogram_equalization),
            ("Sobel Edge Detection", self.apply_sobel_operator),
            ("Prewitt Edge Detection", self.apply_prewitt_operator),
            ("Kirsch Edge Detection", self.apply_kirsch_operator),
            ("Homogeneity Operator", self.apply_homogeneity_operator),
            ("Difference Operator", self.apply_difference_operator),
            ("Gaussian Edge Detection", self.apply_difference_of_gaussians),
            ("Contrast Edge Detection", self.apply_contrast_edge_detection),
            ("Variance", self.apply_variance_operator),
            ("Range", self.apply_range_operator),
            ("High-Pass Filter", self.apply_high_pass_filter),
            ("Low-Pass Filter", self.apply_low_pass_filter),
            ("Median Filter", self.apply_median_filter),
            ("Add Image Copy", self.add_image_copy),
            ("Subtract Image Copy", self.subtract_image_copy),
            ("Invert Image", self.invert_image),
            ("Manual Histogram Segmentation", self.apply_manual_histogram),
            ("Peak Histogram Segmentation", self.apply_peak_histogram),
            ("Valley Histogram Segmentation", self.apply_valley_histogram),
            ("Adaptive Histogram Segmentation", self.apply_adaptive_histogram),
        ]

        # Create a grid layout for operation buttons
        num_cols = 2  # Number of columns in the grid
        for i, (op_name, op_func) in enumerate(operations):
            button = ttk.Button(self.operation_frame, text=op_name, command=op_func)
            button.grid(row=i // num_cols, column=i % num_cols, padx=5, pady=3, sticky="ew")

        # Make sure the columns expand proportionally
        for col in range(num_cols):
            self.operation_frame.grid_columnconfigure(col, weight=1)

    # Conversion Operations
    def convert_to_grayscale(self):
        if self.original_image:
            self.processed_image = self.original_image.convert("L")
            self.display_image(self.processed_image)

    # Thresholding with fixed threshold of 128
    def apply_threshold(self):
        if self.original_image:
            grayscale = np.array(self.original_image.convert("L"))
            threshold_value = 128  # Fixed threshold value
            binary_image = (grayscale > threshold_value) * 255
            self.processed_image = Image.fromarray(binary_image.astype(np.uint8))
            self.display_image(self.processed_image)

    # Halftone with mean as the threshold
    def apply_halftone_simple(self):
        if self.original_image:
            grayscale = np.array(self.original_image.convert("L"))
            threshold_value = grayscale.mean()  # Mean threshold value
            halftone_image = (grayscale > threshold_value) * 255
            self.processed_image = Image.fromarray(halftone_image.astype(np.uint8))
            self.display_image(self.processed_image)


    # Halftone (Error Diffusion)
    def apply_halftone_advanced(self):
        if self.original_image:
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)
            halftone = grayscale.copy()
            for y in range(grayscale.shape[0]):
                for x in range(grayscale.shape[1]):
                    old_pixel = halftone[y, x]
                    new_pixel = 255 if old_pixel > 127 else 0
                    halftone[y, x] = new_pixel
                    quant_error = old_pixel - new_pixel
                    if x + 1 < grayscale.shape[1]:
                        halftone[y, x + 1] += quant_error * 7 / 16
                    if y + 1 < grayscale.shape[0]:
                        if x > 0:
                            halftone[y + 1, x - 1] += quant_error * 3 / 16
                        halftone[y + 1, x] += quant_error * 5 / 16
                        if x + 1 < grayscale.shape[1]:
                            halftone[y + 1, x + 1] += quant_error * 1 / 16
            self.processed_image = Image.fromarray(halftone.astype(np.uint8))
            self.display_image(self.processed_image)

    # Histogram
    def show_histogram(self):
        # Use the currently displayed image
        target_image = self.processed_image if self.processed_image else self.original_image

        if target_image:
            # Convert image to grayscale and compute the histogram
            grayscale = np.array(target_image.convert("L"))
            histogram, bins = np.histogram(grayscale, bins=256, range=(0, 256))
            
            # Plot the histogram
            plt.figure(figsize=(8, 6))
            plt.bar(bins[:-1], histogram, width=1.0, edgecolor='black', align='edge')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.title('Image Histogram')
            plt.grid(True)
            plt.tight_layout()
            
            # Show the plot
            plt.show()



    def apply_histogram_equalization(self):
        if self.original_image:
            grayscale = np.array(self.original_image.convert("L"))
            histogram, bins = np.histogram(grayscale.flatten(), 256, [0, 256])
            cdf = histogram.cumsum()
            cdf_normalized = cdf * (255 / cdf[-1])
            equalized = cdf_normalized[grayscale.astype(int)]
            self.processed_image = Image.fromarray(equalized.astype(np.uint8))
            self.display_image(self.processed_image)

    # Sobel Operator
    def apply_sobel_operator(self):
        if self.original_image:
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            Gx = convolve2d(grayscale, Kx, boundary='symm', mode='same')
            Gy = convolve2d(grayscale, Ky, boundary='symm', mode='same')
            sobel = np.sqrt(Gx**2 + Gy**2)
            sobel = (sobel / sobel.max()) * 255
            self.processed_image = Image.fromarray(sobel.astype(np.uint8))
            self.display_image(self.processed_image)

    # Prewitt Operator
    def apply_prewitt_operator(self):
        if self.original_image:
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)
            Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            Ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            Gx = convolve2d(grayscale, Kx, boundary='symm', mode='same')
            Gy = convolve2d(grayscale, Ky, boundary='symm', mode='same')
            prewitt = np.sqrt(Gx**2 + Gy**2)
            prewitt = (prewitt / prewitt.max()) * 255
            self.processed_image = Image.fromarray(prewitt.astype(np.uint8))
            self.display_image(self.processed_image)

    # Kirsch Operator
    def apply_kirsch_operator(self):
        if self.original_image:
            # Convert the image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)

            # Define Kirsch compass masks
            kernels = [
                np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # 0 degrees (East)
                np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # 45 degrees (Northeast)
                np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # 90 degrees (North)
                np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # 135 degrees (Northwest)
                np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # 180 degrees (West)
                np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # 225 degrees (Southwest)
                np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # 270 degrees (South)
                np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])   # 315 degrees (Southeast)
            ]

            # Apply each kernel to the image
            responses = [convolve2d(grayscale, kernel, boundary='symm', mode='same') for kernel in kernels]

            # Compute the maximum response (edge magnitude) and the direction index
            magnitude = np.max(responses, axis=0)  # Maximum response across all directions
            direction = np.argmax(responses, axis=0)  # Direction index corresponding to the maximum

            # Normalize the magnitude to 0-255 for display
            magnitude_normalized = (magnitude / magnitude.max()) * 255

            # Map direction indices to angles (0° to 315°)
            angle_map = np.array([0, 45, 90, 135, 180, 225, 270, 315])
            edge_directions = angle_map[direction]

            # Convert magnitude to image for display
            magnitude_image = Image.fromarray(magnitude_normalized.astype(np.uint8))
            self.processed_image = magnitude_image
            self.display_image(magnitude_image)


    # Homogeneity Operator (Example Implementation)
    def apply_homogeneity_operator(self):
        if self.original_image:
            # Convert the image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)

            # Pad the grayscale image to handle borders
            padded_image = np.pad(grayscale, pad_width=1, mode='reflect')

            # Initialize the output array
            homogeneity = np.zeros_like(grayscale)

            # Iterate through the image, applying the operation
            for i in range(1, padded_image.shape[0] - 1):
                for j in range(1, padded_image.shape[1] - 1):
                    # Extract the 3x3 neighborhood
                    local_window = padded_image[i - 1:i + 2, j - 1:j + 2]
                    center_pixel = padded_image[i, j]

                    # Compute the maximum absolute difference
                    max_difference = np.max(np.abs(local_window - center_pixel))
                    homogeneity[i - 1, j - 1] = max_difference

            # Normalize the result to range [0, 255]
            homogeneity_normalized = (homogeneity / homogeneity.max()) * 255

            # Convert to an image and display
            self.processed_image = Image.fromarray(homogeneity_normalized.astype(np.uint8))
            self.display_image(self.processed_image)


    # Difference Operator
    def apply_difference_operator(self):
        if self.original_image:
            # Convert the image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)

            # Pad the grayscale image to handle borders
            padded_image = np.pad(grayscale, pad_width=1, mode='reflect')

            # Initialize the output array
            difference = np.zeros_like(grayscale)

            # Iterate through the image, applying the operation
            for i in range(1, padded_image.shape[0] - 1):
                for j in range(1, padded_image.shape[1] - 1):
                    # Extract the 3x3 neighborhood
                    local_window = padded_image[i - 1:i + 2, j - 1:j + 2]
                    center_pixel = padded_image[i, j]

                    # Compute the sum of absolute differences
                    total_difference = np.sum(np.abs(local_window - center_pixel))
                    difference[i - 1, j - 1] = total_difference

            # Normalize the result to range [0, 255]
            difference_normalized = (difference / difference.max()) * 255

            # Convert to an image and display
            self.processed_image = Image.fromarray(difference_normalized.astype(np.uint8))
            self.display_image(self.processed_image)


    # Difference of Gaussians (Example Implementation)
    def apply_difference_of_gaussians(self):
        if self.original_image:
            # Convert image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)

            # Define the 7x7 and 9x9 masks
            kernel_7x7 = np.array([
                [0, 0, -1, -1, -1, 0, 0],
                [0, -2, -3, -3, -3, -2, 0],
                [-1, -3, 5, 5, 5, -3, -1],
                [-1, -3, 5, 16, 5, -3, -1],
                [-1, -3, 5, 5, 5, -3, -1],
                [0, -2, -3, -3, -3, -2, 0],
                [0, 0, -1, -1, -1, 0, 0]
            ])

            kernel_9x9 = np.array([
                [0, 0, 0, -1, -1, -1, 0, 0, 0],
                [0, 0, -2, -3, -3, -3, -2, 0, 0],
                [0, -2, -3, 5, 5, 5, -3, -2, 0],
                [-1, -3, 5, 9, 9, 9, 5, -3, -1],
                [-1, -3, 5, 9, 19, 9, 5, -3, -1],
                [-1, -3, 5, 9, 9, 9, 5, -3, -1],
                [0, -2, -3, 5, 5, 5, -3, -2, 0],
                [0, 0, -2, -3, -3, -3, -2, 0, 0],
                [0, 0, 0, -1, -1, -1, 0, 0, 0]
            ])

            # Convolve the grayscale image with each kernel
            response_7x7 = convolve2d(grayscale, kernel_7x7, boundary='symm', mode='same')
            response_9x9 = convolve2d(grayscale, kernel_9x9, boundary='symm', mode='same')

            # Compute the Difference of Gaussians
            dog_result = response_9x9 - response_7x7

           # Apply thresholding to enhance edges
            threshold_value = 50  # You can adjust this value as needed
            dog_result[dog_result < threshold_value] = 0  # Suppress weak edges
            dog_result[dog_result >= threshold_value] = 255  # Enhance strong edges

            # Scale the result for better visibility
            scale_factor = 1.5  # You can adjust this as needed
            dog_result = np.clip(dog_result * scale_factor, 0, 255).astype(np.uint8)

            # Convert the result to a PIL image for visualization
            self.processed_image = Image.fromarray(dog_result)

            # Display the processed image
            self.display_image(self.processed_image)




    # Contrast Edge Detection (Example Implementation)
    def apply_contrast_edge_detection(self):
        if self.original_image:
            # Convert image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)

            # Define the smoothing mask and edge detection mask
            smoothing_mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
            edge_mask = np.array([[-1, 0, -1], [0, 4, 0], [-1, 0, -1]])

            # Step 1: Apply smoothing to reduce noise
            smoothed = convolve2d(grayscale, smoothing_mask, boundary='symm', mode='same')

            # Step 2: Apply edge detection to the smoothed image
            edge_detected = convolve2d(smoothed, edge_mask, boundary='symm', mode='same')

            # Normalize the result to fit within the range [0, 255]
            edge_detected = (edge_detected - edge_detected.min()) / (edge_detected.max() - edge_detected.min()) * 255

            # Convert to image and display
            self.processed_image = Image.fromarray(edge_detected.astype(np.uint8))
            self.display_image(self.processed_image)


    # Variance (Example Implementation)
    def apply_variance_operator(self):
        if self.original_image:
            # Convert the image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)

            # Define a simple averaging kernel
            kernel = np.ones((3, 3)) / 9

            # Compute the mean of the squared intensities
            mean_of_squares = convolve2d(grayscale ** 2, kernel, boundary='symm', mode='same')

            # Compute the square of the mean intensities
            square_of_mean = convolve2d(grayscale, kernel, boundary='symm', mode='same') ** 2

            # Compute the variance
            variance = mean_of_squares - square_of_mean

            # Normalize the variance to the range [0, 255] for display
            variance_normalized = (variance / variance.max()) * 255

            # Convert the result to an image
            self.processed_image = Image.fromarray(variance_normalized.astype(np.uint8))
            self.display_image(self.processed_image)


    # Range (Example Implementation)
    def apply_range_operator(self):
        if self.original_image:
            # Convert the image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)

            # Define the sliding window size (3x3)
            padded_image = np.pad(grayscale, pad_width=1, mode='reflect')

            # Initialize an array to store the range result
            range_result = np.zeros_like(grayscale)

            # Compute the range for each pixel
            for i in range(1, padded_image.shape[0] - 1):
                for j in range(1, padded_image.shape[1] - 1):
                    local_window = padded_image[i - 1:i + 2, j - 1:j + 2]
                    range_result[i - 1, j - 1] = np.max(local_window) - np.min(local_window)

            # Normalize the range result to 0-255
            range_normalized = (range_result / range_result.max()) * 255

            # Convert the result to an image
            self.processed_image = Image.fromarray(range_normalized.astype(np.uint8))
            self.display_image(self.processed_image)


    # High-Pass Filter
    def apply_high_pass_filter(self):
        if self.original_image:
            # Ensure the image is in grayscale
            if self.original_image.mode != 'L':
                grayscale = self.original_image.convert('L')
            else:
                grayscale = self.original_image

            # Convert the grayscale image to a NumPy array
            img_array = np.array(grayscale, dtype=np.float32)
            
            # Define the high-pass filter kernel
            kernel = np.array([
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]
            ], dtype=np.float32)
            
            # Perform convolution manually
            pad_size = kernel.shape[0] // 2
            padded_image = np.pad(img_array, pad_size, mode='reflect')  # Reflect padding to handle edges
            filtered_array = np.zeros_like(img_array)

            # Convolution
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    region = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
                    filtered_array[i, j] = np.sum(region * kernel)

            # Clip the values to the range [0, 255]
            filtered_array = np.clip(filtered_array, 0, 255)

            # Convert the result back to a PIL image
            self.processed_image = Image.fromarray(filtered_array.astype(np.uint8))
            
            # Display the processed image
            self.display_image(self.processed_image)

    # Low-Pass Filter
    def apply_low_pass_filter(self):
        if self.original_image:
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)
            kernel = np.ones((5, 5)) / 25  # Simple averaging kernel
            low_pass = convolve2d(grayscale, kernel, boundary='symm', mode='same')
            low_pass = (low_pass / low_pass.max()) * 255
            self.processed_image = Image.fromarray(low_pass.astype(np.uint8))
            self.display_image(self.processed_image)

    # Median Filter (Using a simple method with scipy)
    def apply_median_filter(self):
        if self.original_image:
            grayscale = np.array(self.original_image.convert("L"), dtype=np.float32)

            # Pad the grayscale image to handle borders
            padded_image = np.pad(grayscale, pad_width=1, mode="reflect")

            # Create an output array to store the median-filtered image
            median_filtered = np.zeros_like(grayscale)

            # Manually compute the median for each 3x3 neighborhood
            for i in range(1, padded_image.shape[0] - 1):
                for j in range(1, padded_image.shape[1] - 1):
                    # Extract the 3x3 neighborhood
                    local_window = padded_image[i - 1:i + 2, j - 1:j + 2]
                    
                    # Compute the median and assign it to the output array
                    median_filtered[i - 1, j - 1] = np.median(local_window)

            # Convert the result to an image and display
            self.processed_image = Image.fromarray(median_filtered.astype(np.uint8))
            self.display_image(self.processed_image)


    # Add Image Copy (Add the image to itself)
    # Add Image Copy
    def add_image_copy(self):
        if self.original_image:
            # Step 1: Convert the original image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.uint8)
            
            # Step 2: Add the grayscale image to itself
            added_image = np.clip(grayscale + grayscale, 0, 255)  # Ensure values are in the range [0, 255]
            
            # Step 3: Convert the result back to an image
            self.processed_image = Image.fromarray(added_image.astype(np.uint8))
            self.display_image(self.processed_image)

    # Subtract Image Copy
    def subtract_image_copy(self):
        if self.original_image:
            # Step 1: Convert the original image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.uint8)
            
            # Step 2: Subtract a copy of the grayscale image from itself
            subtracted_image = np.clip(grayscale - grayscale, 0, 255)  # Result will be all zeros
            
            # Convert the result back to an image
            self.processed_image = Image.fromarray(subtracted_image.astype(np.uint8))
            self.display_image(self.processed_image)


    # Invert Image
    def invert_image(self):
        if self.original_image:
            inverted = ImageOps.invert(self.original_image)
            self.processed_image = inverted
            self.display_image(self.processed_image)

    # Manual Histogram Segmentation
    def apply_manual_histogram(self):
        if self.original_image:
            grayscale = np.array(self.original_image.convert("L"))
            threshold = 128  # Example threshold value
            manual_segmented = (grayscale > threshold) * 255
            self.processed_image = Image.fromarray(manual_segmented.astype(np.uint8))
            self.display_image(self.processed_image)

    # Peak Histogram Segmentation
    def apply_peak_histogram(self):
        if self.original_image:
            # Convert the image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.uint8)

            # Step 1: Compute Histogram
            histogram, bin_edges = np.histogram(grayscale, bins=256, range=(0, 255))

            # Step 2: Find Peaks in the Histogram
            peaks, _ = find_peaks(histogram, distance=10)  # Minimum distance to separate peaks

            # Step 3: Sort Peaks by Intensity (Descending Order)
            sorted_peaks = sorted(peaks, key=lambda p: histogram[p], reverse=True)

            if len(sorted_peaks) < 2:
                raise ValueError("Not enough peaks found for thresholding. Check the image.")

            # Step 4: Select Background and Object Peaks
            background_peak = sorted_peaks[0]  # Highest peak (assumed to be the background)
            object_peak = sorted_peaks[1]     # Second highest peak (assumed to be the object)

            # Ensure peaks are in ascending order
            if background_peak > object_peak:
                background_peak, object_peak = object_peak, background_peak

            # Step 5: Calculate Threshold
            threshold = (background_peak + object_peak) // 2  # Midpoint between the two peaks

            # Step 6: Apply Thresholding
            peak_segmented = (grayscale > threshold) * 255

            # Convert the result to a PIL image
            self.processed_image = Image.fromarray(peak_segmented.astype(np.uint8))
            
            # Display the processed image
            self.display_image(self.processed_image)

    # Valley Histogram Segmentation
    def apply_valley_histogram(self):
        if self.original_image:
            # Convert the image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.uint8)

            # Step 1: Calculate the Histogram
            histogram, bin_edges = np.histogram(grayscale, bins=256, range=(0, 255))

            # Step 2: Detect Peaks in the Histogram
            peaks, _ = find_peaks(histogram, distance=10)  # Minimum distance between peaks

            # Step 3: Sort Peaks by Prominence (Descending Order)
            sorted_peaks = sorted(peaks, key=lambda p: histogram[p], reverse=True)

            if len(sorted_peaks) < 2:
                raise ValueError("Not enough peaks found for valley detection. Check the image.")

            # Step 4: Find the Valley Between Peaks
            background_peak = sorted_peaks[0]  # Highest peak (background)
            object_peak = sorted_peaks[1]     # Second highest peak (object)

            # Ensure peaks are in ascending order for valley detection
            if background_peak > object_peak:
                background_peak, object_peak = object_peak, background_peak

            # Find the valley (minimum histogram value) between the two peaks
            valley_index = np.argmin(histogram[background_peak:object_peak]) + background_peak

            # Step 5: Segment the Image
            # Pixels below the valley are set to 0 (background)
            # Pixels above the valley retain their intensity
            valley_segmented = np.where(grayscale > valley_index, grayscale, 0)

            # Convert the segmented image back to a PIL image
            self.processed_image = Image.fromarray(valley_segmented.astype(np.uint8))
            
            # Display the processed image
            self.display_image(self.processed_image)

    # Adaptive Histogram Equalization
    def apply_adaptive_histogram(self, region_size=32):
        if self.original_image:
            # Step 1: Convert the image to grayscale
            grayscale = np.array(self.original_image.convert("L"), dtype=np.uint8)
            height, width = grayscale.shape

            # Prepare an empty array for the segmented image
            segmented_image = np.zeros_like(grayscale)

            # Step 2: Divide the image into regions
            for i in range(0, height, region_size):
                for j in range(0, width, region_size):
                    # Extract a sub-region
                    region = grayscale[i:i+region_size, j:j+region_size]

                    # Step 3: Calculate the Histogram
                    histogram, _ = np.histogram(region, bins=256, range=(0, 255))

                    # Step 4: Detect Peaks in the Histogram
                    peaks, _ = find_peaks(histogram, prominence=5)

                    if len(peaks) < 2:
                        # If less than two peaks, use the global mean as the threshold
                        global_threshold = grayscale.mean()
                        segmented_region = np.where(region > global_threshold, 255, 0)
                        segmented_image[i:i+region_size, j:j+region_size] = segmented_region
                        continue

                    # Step 5: Sort Peaks by Prominence
                    sorted_peaks = sorted(peaks, key=lambda p: histogram[p], reverse=True)

                    # Step 6: Find the Valley Between Peaks
                    background_peak = sorted_peaks[0]
                    object_peak = sorted_peaks[1]

                    if background_peak > object_peak:
                        background_peak, object_peak = object_peak, background_peak

                    valley_index = np.argmin(histogram[background_peak:object_peak + 1]) + background_peak

                    # Step 7: First-Pass Segmentation
                    first_pass_segment = np.where(region > valley_index, region, 0)

                    # Step 8: Calculate New Thresholds from Mean Segments
                    object_mean = (
                        first_pass_segment[first_pass_segment > 0].mean()
                        if np.any(first_pass_segment > 0)
                        else valley_index
                    )
                    background_mean = (
                        region[region <= valley_index].mean()
                        if np.any(region <= valley_index)
                        else valley_index
                    )

                    # Step 9: Adjust Threshold Using New Means
                    adaptive_threshold = (object_mean + background_mean) / 2

                    # Step 10: Second-Pass Segmentation
                    segmented_region = np.where(region > adaptive_threshold, 255, 0)
                    segmented_image[i:i+region_size, j:j+region_size] = segmented_region

            # Convert the segmented image back to a PIL image
            self.processed_image = Image.fromarray(segmented_image.astype(np.uint8))

            # Display the processed image
            self.display_image(self.processed_image)

# Main program
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()