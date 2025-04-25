# Image Processing GUI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green)
![Pillow](https://img.shields.io/badge/Image_Pillow-9.0+-orange)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive desktop application for performing various image processing operations with real-time previews.

![image](https://github.com/user-attachments/assets/c02199a8-ec35-4cd7-85fb-a6c2679a5fbd)

## Features

### Core Operations
- **Color Manipulation**
  - Grayscale conversion
  - Image inversion
  - Thresholding (fixed and adaptive)
  
- **Halftoning Techniques**
  - Simple threshold halftone
  - ![image](https://github.com/user-attachments/assets/facf2372-f764-48d6-bd9b-72eae669bdb0)

  - Error diffusion (Floyd-Steinberg)
  - ![image](https://github.com/user-attachments/assets/69353cdf-76cc-4670-b5dd-8371ad253ecd)


### Edge Detection
- Sobel operator
- ![image](https://github.com/user-attachments/assets/309f5687-abd9-421f-a86a-35cfb26b719f)

- Prewitt operator
- ![image](https://github.com/user-attachments/assets/591c0dab-c379-466f-afd4-6c36a6de2624)

- Kirsch compass operator
- ![image](https://github.com/user-attachments/assets/47e7ecdf-60db-4d1a-8813-9a96e02f9dd3)

- Homogeneity operator
- ![image](https://github.com/user-attachments/assets/2345efbd-67f8-42c3-a3f3-fe9181ad994a)

- Difference operator
- ![image](https://github.com/user-attachments/assets/34c9a370-04ea-48ae-a903-bc4eb0fe248e)


### Spatial Filters
- High-pass filter
- ![image](https://github.com/user-attachments/assets/17839181-0a05-485a-ad7d-3b274d9fb97a)

- Low-pass filter
- ![image](https://github.com/user-attachments/assets/eb8d7413-92d0-4676-8787-73643e1caa94)

- Median filter
- ![image](https://github.com/user-attachments/assets/f18eac27-96b3-497d-b30b-13f1e62d8e55)

- Gaussian edge detection
- Contrast edge detection
- ![image](https://github.com/user-attachments/assets/cb0c099d-c633-450d-aa85-a9596fa7ca6f)


### Histogram Operations
- Histogram visualization
- ![image](https://github.com/user-attachments/assets/3ba44432-ce8e-4da1-a9d3-398746d97530)

- Histogram equalization
- ![image](https://github.com/user-attachments/assets/77f2fc80-f927-4787-b643-fc7ba6faf179)

- Manual segmentation
- Peak-based segmentation
- Valley-based segmentation
- Adaptive segmentation

### Image Arithmetic
- Image addition
- Image subtraction
- Variance operator
- Range operator

## Installation

1. **Prerequisites**:
   - Python 3.8 or higher
   - pip package manager

2. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/image-processing-gui.git
   cd image-processing-gui
