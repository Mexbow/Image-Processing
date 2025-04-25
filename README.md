# Image Processing GUI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green)
![Pillow](https://img.shields.io/badge/Image_Pillow-9.0+-orange)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive desktop application for performing various image processing operations with real-time previews.

![Application Screenshot](screenshot.png)

## Features

### Core Operations
- **Color Manipulation**
  - Grayscale conversion
  - Image inversion
  - Thresholding (fixed and adaptive)
  
- **Halftoning Techniques**
  - Simple threshold halftone
  - Error diffusion (Floyd-Steinberg)

### Edge Detection
- Sobel operator
- Prewitt operator
- Kirsch compass operator
- Homogeneity operator
- Difference operator

### Spatial Filters
- High-pass filter
- Low-pass filter
- Median filter
- Gaussian edge detection
- Contrast edge detection

### Histogram Operations
- Histogram visualization
- Histogram equalization
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
