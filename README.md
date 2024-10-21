# Bag of Words Image Recognition Project

This project implements a Bag of Words (BoW) model for image recognition, utilizing feature extraction with Histogram of Oriented Gradients (HOG) and K-means clustering for visual word generation. The main objective of this project is to classify images based on their visual features by representing each image as a histogram of visual words and employing a nearest-neighbor approach for classification.

## Project Structure

```bash
lab2-recognition-assignment/
├── bow.py                  # Functions for generating Bag of Words histograms
├── kmeans.py               # Implementation of K-means clustering for visual word assignment
├── run_experiments.py      # Script for running experiments and evaluating the model
├── setup.sh                # Shell script for setting up the project environment
├── .gitignore              # Git configuration to ignore unnecessary files
└── requirements.txt        # Python dependencies required for the project
```

## Features

This project provides the following core functionalities:

- **HOG-based Feature Extraction**: Extracts visual features from images using the Histogram of Oriented Gradients (HOG).
- **K-means Clustering for Visual Words**: Groups similar HOG features into clusters, each representing a visual word.
- **Bag of Words Representation**: Represents each image as a histogram of visual words.
- **Nearest Neighbor Classification**: Classifies images based on their BoW histograms using the nearest-neighbor algorithm.

## Key Components

1. **HOG Feature Extraction**  
   The Histogram of Oriented Gradients (HOG) is used to extract features from the images. This method captures local edge information by calculating gradient orientations across the image.

2. **K-means Clustering**  
   K-means clustering groups similar HOG features into clusters, each representing a "visual word." This visual vocabulary allows the algorithm to map similar features across different images.

3. **Bag of Words Representation**  
   Each image is represented as a histogram of visual words, where the frequency of each visual word (cluster) in the image forms the histogram. This allows for compact representation and comparison of images.

4. **Nearest Neighbor Classification**  
   Using the Bag of Words histograms, the nearest neighbor classification compares the histograms of test images to those of training images. Based on the closest matches, it classifies images as positive or negative.

## Results & Discussion

The model was tested with various values of K (the number of clusters in K-means) to evaluate its effect on the classification performance. Below are some observations:

- **K = 5**: The model performed well in detecting negative samples, with a negative accuracy of 98%, but struggled with positive samples.
- **K = 7**: This provided the best performance, with a balanced accuracy of 87.75% for positive samples and 96% for negative samples.
- **K = 10 and beyond**: Increasing K beyond 7 did not significantly improve the model's performance. The positive accuracy remained stable, but the negative accuracy began to fluctuate, and overfitting may have occurred.

## Installation and Setup

### Requirements
- Python 3.8+
- Install all dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/berkearda/lab2-recognition-assignment.git
   cd lab2-recognition-assignment
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main experiment script:

   ```bash
   python run_experiments.py
   ```

### Conclusion

This project implements an image classification system using the Bag of Words approach. By leveraging HOG features, K-means clustering, and nearest neighbor classification, the project effectively classifies images. The best performance was achieved with K = 7, highlighting the importance of selecting an optimal number of visual words for robust image classification.
