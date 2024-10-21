import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse
import logging
import csv

def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    h, w = img.shape

    if nPointsX <= 1 or nPointsY <= 1:
        raise ValueError("nPointsX and nPointsY must be greater than 1 to form a grid.")
    if border >= w // 2 or border >= h // 2:
        raise ValueError("Border is too large relative to the image size.")

    # Compute the step size between grid points
    stepX = (w - 2 * border) // (nPointsX - 1)
    stepY = (h - 2 * border) // (nPointsY - 1)

    # Generate grid points
    vPoints = np.array([[border + j * stepX, border + i * stepY] 
                            for i in range(nPointsY) 
                            for j in range(nPointsX)])
    return vPoints


def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    # Compute gradient magnitudes and angles (in degrees)
    magnitude = np.hypot(grad_x, grad_y)
    angle = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Convert to degrees
    angle[angle < 0] += 180  # Ensure angles are between 0 and 180 degrees

    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        center_x = round(vPoints[i, 0])
        center_y = round(vPoints[i, 1])

        desc = []
        for cell_y in range(-2, 2):
            for cell_x in range(-2, 2):
                start_y = center_y + (cell_y) * h
                end_y = center_y + (cell_y + 1) * h

                start_x = center_x + (cell_x) * w
                end_x = center_x + (cell_x + 1) * w

                # TODO
                # compute the angles
                cell_magnitude = magnitude[start_y:end_y, start_x:end_x]
                cell_angle = angle[start_y:end_y, start_x:end_x]
                # compute the histogram
                hist, _ = np.histogram(cell_angle, bins=nBins, range=(0, 180), weights=cell_magnitude)
                desc.extend(hist)
                ...

        descriptors.append(desc)

    # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)
    descriptors = np.asarray(descriptors)
    return descriptors


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + \
        sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    vFeatures = []
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [h, w, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        # TODO start
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        # [100, 128]
        features = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        # all features from one image [n_vPoints, 128] (100 grid points)
        vFeatures.append(features)
        # TODO end

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    # [n_imgs*n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])
    print('number of extracted features: ', len(vFeatures))

    # Cluster the features using K-Means
    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    num_centers = vCenters.shape[0]
    histo = np.zeros(num_centers, dtype=np.float32)

    # TODO
    distances = np.linalg.norm(vFeatures[:, np.newaxis] - vCenters, axis=2)
    closest_center_idxs = np.argmin(distances, axis=1)
    histo = np.bincount(closest_center_idxs, minlength=num_centers)
    total_count = np.sum(histo)

    if total_count > 0:
        histo = histo.astype(np.float32) / total_count

    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        img = cv2.imread(vImgNames[i])  # [h, w, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        vPoints = grid_points(img, nPointsX, nPointsY, border)
        vFeatures = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        bowHist = bow_histogram(vFeatures, vCenters)
        vBoW.append(bowHist)
    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW


def bow_recognition_nearest(histogram, vBoWPos, vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = None, None

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    # TODO
    DistPos = np.min(np.linalg.norm(histogram - vBoWPos, axis=1))
    DistNeg = np.min(np.linalg.norm(histogram - vBoWNeg, axis=1))
    
    if (DistPos < DistNeg):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel


if __name__ == '__main__':

    # Use argparse to allow the script to accept hyperparameters from the command line
    parser = argparse.ArgumentParser(description='Bag of Words (BoW) using KMeans and HOG descriptors')
    
    # Add arguments for k and numiter
    parser.add_argument('--k', type=int, default=48, help='Number of k-means clusters')
    parser.add_argument('--numiter', type=int, default=300, help='Maximum number of iterations for k-means')
    parser.add_argument('--output', type=str, default='results.csv', help='File to save results to')

    args = parser.parse_args()
    
    # Retrieve the values from the command-line arguments
    k = args.k
    numiter = args.numiter
    output_file = args.output

    # Log the parameters
    logging.info(f'Starting BoW with k={k}, numiter={numiter}')

    # set a fixed random seed
    np.random.seed(42)
    
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)
    print('codebook created.')
    print('codebook shape:', vCenters.shape)
    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(
        nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(
            vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(
        nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(
            vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)

    # Get the number of samples in each test group
    n_pos_test = vBoWPos_test.shape[0]
    n_neg_test = vBoWNeg_test.shape[0]

    # Calculate the weighted average accuracy
    avg_accuracy = ((n_pos_test * acc_pos) + (n_neg_test * acc_neg)) / (n_pos_test + n_neg_test)
    logging.info(f'Average weighted accuracy for k={k} and numiter={numiter}: {avg_accuracy:.4f}')

    # Save the results to a CSV file
    with open(output_file, 'a', newline='') as csvfile:
        result_writer = csv.writer(csvfile)
        result_writer.writerow([k, numiter, acc_pos, acc_neg, avg_accuracy])

    logging.info(f'Results saved to {output_file}')