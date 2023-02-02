# CLASS: 
    # pre_process_image
    # METHODS:
        # __init__
            # INPUT:
                # image_dir = (str) a full path to an image with multiple beetles and possibly a size reference circle
            # OUTPUT(ATTRIBUTES):
                # image_dir = (str) the same directory as is given as an input to the iamge that is being processed
                # image = (np.array) the original compound image
                # grey_image = (np.array) the original compound image in greyscale
                # bw_image = (np.array) the original image in binary black and white
                # inv_bw_image = (np.array) the original image inverted black and white binary
                # clear_inv_bw_image = (np.array) the inverted black and white binary original image with all components touching the border removed
        # segment
            # INPUT:
                # cluster_num = (int) {default=2} the number of clusters used for kmeans to pick only the cluster with alrgest blobs
                # image_edge_buffer = (int) {default=50} number of pixels to add to box borders
            # OUTPUT(ATTRIBUTES):
                # cluster_num = (int) the same as the input
                # image_edge_buffer = (int) the same as the input
                # labeled_image = (np.array) the original compound image that is labelled
                # max_kmeans_label = (int) the label of the cluster with the largest object/blob
                # image_selected_df = (pd.DataFrame) a dataframe with columns describing each segmented image: 
                                                                                # 'centroid' = centre of the image
                                                                                # 'bbox-0' = border 0
                                                                                # 'bbox-1' = border 1
                                                                                # 'bbox-2' = border 2
                                                                                # 'bbox-3' = border 3
                                                                                # 'orientation' = angle of image segment
                                                                                # 'axis_major_length'
                                                                                # 'axis_minor_length'
                                                                                # 'area'
                                                                                # 'area_filled'
                # image_properties_df = (pd.DataFrame) similar to the image_selected_df, but inlcudes all the artefacts that are picked up
                # col_image_lst = (list) a list with all the segmented images in color
                # inv_bw_image_lst = (list) a list with all the segmented images in inverted binary black and white
                # image_segment_count = (int) number of segmented images extracted from the compound image
        # detect_outlier
            # INPUT:
                # None
            # OUTPUT(ATTRIBUTES):
                # image_array = (np.array) an array of the list of color segemented images (number of images, (R,G,B))
                # r_ar_lst = (list) a list of arrays with flattened images red values
                # g_ar_lst = (list) a list of arrays with flattened images green values
                # b_ar_lst = (list) a list of arrays with flattened images blue values
                # all_ar_lst = (list) a list of arrays with flattened images all red, green, and blue values
                # px_dens_dist = (np.array) frequency distribution at 0-255 of all the values for each pixel
                # corr_coef = (np.array) a square array of length equal to the number of segmented images showing the spearman correlation bewteen images
                # corr_pval = (np.array) the pvalues associatedwith each correlation
                # corr_coef_sum = (np.array) the sum of the correlations across each iamge compared to all others
                # outlier_idx = (int) the index of the image with the lowest spearman correlation sum
                # outlier_val = (float) the lowest sum correlation value
                # outlier_col_image = (np.array) the color image of what is detected as the outlier
                # outlier_inv_bw_image = (np.array) the inverted black on white image of the outlier segmented image
                # outlier_bw_image = (np.array) the white on black image of the outlier segmented image
                # image_selected_df = (pd.DataFrame) an updated dataframe that contains the circle identification data
        # estimate_size
            # INPUT:
                # known_radius = (int) {default=1} the radius of the reference circle (shoudl be approximately the same size as the specimens to work best)
                # canny_sigma = (int) {default=5} this describes how strict the cleaning border is for identifying the circle to place over the reference circle
                # outlier_bw_image = (np.array) {default should be self.outlier_bw_image} change this when the circle is falsely detected
                # outlier_idx = (int) {default should be self.outlier_idx} change this when the circle is falsely detected
            # OUTPUT(ATTRIBUTES):
                # outlier_bw_image = (np.array) an updated version of the outlier iamge with a clean circle clear of artifacts
                # outlier_idx = (int) same as the input
                # clean_inv_bw_image_lst = (list) a list of cleaned white on black images no blobs touching hte border
                # image_selected_df = (pd.DataFrame) an update to the dataframe of metadata containing pixel counts and relative area in mm^2 of all segmented images
# *black and white is white on black

# import requirements
import numpy as np
import pandas as pd
from math import ceil
from skimage import io
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import disk
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

class pre_process_image:
    # initialize image to be segmented from path
    def __init__(self, image_dir):
        self.image_dir = image_dir.replace('\\','/') # full directory path to image
        self.image = io.imread(image_dir) # read image from directory
        self.grey_image = rgb2gray(self.image) #convert image to greyscale
        self.bw_image = self.grey_image > threshold_otsu(self.grey_image) # binarize image ot be black & white
        self.inv_bw_image = np.invert(self.bw_image) # invert black and white image
        self.clear_inv_bw_image = clear_border(self.inv_bw_image) # remove anything touching image border
    
    # segment the image into smaller images
    def segment(self, cluster_num=2, image_edge_buffer=50):
        self.cluster_num = cluster_num
        self.image_edge_buffer = image_edge_buffer
        self.labeled_image = label(self.clear_inv_bw_image) #label image
        image_properties_df = pd.DataFrame( # get the properties of each image used to segment blobs in image
            regionprops_table(
                self.labeled_image, 
                properties=('centroid',
                           'bbox',
                           'orientation',
                           'axis_major_length',
                           'axis_minor_length',
                           'area',
                           'area_filled')
                                    )
                                )
        # cluster boxes of blobs by size
        kmean_result = KMeans(n_clusters=cluster_num).fit(
            np.array(
                image_properties_df[['axis_major_length', 'axis_minor_length']]
            )
        )
        image_properties_df['kmeans_label'] = kmean_result.labels_
        # keep only the largest cluster (ball bearing needs to be a similar size as the beetles)
        self.max_kmeans_label = int(image_properties_df.kmeans_label[image_properties_df['area'] == image_properties_df['area'].max()])
        image_selected_df = image_properties_df[image_properties_df['kmeans_label']==self.max_kmeans_label]
        self.image_properties_df = image_properties_df
        # enlarge the boxes around blobs with buffer
        coord_df = image_selected_df.loc[:,['bbox-0','bbox-1','bbox-2','bbox-3']].copy()
        coord_df = coord_df.reset_index(drop = True)
        image_selected_df = image_selected_df.reset_index(drop = True)
        coord_df.loc[:,['bbox-0','bbox-1']] = coord_df.loc[:,['bbox-0','bbox-1']]-self.image_edge_buffer
        coord_df.loc[:,['bbox-2','bbox-3']] = coord_df.loc[:,['bbox-2','bbox-3']]+self.image_edge_buffer
        image_selected_df.loc[:,['bbox-0','bbox-1','bbox-2','bbox-3']] = coord_df.loc[:,['bbox-0','bbox-1','bbox-2','bbox-3']]
        self.image_selected_df = image_selected_df
        # crop blobs from image based on box sizes and add to list
        col_image_lst = []
        inv_bw_image_lst = []
        for i in range(len(image_selected_df)):
            coord_i = image_selected_df.iloc[i]
            # color images
            crop_img = self.image[int(coord_i['bbox-0']):int(coord_i['bbox-2']), int(coord_i['bbox-1']):int(coord_i['bbox-3'])]
            col_image_lst.append(crop_img)
            # inverted black and white images
            crop_bw_img = self.inv_bw_image[int(coord_i['bbox-0']):int(coord_i['bbox-2']), int(coord_i['bbox-1']):int(coord_i['bbox-3'])]
            inv_bw_image_lst.append(crop_bw_img)
        self.col_image_lst = col_image_lst
        self.inv_bw_image_lst = inv_bw_image_lst
        self.image_segment_count = len(col_image_lst)
        
    def detect_outlier(self):
        # convert list to numpy array
        self.image_array = np.copy(np.array(self.col_image_lst, dtype='object'))
        # initialize lists to store data in
        r_ar_lst = []
        g_ar_lst = []
        b_ar_lst = []
        all_ar_lst = []
        for l in range(self.image_segment_count):
            # flatten arrays
            img_var = self.image_array[l]
            r_ar = img_var[:,:,0].flatten() # red
            g_ar = img_var[:,:,1].flatten() # green
            b_ar = img_var[:,:,2].flatten() # blue
            all_ar = img_var.flatten() # all
            # collect data in lists
            r_ar_lst.append(r_ar)
            g_ar_lst.append(g_ar)
            b_ar_lst.append(b_ar)
            all_ar_lst.append(all_ar)
        self.r_ar_lst = r_ar_lst
        self.g_ar_lst = g_ar_lst
        self.b_ar_lst = b_ar_lst
        self.all_ar_lst = all_ar_lst
        # get frequency of values at each rgb value(0-255)
        values_array = all_ar_lst # use all, but can use any color
        temp_dist_ar = np.zeros(shape=(255, self.image_segment_count))
        for i in range(self.image_segment_count):
            unique, counts = np.unique(values_array[i], return_counts=True)
            temp_dict = dict(zip(unique, counts))
            for j in temp_dict.keys():
                temp_dist_ar[j-1][i] = temp_dict[j]
        self.px_dens_dist = temp_dist_ar
        # calculate the spearman correlation of distributions between images
        # use spearman because it is a non-parametric measures
        # use the sum of the correlation coefficients to identify the outlier image
        corr_ar = np.array(spearmanr(temp_dist_ar, axis=0))
        corr_coef_ar = corr_ar[0,:,:]
        corr_pval_ar = corr_ar[1,:,:]
        corr_sum_ar = corr_coef_ar.sum(axis=0)
        self.corr_coef = corr_coef_ar
        self.corr_pval = corr_pval_ar
        self.corr_coef_sum = corr_sum_ar
        self.outlier_idx = corr_sum_ar.argmin()
        self.outlier_val = corr_sum_ar.min()
        self.outlier_col_image = self.col_image_lst[self.outlier_idx]
        self.outlier_inv_bw_image = self.inv_bw_image_lst[self.outlier_idx]
        self.outlier_bw_image = np.invert(self.outlier_inv_bw_image)
        # update metadata dataframe
        self.image_selected_df['circle_class'] = 'non_circle'
        self.image_selected_df.loc[self.outlier_idx, 'circle_class'] = 'circle'
        
    def estimate_size(self, outlier_bw_image, outlier_idx, known_radius=1, canny_sigma=5):
        self.outlier_idx = outlier_idx
        self.outlier_bw_image = outlier_bw_image
        outlier_inv_bw_image = np.invert(outlier_bw_image)
        # remove the border touching blobs of all b&w images
        clean_inv_bw_image_lst = []
        for inv_bw_image in self.inv_bw_image_lst:
            # bw_image = np.invert(inv_bw_image)
            clean_inv_bw_image = clear_border(inv_bw_image)
            clean_inv_bw_image_lst.append(clean_inv_bw_image)
        # default is the image detected with detect_outlier
        # change outlier_bw_image if this is not the ball bearing
        edges = canny(outlier_bw_image, sigma=canny_sigma)
        # Detect radius
        max_r = int((max(outlier_inv_bw_image.shape)/2) + (self.image_edge_buffer/2)) # max radius
        min_r = int((max_r-self.image_edge_buffer) - (self.image_edge_buffer/2)) # min radius
        hough_radii = np.arange(min_r, max_r, 10)
        hough_res = hough_circle(edges, hough_radii)
        # Select the most prominent circle
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        circy, circx = disk((cy[0], cx[0]), radii[0])
        # change the outlier image to fill in the circle
        outlier_inv_bw_image[circy, circx] = True
        self.outlier_inv_bw_image = outlier_inv_bw_image
        clean_inv_bw_image_lst[outlier_idx] = outlier_inv_bw_image
        self.clean_inv_bw_image_lst = clean_inv_bw_image_lst
        # get the area of the ball bearing based on the known radius
        circle_area = np.pi*(known_radius**2)
        px_count_lst = []
        for bw_img in clean_inv_bw_image_lst:
            px_count = np.unique(bw_img, return_counts=True)[1][1]
            px_count_lst.append(px_count)
        self.image_selected_df['pixel_count'] = px_count_lst
        circle_px_count = px_count_lst[outlier_idx]
        area_ar = (np.array(px_count_lst)/circle_px_count)*circle_area
        self.image_selected_df['area'] = area_ar
        # NB If outlier_bw_image is changed from default
        # make sure to update clean_inv_bw_image_lst[selected iamge index] = self.outlier_bw_image
        # outside of the method