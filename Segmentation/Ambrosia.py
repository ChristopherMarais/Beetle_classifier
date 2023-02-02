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
                # cluster_num = () {default=2}
                # image_edge_buffer = () {default=50}
            # OUTPUT(ATTRIBUTES):
                # cluster_num = ()
                # image_edge_buffer = ()
                # labeled_image = ()
                # max_kmeans_label = ()
                # image_selected_df = ()
                # col_image_lst = ()
                # inv_bw_image_lst = ()
                # image_segment_count = ()
        # detect_outlier
            # INPUT:
                # None
            # OUTPUT(ATTRIBUTES):
                # image_array = ()
                # r_ar_lst = ()
                # g_ar_lst = ()
                # b_ar_lst = ()
                # all_ar_lst = ()
                # px_dens_dist = ()
                # corr_coef = ()
                # self.corr_coef = ()
                # corr_pval = ()
                # corr_coef_sum = ()
                # outlier_idx = ()
                # outlier_val = ()
                # outlier_col_image = ()
                # outlier_inv_bw_image = ()
                # outlier_bw_image = ()
                # image_selected_df = ()
        # estimate_size
            # INPUT:
                # known_radius = () {default=1}
                # canny_sigma = () {default=5}
                # outlier_bw_image = () {default=self.outlier_bw_image}
                # outlier_idx = () {default=self.outlier_idx}
            # OUTPUT(ATTRIBUTES):
                # outlier_bw_image
                # outlier_idx
                # clean_bw_image_lst
                # outlier_bw_image
                # image_selected_df

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
        self.bw_image = self.grey_image > threshold_otsu(image_gray) # binarize image ot be black & white
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
        kmean_result = KMeans(n_clusters=clust_count).fit(
            np.array(
                image_properties_df[['axis_major_length', 'axis_minor_length']]
            )
        )
        image_properties_df['kmeans_label'] = kmean_result.labels_
        # keep only the largest cluster (ball bearing needs to be a similar size as the beetles)
        self.max_kmeans_label = int(image_properties_df.kmeans_label[image_properties_df['area'] == image_properties_df['area'].max()])
        image_selected_df = image_properties_df[image_properties_df['kmeans_label']==self.max_kmeans_label]
        # enlarge the boxes around blobs with buffer
        coord_df = image_selected_df.loc[:,['bbox-0','bbox-1','bbox-2','bbox-3']].copy()
        coord_df = coord_df.reset_index()
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
            crop_img = image[coord_i['bbox-0']:coord_i['bbox-2'], coord_i['bbox-1']:coord_i['bbox-3']]
            col_image_lst.append(crop_img)
            # inverted black and white images
            crop_bw_img = self.inv_bw_image[coord_i['bbox-0']:coord_i['bbox-2'], coord_i['bbox-1']:coord_i['bbox-3']]
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
            img_var = image_array[l]
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
        self.image_selected_df['circle_class'] = 'beetle'
        self.image_selected_df['circle_class'][self.outlier_idx] = 'circle'
        
    def estimate_size(self, known_radius=1 canny_sigma=5, outlier_bw_image=self.outlier_bw_image, outlier_idx=self.outlier_idx):
        self.outlier_bw_image = outlier_bw_image
        self.outlier_idx = outlier_idx
        # remove the border touching blobs of all b&w images
        clean_bw_image_lst = []
        for inv_bw_image in self.inv_bw_image_lst:
            bw_image = np.invert(inv_bw_image)
            clean_bw_image = clear_border(bw_image)
            clean_bw_image_lst.append(clean_bw_image)
        self.clean_bw_image_lst = clean_bw_image_lst
        # default is the image detected with detect_outlier
        # change outlier_bw_image if this is not the ball bearing
        edges = canny(outlier_bw_image, sigma=canny_sigma)
        # Detect radius
        max_r = int((max(bw_inv_img.shape)/2) + (buffer_size/2)) # max radius
        min_r = int((max_r-buffer_size) - (buffer_size/2)) # min radius
        hough_radii = np.arange(min_r, max_r, 10)
        hough_res = hough_circle(edges, hough_radii)
        # Select the most prominent circle
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        circy, circx = disk((cy[0], cx[0]), radii[0])
        # change the outlier image to fill in the circle
        outlier_bw_image[circy, circx] = True
        self.outlier_bw_image = outlier_bw_image
        # get the area of the ball bearing based on the known radius
        circle_area = np.pi*(known_radius**2)
        px_count_lst = []
        for bw_img in inv_bw_image_lst:
            px_count = np.unique(bw_img, return_counts=True)[1][1]
            px_count_lst.append(px_count)
        self.image_selected_df['pixel_count'] = px_count_lst
        circle_px_count = px_count_lst[outlier_idx]
        area_ar = (np.array(circle_px_count)/circle_px_count)*circle_area
        self.image_selected_df['area'] = area_ar
        # NB If outlier_bw_image is changed from default
        # make sure to update clean_bw_image_lst[selected iamge index] = self.outlier_bw_image
        # outside of the method