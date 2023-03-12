import os
import sys
import re
import pandas as pd
from Ambrosia import pre_process_image
from skimage import io

def segmentation(folder_dir, output_dir):
    try:
        metadata_df = pd.read_csv(output_dir+"segmented_images_metadata.csv", index_col="Unnamed: 0",low_memory=False)
    except:
        metadata_df = pd.DataFrame(columns=['centroid-0', 
                                            'centroid-1', 
                                            'bbox-0', 
                                            'bbox-1', 
                                            'bbox-2', 
                                            'bbox-3',
                                            'orientation', 
                                            'axis_major_length',
                                            'axis_minor_length', 
                                            'area',
                                            'area_filled',
                                            'real_area',
                                            'kmeans_label', 
                                            'circle_class', 
                                            'pixel_count',
                                            'composite_image_path', 
                                            'species', 
                                            'vial', 
                                            'subset',
                                            'composite_image_number', 
                                            'segmented_image_name'])

    for root, dirs, files in os.walk(folder_dir):
        for filename in files:
            if (filename.endswith(".JPG") or filename.endswith(".JPEG") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
                file = os.path.join(root, filename)
                if "Attempt" in file:
                    file.replace(file, "")
                else:
                    label = re.sub("Beetle_classification_deep_data|Vial_|Subset_|IMG_|.JPG", "", file)
                    label = label.split("\\")
                # test if iamge has already been processed    
                if file not in list(set(metadata_df['composite_image_path'])):
                    try:
                        pre_process = pre_process_image(image_dir = file, manual_thresh_buffer=0.15)
                        pre_process.segment(cluster_num=2, 
                                            image_edge_buffer=50)
                        pre_process.detect_outlier()
                        pre_process.estimate_size(outlier_idx=pre_process.outlier_idx, 
                                                  known_radius=1, 
                                                  canny_sigma=5)

                        # add info to table
                        seg_img_df = pre_process.image_selected_df
                        seg_img_df['composite_image_path'] = file
                        seg_img_df['species'] = label[-4]
                        seg_img_df['vial'] = label[-3]
                        seg_img_df['subset'] = label[-2]
                        seg_img_df['composite_image_number'] = label[-1]
                        seg_img_df['segmented_image_name'] = seg_img_df['species']+"_"+seg_img_df['vial']+"_"+seg_img_df['subset']+"_"+seg_img_df['composite_image_number']+"_"+seg_img_df.index.astype(str)
                        metadata_df = pd.concat([metadata_df, seg_img_df])
                    except:
                        print(file) # this will print the file htat results in an error

                    # save images
                    for j in range(len(pre_process.col_image_lst)):
                        image = pre_process.col_image_lst[j]
                        image_name = seg_img_df.loc[j]['segmented_image_name']
                        io.imsave(output_dir+image_name+".JPG", image)
                else:
                    print("Already processed image: "+ file)

                metadata_df = metadata_df.reset_index(drop=True)
                metadata_df.to_csv(output_dir+"segmented_images_metadata.csv")

                
                
# apply function
# default directories
f_dir = "Z:\lab records\Christopher_Marais\Beetle_classification_deep_data"
o_dir = "Z:\lab records\Christopher_Marais\Beetle_classification_deep_data_segmented\\"


try:
    segmentation(folder_dir=sys.argv[1], output_dir=sys.argv[2])
except:
    segmentation(folder_dir=f_dir, output_dir=o_dir)