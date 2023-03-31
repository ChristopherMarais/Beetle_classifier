import os
import dill
import timm
import numpy as np
from fastai.tabular.all import *
from fastai.vision.all import *
from fastai.vision.utils import get_image_files
from Ambrosia import pre_process_image
from huggingface_hub import from_pretrained_fastai, push_to_hub_fastai
import gradio as gr

# this function only describes how much a singular value in al ist stands out.
# if all values in the lsit are high or low this is 1
# the smaller the proportiopn of number of disimilar vlaues are to other more similar values the lower this number
# the larger the gap between the dissimilar numbers and the simialr number the smaller this number
# only able to interpret probabilities or values between 0 and 1
# this function outputs an estimate an inverse of the classification confidence based on the probabilities of all the classes.
# the wedge threshold splits the data on a threshold with a magnitude of a positive int to force a ledge/peak in the data
def unkown_prob_calc(probs, wedge_threshold, wedge_magnitude=1, wedge='strict'):
    if wedge =='strict':
        increase_var = (1/(wedge_magnitude))
        decrease_var = (wedge_magnitude)
    if wedge =='dynamic': # this allows pointsthat are furhter from the threshold ot be moved less and points clsoer to be moved more
        increase_var = (1/(wedge_magnitude*((1-np.abs(probs-wedge_threshold)))))
        decrease_var = (wedge_magnitude*((1-np.abs(probs-wedge_threshold))))
    else:
        print("Error: use 'strict' (default) or 'dynamic' as options for the wedge parameter!")
    probs = np.where(probs>=wedge_threshold , probs**increase_var, probs)
    probs = np.where(probs<=wedge_threshold , probs**decrease_var, probs)
    diff_matrix = np.abs(probs[:, np.newaxis] - probs)
    diff_matrix_sum = np.sum(diff_matrix)
    probs_sum = np.sum(probs)
    class_val = (diff_matrix_sum/probs_sum)
    max_class_val = ((len(probs)-1)*2)
    kown_prob = class_val/max_class_val
    unknown_prob = 1-kown_prob
    return(unknown_prob)

# load model
# learn = load_learner("E:\\GIT_REPOS\\Beetle_classifier\\Models\\beetle_classifier.pkl", cpu=False)
learn = load_learner(r"beetle_classifier.pkl", cpu=False)
# get class names
labels = np.append(np.array(learn.dls.vocab), "Unknown")

def predict(img):
    # Segment image into smaller images
    pre_process = pre_process_image(manual_thresh_buffer=0.15, image = img) # use image_dir if directory of image used
    pre_process.segment(cluster_num=2, 
                        image_edge_buffer=50)
    # get predictions for all segments
    conf_dict_lst = []
    output_lst = []
    img_cnt = len(pre_process.col_image_lst)
    for i in range(0,img_cnt):
        prob_ar = np.array(learn.predict(pre_process.col_image_lst[i])[2])
        unkown_prob = unkown_prob_calc(probs=prob_ar, wedge_threshold=0.85, wedge_magnitude=5, wedge='dynamic')
        prob_ar = np.append(prob_ar, unkown_prob)
        prob_ar = np.around(prob_ar*100, decimals=1)
        
        conf_dict = {labels[i]: float(prob_ar[i]) for i in range(len(prob_ar))}
        conf_dict = dict(sorted(conf_dict.items(), key=lambda item: item[1], reverse=True))
        conf_dict_lst.append(str(conf_dict))
        result = list(zip(pre_process.col_image_lst, conf_dict_lst))
                
    return(result)

with gr.Blocks() as demo:
    with gr.Column(variant="panel"):
        with gr.Row(variant="compact"):
            inputs = gr.Image()
            btn = gr.Button("Classify").style(full_width=False)

        gallery = gr.Gallery(
            label="Show images", show_label=True, elem_id="gallery"
        ).style(grid=[8], height="auto")

    btn.click(predict, inputs, gallery)
    demo.launch(share=True)