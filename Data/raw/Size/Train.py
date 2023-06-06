import timm
import torch
import wandb
import fastai
import dill
import re
import random
import PIL
import numpy as np
from fastai.vision.augment import cutout_gaussian
from fastai.callback.wandb import *
from fastai.vision.all import *
from fastai.vision.core import *
from fastai.text.core import RegexLabeller
from fastai.vision.utils import get_image_files
from fastai.data.block import DataBlock
from fastai.data.core import *
from fastai.tabular.all import *
from fastcore.foundation import L
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from huggingface_hub import notebook_login, push_to_hub_fastai, from_pretrained_fastai
from torchvision.transforms import GaussianBlur
# os.environ['WANDB_WATCH'] = 'false'
os.environ['WANDB_NOTEBOOK_NAME'] = 'Train.ipynb'


config = SimpleNamespace(
    batch_size=64,  #16, #256,
    epochs=5,
    lr=3e-3,
    img_size=224, # 224, 256 for small model on huggingface
    seed=42,
    pretrained=True,
    top_k_losses=5,
    model_name="maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k",# "maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k",# maxvit_nano_rw_256.sw_in1k for HF spaces
    wandb_project="Beetle_classifier", 
    wandb_group="ambrosia_symbiosis",
    job_type="training_cv_5fold_ooc"
    )

# Define a custom transform for Gaussian blur
def gaussian_blur(x, p=0.5, kernel_size_min=3, kernel_size_max=20, sigma_min=0.1, sigma_max=3):
    if x.ndim == 4:
        for i in range(x.shape[0]):
            if random.random() < p:
                kernel_size = random.randrange(kernel_size_min, kernel_size_max + 1, 2)
                sigma = random.uniform(sigma_min, sigma_max)
                x[i] = GaussianBlur(kernel_size=kernel_size, sigma=sigma)(x[i])
    return x

# def custom_parent_label(fname, known_categories):
#     category = parent_label(fname)
#     if category not in known_categories:
#         return "Unknown"
#     else:
#         return category

# def get_categories(fnames):
#     categories = []
#     for fname in fnames:
#         category = parent_label(fname)
#         if category not in categories:
#             categories.append(category)
#     return categories

def get_image_files_exclude(path, folders=('train','valid'), exclude_folder=None):
    files = get_image_files(path=path, recurse=True, folders=folders)
    if exclude_folder:
            files = L([f for f in files if (exclude_folder not in str(f))])
    return files
    

def get_images(dataset_path, batch_size, img_size, seed, subfolders=('train','valid'), exclude_folder=None):
    "The beetles dataset"
    files = get_image_files_exclude(path=dataset_path, folders=subfolders, exclude_folder=exclude_folder)
    # files = get_image_files(path=dataset_path, recurse=True, folders=subfolders)
    transforms = aug_transforms(    # transformatiosn that are only applied ot training and not inference
                           batch=False,
                           pad_mode='zeros',
                           size=img_size,
                           p_affine=0.8,
                           p_lighting=0.8,
                           max_rotate=360.0,
                           mult=1.0, 
                           do_flip=True, 
                           flip_vert=False,
                           min_zoom=1.0,
                           max_zoom=1.1, 
                           max_lighting=0.75,
                           max_warp=0.2, 
                           mode='bilinear', 
                           align_corners=True,
                           min_scale=1.0,
                           xtra_tfms=[RandomErasing(p=0.8, max_count=5, sh=0.25)]) # this adds random erasing to entire batches
    transforms.append(partial(gaussian_blur, p=0.8))
    transforms.append(Normalize.from_stats(*imagenet_stats))
    # categories = get_categories(files)
    dblock = DataBlock(blocks = (ImageBlock, CategoryBlock),#(vocab=categories+["Unknown"])),
                       get_items = partial(get_image_files_exclude, 
                                           folders=subfolders, 
                                           exclude_folder=exclude_folder),
                       # get_items = get_image_files,
                       splitter = GrandparentSplitter(train_name=subfolders[0], valid_name=subfolders[1]),
                       get_y = parent_label, # partial(custom_parent_label, known_categories=categories),
                       item_tfms = Resize(img_size, ResizeMethod.Pad, pad_mode='zeros'), # resize trasnformation is applied during inference too                                    
                       batch_tfms = transforms)
    dls = dblock.dataloaders(dataset_path, bs = batch_size, num_workers=4)
    return dls

def train(config, dataset_path, subfolders=('train','valid'), exclude_folder=None):
    "Train the model using the supplied config"
    dls = get_images(dataset_path=dataset_path, 
                     batch_size=config.batch_size, 
                     img_size=config.img_size, 
                     seed=config.seed, 
                     subfolders=subfolders, 
                     exclude_folder=exclude_folder)
    labels = np.array([re.split(r'/|\\', str(x))[-2] for x in dls.items])
    classes = np.unique(labels)# for label in labels if label != "Unknown"])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights = {c: w for c, w in zip(classes, weights)}
    weights = tensor([class_weights[c] for c in dls.vocab]).to(dls.device)
    # wandb.init(project=config.wandb_project, group=config.wandb_group, job_type=config.job_type, config=config) # it is a good idea to keep these functions out of the training function due to some exporting issues
    cbs = [MixedPrecision(), ShowGraphCallback(), SaveModelCallback(), WandbCallback(log='gradients')] # (all, parameters, gradients or None) parameters and all does nto work currently wandb needs to be updated
    learn = vision_learner(dls, 
                           config.model_name, 
                           loss_func=LabelSmoothingCrossEntropy(weight=weights), # this fucntion is used for class imbalance it is a regularization technique # LabelSmoothingCrossEntropyFlat is used for multi dimensional data
                           metrics=[error_rate, 
                                    accuracy, 
                                    top_k_accuracy], 
                           cbs=cbs, 
                           pretrained=config.pretrained)
    learn.fine_tune(config.epochs, base_lr=config.lr)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    interp.plot_top_losses(config.top_k_losses, nrows=config.top_k_losses)
    # wandb.finish() # it is a good idea to keep these functions out of the training function due to some exporting issues
    return learn

# this function only describes how much a singular value in al ist stands out.
# if all values in the lsit are high or low this is 1
# the smaller the proportiopn of number of disimilar vlaues are to other more similar values the lower this number
# the larger the gap between the dissimilar numbers and the simialr number the smaller this number
# only able to interpret probabilities or values between 0 and 1
# this function outputs an estimate an inverse of the classification confidence based on the probabilities of all the classes.
# the wedge threshold splits the data on a threshold with a magnitude of a positive int to force a ledge/peak in the data
def unknown_prob_calc(probs, wedge_threshold, wedge_magnitude=1, wedge='strict'):
    if wedge =='strict':
        increase_var = (1/(wedge_magnitude))
        decrease_var = (wedge_magnitude)
    if wedge =='dynamic': # this allows pointsthat are furhter from the threshold ot be moved less and points clsoer to be moved more
        increase_var = (1/(wedge_magnitude*((1-np.abs(probs-wedge_threshold)))))
        decrease_var = (wedge_magnitude*((1-np.abs(probs-wedge_threshold))))
    # else:
    #     print("Error: use 'strict' (default) or 'dynamic' as options for the wedge parameter!")
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

# species_lst = [
#     "Coccotypes_dactyliperda",
#     "Hylesinus_varius",
#     "Monarthrum_fasciatum",
#     "Phloeosinus_dentatus",
#     "Pityophthorus_juglandis",
#     "Platypus_cylindrus",
#     "Pycnarthrum_hispidium",
#     "Scolotodes_schwarzi",
#     "Xyleborinus_saxesenii",
#     "Xyleborus_affinis",
#     "Xylosandrus_compactus",
#     "Xylosandrus_crassiusculus",
#     None
# ]

# for species in species_lst:
#     # CV training
#     valid_targets = []
#     valid_preds = []
#     valid_class = []
#     test_targets = []
#     test_preds = []
#     test_class = []
#     for i in range(1, 6):
#         # Train Model
#         print("Training: ", str(i))
#         dataset_path = r"/blue/hulcr/gmarais/Beetle_data/kfold_images/train_data"
#         # dataset_path = r"F:\Beetle_data\kfold_images\train_data"
#         wandb.init(project=config.wandb_project, group=config.wandb_group, job_type=config.job_type, config=config)
#         learn = train(config=config, 
#                       dataset_path=dataset_path, 
#                       subfolders=('train_'+str(i),'valid_'+str(i)),
#                      exclude_folder=species)
#         wandb.finish()

#         # get predicstions and labels
#         print("Testing: ", str(i))
#         learn.remove_cb(WandbCallback)
#         # Get validation labels and predictions
#         files = get_image_files_exclude(path=dataset_path, folders=('valid_'+str(i)), exclude_folder=species) # get files from directory
#         test_dl = learn.dls.test_dl(files, with_labels=True) # load data as a dataloader
#         preds, targets = learn.get_preds(dl=test_dl)
#         valid_targets.append(targets.cpu().numpy())
#         max_probs, max_classes = preds.max(dim=1)
#         valid_class.append(max_classes.cpu().numpy())
#         valid_preds.append(max_probs.cpu().numpy())
#         pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
#         pred_df.to_csv(str(species)+"_Validation_prediction_probabilities_fold-"+str(i)+".csv")
#         # unknown validation
#         files = get_image_files_exclude(path=dataset_path+'/valid_'+str(i), folders=(str(species)), exclude_folder=None) # get files from directory
#         test_dl = learn.dls.test_dl(files, with_labels=False) # load data as a dataloader
#         preds, targets = learn.get_preds(dl=test_dl)
#         pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
#         pred_df.to_csv(str(species)+"_Unknown_Validation_prediction_probabilities_fold-"+str(i)+".csv")

#         # testing data
#         test_dataset_path = r"/blue/hulcr/gmarais/Beetle_data/kfold_images"
#         files = get_image_files_exclude(path=test_dataset_path, folders=('test_data'), exclude_folder=species) # get files from directory
#         test_dl = learn.dls.test_dl(files, with_labels=True) # load data as a dataloader
#         preds, targets = learn.get_preds(dl=test_dl)
#         test_targets.append(targets.cpu().numpy())
#         max_probs, max_classes = preds.max(dim=1)
#         test_class.append(max_classes.cpu().numpy())
#         test_preds.append(max_probs.cpu().numpy())
#         pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
#         pred_df.to_csv(str(species)+"_Testing_prediction_probabilities_fold-"+str(i)+".csv")
#         # unknown testing 
#         files = get_image_files_exclude(path=test_dataset_path+'/test_data', folders=(str(species)), exclude_folder=None) # get files from directory
#         test_dl = learn.dls.test_dl(files, with_labels=False) # load data as a dataloader
#         preds, targets = learn.get_preds(dl=test_dl)
#         pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
#         pred_df.to_csv(str(species)+"_Unknown_Testing_prediction_probabilities_fold-"+str(i)+".csv")

#     # save results to a csv
#     class_df = pd.DataFrame(valid_class).T
#     class_df.columns = [str(col)+'_class' for col in class_df.columns] 
#     target_df = pd.DataFrame(valid_targets).T
#     target_df.columns = [str(col)+'_target' for col in target_df.columns] 
#     result_df = pd.concat([class_df, target_df], axis=1)
#     result_df.to_csv(str(species)+"_Validation_5-fold_CV_classification_results.csv")

#     class_df = pd.DataFrame(test_class).T
#     class_df.columns = [str(col)+'_class' for col in class_df.columns] 
#     target_df = pd.DataFrame(test_targets).T
#     target_df.columns = [str(col)+'_target' for col in target_df.columns] 
#     result_df = pd.concat([class_df, target_df], axis=1)
#     result_df.to_csv(str(species)+"_Testing_5-fold_CV_classification_results.csv")


loop_lst = [
{'species': 'Coccotypes_dactyliperda',
'test_file_loop_name': 'Coccotypes_dactyliperda_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Coccotypes_dactyliperda_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Coccotypes_dactyliperda',
'test_file_loop_name': 'Coccotypes_dactyliperda_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Coccotypes_dactyliperda_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Coccotypes_dactyliperda',
'test_file_loop_name': 'Coccotypes_dactyliperda_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Coccotypes_dactyliperda_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Coccotypes_dactyliperda',
'test_file_loop_name': 'Coccotypes_dactyliperda_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Coccotypes_dactyliperda_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Coccotypes_dactyliperda',
'test_file_loop_name': 'Coccotypes_dactyliperda_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Coccotypes_dactyliperda_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Hylesinus_varius',
'test_file_loop_name': 'Hylesinus_varius_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Hylesinus_varius_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Hylesinus_varius',
'test_file_loop_name': 'Hylesinus_varius_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Hylesinus_varius_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Hylesinus_varius',
'test_file_loop_name': 'Hylesinus_varius_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Hylesinus_varius_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Hylesinus_varius',
'test_file_loop_name': 'Hylesinus_varius_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Hylesinus_varius_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Hylesinus_varius',
'test_file_loop_name': 'Hylesinus_varius_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Hylesinus_varius_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Monarthrum_fasciatum',
'test_file_loop_name': 'Monarthrum_fasciatum_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Monarthrum_fasciatum_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Monarthrum_fasciatum',
'test_file_loop_name': 'Monarthrum_fasciatum_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Monarthrum_fasciatum_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Monarthrum_fasciatum',
'test_file_loop_name': 'Monarthrum_fasciatum_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Monarthrum_fasciatum_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Monarthrum_fasciatum',
'test_file_loop_name': 'Monarthrum_fasciatum_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Monarthrum_fasciatum_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Monarthrum_fasciatum',
'test_file_loop_name': 'Monarthrum_fasciatum_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Monarthrum_fasciatum_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Phloeosinus_dentatus',
'test_file_loop_name': 'Phloeosinus_dentatus_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Phloeosinus_dentatus_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Phloeosinus_dentatus',
'test_file_loop_name': 'Phloeosinus_dentatus_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Phloeosinus_dentatus_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Phloeosinus_dentatus',
'test_file_loop_name': 'Phloeosinus_dentatus_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Phloeosinus_dentatus_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Phloeosinus_dentatus',
'test_file_loop_name': 'Phloeosinus_dentatus_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Phloeosinus_dentatus_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Phloeosinus_dentatus',
'test_file_loop_name': 'Phloeosinus_dentatus_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Phloeosinus_dentatus_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Pityophthorus_juglandis',
'test_file_loop_name': 'Pityophthorus_juglandis_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Pityophthorus_juglandis_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Pityophthorus_juglandis',
'test_file_loop_name': 'Pityophthorus_juglandis_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Pityophthorus_juglandis_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Pityophthorus_juglandis',
'test_file_loop_name': 'Pityophthorus_juglandis_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Pityophthorus_juglandis_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Pityophthorus_juglandis',
'test_file_loop_name': 'Pityophthorus_juglandis_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Pityophthorus_juglandis_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Pityophthorus_juglandis',
'test_file_loop_name': 'Pityophthorus_juglandis_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Pityophthorus_juglandis_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Platypus_cylindrus',
'test_file_loop_name': 'Platypus_cylindrus_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Platypus_cylindrus_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Platypus_cylindrus',
'test_file_loop_name': 'Platypus_cylindrus_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Platypus_cylindrus_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Platypus_cylindrus',
'test_file_loop_name': 'Platypus_cylindrus_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Platypus_cylindrus_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Platypus_cylindrus',
'test_file_loop_name': 'Platypus_cylindrus_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Platypus_cylindrus_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Platypus_cylindrus',
'test_file_loop_name': 'Platypus_cylindrus_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Platypus_cylindrus_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Pycnarthrum_hispidium',
'test_file_loop_name': 'Pycnarthrum_hispidium_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Pycnarthrum_hispidium_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Pycnarthrum_hispidium',
'test_file_loop_name': 'Pycnarthrum_hispidium_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Pycnarthrum_hispidium_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Pycnarthrum_hispidium',
'test_file_loop_name': 'Pycnarthrum_hispidium_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Pycnarthrum_hispidium_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Pycnarthrum_hispidium',
'test_file_loop_name': 'Pycnarthrum_hispidium_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Pycnarthrum_hispidium_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Pycnarthrum_hispidium',
'test_file_loop_name': 'Pycnarthrum_hispidium_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Pycnarthrum_hispidium_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Scolotodes_schwarzi',
'test_file_loop_name': 'Scolotodes_schwarzi_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Scolotodes_schwarzi_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Scolotodes_schwarzi',
'test_file_loop_name': 'Scolotodes_schwarzi_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Scolotodes_schwarzi_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Scolotodes_schwarzi',
'test_file_loop_name': 'Scolotodes_schwarzi_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Scolotodes_schwarzi_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Scolotodes_schwarzi',
'test_file_loop_name': 'Scolotodes_schwarzi_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Scolotodes_schwarzi_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Scolotodes_schwarzi',
'test_file_loop_name': 'Scolotodes_schwarzi_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Scolotodes_schwarzi_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Xyleborinus_saxesenii',
'test_file_loop_name': 'Xyleborinus_saxesenii_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Xyleborinus_saxesenii_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Xyleborinus_saxesenii',
'test_file_loop_name': 'Xyleborinus_saxesenii_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Xyleborinus_saxesenii_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Xyleborinus_saxesenii',
'test_file_loop_name': 'Xyleborinus_saxesenii_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Xyleborinus_saxesenii_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Xyleborinus_saxesenii',
'test_file_loop_name': 'Xyleborinus_saxesenii_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Xyleborinus_saxesenii_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Xyleborinus_saxesenii',
'test_file_loop_name': 'Xyleborinus_saxesenii_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Xyleborinus_saxesenii_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Xyleborus_affinis',
'test_file_loop_name': 'Xyleborus_affinis_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Xyleborus_affinis_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Xyleborus_affinis',
'test_file_loop_name': 'Xyleborus_affinis_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Xyleborus_affinis_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Xyleborus_affinis',
'test_file_loop_name': 'Xyleborus_affinis_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Xyleborus_affinis_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Xyleborus_affinis',
'test_file_loop_name': 'Xyleborus_affinis_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Xyleborus_affinis_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Xyleborus_affinis',
'test_file_loop_name': 'Xyleborus_affinis_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Xyleborus_affinis_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Xylosandrus_compactus',
'test_file_loop_name': 'Xylosandrus_compactus_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Xylosandrus_compactus_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Xylosandrus_compactus',
'test_file_loop_name': 'Xylosandrus_compactus_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Xylosandrus_compactus_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Xylosandrus_compactus',
'test_file_loop_name': 'Xylosandrus_compactus_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Xylosandrus_compactus_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Xylosandrus_compactus',
'test_file_loop_name': 'Xylosandrus_compactus_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Xylosandrus_compactus_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Xylosandrus_compactus',
'test_file_loop_name': 'Xylosandrus_compactus_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Xylosandrus_compactus_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': 'Xylosandrus_crassiusculus',
'test_file_loop_name': 'Xylosandrus_crassiusculus_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'Xylosandrus_crassiusculus_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': 'Xylosandrus_crassiusculus',
'test_file_loop_name': 'Xylosandrus_crassiusculus_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'Xylosandrus_crassiusculus_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': 'Xylosandrus_crassiusculus',
'test_file_loop_name': 'Xylosandrus_crassiusculus_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'Xylosandrus_crassiusculus_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': 'Xylosandrus_crassiusculus',
'test_file_loop_name': 'Xylosandrus_crassiusculus_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'Xylosandrus_crassiusculus_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': 'Xylosandrus_crassiusculus',
'test_file_loop_name': 'Xylosandrus_crassiusculus_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'Xylosandrus_crassiusculus_Unknown_Testing_prediction_probabilities_fold-5.csv'},
{'species': None,
'test_file_loop_name': 'None_Testing_5-fold_CV_classification_results.csv',
'fold': 1,
'eval_file_loop_name': 'None_Unknown_Testing_prediction_probabilities_fold-1.csv'},
{'species': None,
'test_file_loop_name': 'None_Testing_5-fold_CV_classification_results.csv',
'fold': 2,
'eval_file_loop_name': 'None_Unknown_Testing_prediction_probabilities_fold-2.csv'},
{'species': None,
'test_file_loop_name': 'None_Testing_5-fold_CV_classification_results.csv',
'fold': 3,
'eval_file_loop_name': 'None_Unknown_Testing_prediction_probabilities_fold-3.csv'},
{'species': None,
'test_file_loop_name': 'None_Testing_5-fold_CV_classification_results.csv',
'fold': 4,
'eval_file_loop_name': 'None_Unknown_Testing_prediction_probabilities_fold-4.csv'},
{'species': None,
'test_file_loop_name': 'None_Testing_5-fold_CV_classification_results.csv',
'fold': 5,
'eval_file_loop_name': 'None_Unknown_Testing_prediction_probabilities_fold-5.csv'}]

try:
    for loop in range(len(loop_lst)):
        loop_dict = loop_lst[loop]
        if not os.path.isfile(loop_dict['eval_file_loop_name']):
            species = loop_dict['species']
            i = loop_dict['fold']

            # Train Model
            print("Training: ", str(i))
            dataset_path = r"/blue/hulcr/gmarais/Beetle_data/kfold_images/train_data"
            # dataset_path = r"F:\Beetle_data\kfold_images\train_data"
            wandb.init(project=config.wandb_project, group=config.wandb_group, job_type=config.job_type, config=config)
            learn = train(config=config, 
                          dataset_path=dataset_path, 
                          subfolders=('train_'+str(i),'valid_'+str(i)),
                         exclude_folder=species)
            wandb.finish()

            # get predicstions and labels
            print("Testing: ", str(i))
            learn.remove_cb(WandbCallback)
            # Get validation labels and predictions
            files = get_image_files_exclude(path=dataset_path, folders=('valid_'+str(i)), exclude_folder=species) 
            test_dl = learn.dls.test_dl(files, with_labels=True) # load data as a dataloader
            preds, targets = learn.get_preds(dl=test_dl)
            pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
            pred_df.to_csv(str(species)+"_Validation_prediction_probabilities_fold-"+str(i)+".csv")
            # unknown validation
            files = get_image_files_exclude(path=dataset_path+'/valid_'+str(i), folders=(str(species)), exclude_folder=None) 
            test_dl = learn.dls.test_dl(files, with_labels=False) # load data as a dataloader
            preds, targets = learn.get_preds(dl=test_dl)
            pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
            pred_df.to_csv(str(species)+"_Unknown_Validation_prediction_probabilities_fold-"+str(i)+".csv")

            # testing data
            test_dataset_path = r"/blue/hulcr/gmarais/Beetle_data/kfold_images"
            files = get_image_files_exclude(path=test_dataset_path, folders=('test_data'), exclude_folder=species) 
            test_dl = learn.dls.test_dl(files, with_labels=True) # load data as a dataloader
            preds, targets = learn.get_preds(dl=test_dl)
            pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
            pred_df.to_csv(str(species)+"_Testing_prediction_probabilities_fold-"+str(i)+".csv")
            # unknown testing 
            files = get_image_files_exclude(path=test_dataset_path+'/test_data', folders=(str(species)), exclude_folder=None) 
            test_dl = learn.dls.test_dl(files, with_labels=False) # load data as a dataloader
            preds, targets = learn.get_preds(dl=test_dl)
            pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
            pred_df.to_csv(str(species)+"_Unknown_Testing_prediction_probabilities_fold-"+str(i)+".csv")
except:
    for loop in range(len(loop_lst)):
        loop_dict = loop_lst[loop]
        if not os.path.isfile(loop_dict['eval_file_loop_name']):
            species = loop_dict['species']
            i = loop_dict['fold']

            # Train Model
            print("Training: ", str(i))
            dataset_path = r"/blue/hulcr/gmarais/Beetle_data/kfold_images/train_data"
            # dataset_path = r"F:\Beetle_data\kfold_images\train_data"
            wandb.init(project=config.wandb_project, group=config.wandb_group, job_type=config.job_type, config=config)
            learn = train(config=config, 
                          dataset_path=dataset_path, 
                          subfolders=('train_'+str(i),'valid_'+str(i)),
                         exclude_folder=species)
            wandb.finish()

            # get predicstions and labels
            print("Testing: ", str(i))
            learn.remove_cb(WandbCallback)
            # Get validation labels and predictions
            files = get_image_files_exclude(path=dataset_path, folders=('valid_'+str(i)), exclude_folder=species) 
            test_dl = learn.dls.test_dl(files, with_labels=True) # load data as a dataloader
            preds, targets = learn.get_preds(dl=test_dl)
            pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
            pred_df.to_csv(str(species)+"_Validation_prediction_probabilities_fold-"+str(i)+".csv")
            # unknown validation
            files = get_image_files_exclude(path=dataset_path+'/valid_'+str(i), folders=(str(species)), exclude_folder=None) 
            test_dl = learn.dls.test_dl(files, with_labels=False) # load data as a dataloader
            preds, targets = learn.get_preds(dl=test_dl)
            pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
            pred_df.to_csv(str(species)+"_Unknown_Validation_prediction_probabilities_fold-"+str(i)+".csv")

            # testing data
            test_dataset_path = r"/blue/hulcr/gmarais/Beetle_data/kfold_images"
            files = get_image_files_exclude(path=test_dataset_path, folders=('test_data'), exclude_folder=species) 
            test_dl = learn.dls.test_dl(files, with_labels=True) # load data as a dataloader
            preds, targets = learn.get_preds(dl=test_dl)
            pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
            pred_df.to_csv(str(species)+"_Testing_prediction_probabilities_fold-"+str(i)+".csv")
            # unknown testing 
            files = get_image_files_exclude(path=test_dataset_path+'/test_data', folders=(str(species)), exclude_folder=None) 
            test_dl = learn.dls.test_dl(files, with_labels=False) # load data as a dataloader
            preds, targets = learn.get_preds(dl=test_dl)
            pred_df = pd.DataFrame(preds.cpu().numpy(), columns=learn.dls.vocab)
            pred_df.to_csv(str(species)+"_Unknown_Testing_prediction_probabilities_fold-"+str(i)+".csv")