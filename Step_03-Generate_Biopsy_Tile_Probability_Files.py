import os
import sys
import ast
import pathlib
import glob

import random

import numpy as np
import pandas as pd

# Read Image
import PIL

# TQDM
from tqdm import tqdm, trange

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata

import torchvision.transforms as transforms

import torchvision.datasets as datasets
import torch.utils.data as torchdata

# Date 
from datetime import datetime
from datetime import date

## Setup ##
sep = os.sep
current_dir = os.path.dirname(os.path.abspath(__file__))

### Loop Argument Processing ###
start_random_state = 1234
nb_iteration = 5

if( len(sys.argv)>1 ):
    print(f"#"*50)
    print(f"User Arguments")
    # start_random_state
    try:
        start_random_state = int(sys.argv[sys.argv.index('--start_random_state')+1])
        print(f'(User Defined) start_random_state: {start_random_state}')
    except: pass
    # nb_iteration
    try:
        nb_iteration =  int(sys.argv[sys.argv.index('--nb_iteration')+1])
        print(f'(User Defined) nb_iteration: {nb_iteration}')
    except: pass

for iteration in range(1, nb_iteration+1):
    
    random_state = start_random_state + iteration - 1

    ### Seed ###
    torch.cuda.empty_cache()
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    ## CUDNN ##
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    ### Use best Device (CUDA vs CPU) ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"#"*50)
    print(f"Device Used")
    print(device)
    ## Print Device Properties ##
    if device == torch.device('cuda'):
        print(torch.cuda.get_device_properties( device))


    ########################################
    ############ User Variables ############
    ########################################
    ### Command Line Argument Processing ###
    debugger=False
    augment=False
    augmentation_type = 'GAN' #can also be 'diffusion'
    model_type = 'vgg19' #can also be 'resnet50' or 'resnet152' or 'densenet161'
    dataset_type = 'jbi'

    if( len(sys.argv)>1 ):
        print(f"#"*50)
        print(f"User Arguments")
        print(sys.argv)
        # Use Debugger
        try:
            debugger= (sys.argv[sys.argv.index('--debug')+1] == 'True')
            print(f'(User Defined) Use Debugger: {debugger}')
        except: pass
        # Augment
        try:
            augment= (sys.argv[sys.argv.index('--augment')+1] == 'True')
            print(f'(User Defined) Augmentation: {augment}')
        except: pass
        # augmentation_type
        try:
            augmentation_type = (sys.argv[sys.argv.index('--augmentation_type')+1])
            print(f'(User Defined) augmentation_type: {augmentation_type}')
        except: pass
        # model_type
        try:
            model_type = (sys.argv[sys.argv.index('--model_type')+1])
            print(f'(User Defined) model_type: {model_type}')
        except: pass

    ### User Variables ###
    epoch_list = ['1'] * nb_iteration
    timeStamp = str(date.today())
    BATCH_SIZE = 64
    use_best = True

    if( len(sys.argv)>1 ):
        print(f"#"*50)
        print(f"## User Variables ##")
        # epoch_list
        try:
            epoch_list = ast.literal_eval(sys.argv[sys.argv.index('--epoch_list')+1])
            print(f'(User Defined) epoch_list: {epoch_list}')
        except: pass
        # timeStamp
        try:
            timeStamp = str(sys.argv[sys.argv.index('--timeStamp')+1])
            print(f'(User Defined) timeStamp: {timeStamp}')
        except: pass
        # BATCH_SIZE
        try:
            BATCH_SIZE = int(sys.argv[sys.argv.index('--BATCH_SIZE')+1])
            print(f'(User Defined) BATCH_SIZE: {BATCH_SIZE}')
        except: pass
        # use_best
        try:
            use_best = (sys.argv[sys.argv.index('--use_best')+1] == 'True')
            print(f'(User Defined) use_best: {use_best}')
        except: pass

    ## Use Debugger ##
    if(debugger):
        import pdb; pdb.set_trace() 


    ######################################
    ############# Path Setup #############
    ######################################
    ## Training WSIs ###
    train_wsis_path = f"{sep}home{sep}mainuser{sep}fast_datadrive{sep}HeartTransplantData{sep}2021_Data{sep}train_wsi{sep}"
    rejection_train_wsis = glob.glob(train_wsis_path + f"rejection{sep}[SC]*[0-9]")
    nonrejection_train_wsis = glob.glob(train_wsis_path+ f"nonrejection{sep}[SC]*[0-9]")
    train_wsis = rejection_train_wsis + nonrejection_train_wsis
    train_output_dir = f"{sep}home{sep}mainuser{sep}fast_datadrive{sep}HeartTransplantData{sep}2021_Data{sep}wsi_classifier_training_val_data{sep}wsi_train{sep}"

    ## Test WSIs ###
    test_wsis_path = f"{sep}home{sep}mainuser{sep}fast_datadrive{sep}HeartTransplantData{sep}2021_Data{sep}test_wsi{sep}"
    rejection_test_wsis = glob.glob(test_wsis_path + f"rejection{sep}[SC]*[0-9]")
    nonrejection_test_wsis = glob.glob(test_wsis_path+ f"nonrejection{sep}[SC]*[0-9]")
    test_wsis = rejection_test_wsis + nonrejection_test_wsis
    test_output_dir = f"{sep}home{sep}mainuser{sep}fast_datadrive{sep}HeartTransplantData{sep}2021_Data{sep}wsi_classifier_training_val_data{sep}wsi_test{sep}"
    ## Model Weight Path ##
    epoch = epoch_list[iteration-1]
    
    if(augment): 
        # Augmented
        model_name = f'tile-classifier_{model_type}_augmented_{augmentation_type}_{dataset_type}_{timeStamp}_iteration_{iteration}_seed_{random_state}'
        if use_best:
            model_weights_path = f'{current_dir}{sep}models{sep}{model_name}{sep}{model_name}_{epoch}_best.pt'
        else:
            model_weights_path = f'{current_dir}{sep}models{sep}{model_name}{sep}{model_name}_{epoch}.pt'
        # Output Name
        rejection_score_filename = f"rejection_scores_{model_type}_{timeStamp}_augmented_{augmentation_type}_{dataset_type}_iteration_{iteration}_seed_{random_state}.npy"

    else: 
        # Original
        model_name = f'tile-classifier_{model_type}_{dataset_type}_{timeStamp}_iteration_{iteration}_seed_{random_state}'
        if use_best:
            model_weights_path = f'{current_dir}{sep}models{sep}{model_name}{sep}{model_name}_{epoch}_best.pt'
        else:
            model_weights_path = f'{current_dir}{sep}models{sep}{model_name}{sep}{model_name}_{epoch}.pt'
        # Output Name
        rejection_score_filename = f"rejection_scores_{model_type}_{timeStamp}_{dataset_type}_iteration_{iteration}_seed_{random_state}.npy"


    #####################################
    ########## Print Variables ##########
    #####################################
    ## Parameters Dictionary ##
    param_dict = {}
    param_dict['debugger'] = str(debugger)
    param_dict['augment'] = str(augment)
    param_dict['augmentation_type'] = str(augmentation_type)
    param_dict['model_type'] = str(model_type)

    param_dict['epoch'] = str(epoch)
    param_dict['timeStamp'] = str(timeStamp)
    param_dict['BATCH_SIZE'] = str(BATCH_SIZE)
    param_dict['use_best'] = str(use_best)

    param_dict['model_name'] = str(model_name)
    param_dict['model_weights_path'] = str(model_weights_path)

    param_dict['train_output_dir'] = train_output_dir
    param_dict['test_output_dir'] = test_output_dir
    param_dict['rejection_score_filename'] = rejection_score_filename



    ### Run Model on Iterator ###
    def predict(data_iterator, device):
        labels = torch.Tensor([]).to(device)
        predicted = torch.Tensor([]).to(device)
        n_batches = len(data_iterator)

        with torch.no_grad():
            for i, (x,y) in enumerate(data_iterator):
                print( '\tBatch: %s / %s'% (i,n_batches) , end='\r' )
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                labels = torch.cat((labels, y))
                predicted = torch.cat((predicted, y_pred))
            torch.cuda.empty_cache() 

        softmax = torch.nn.Softmax(dim=1);
        predicted_softmax = softmax(predicted);
        rejection_softmax = predicted_softmax[:, 1].cpu().numpy();

        return labels, rejection_softmax


    ########################################
    ####### Model and Datasets Setup #######
    ########################################
    ### Model: VGG19 with ImageNet weights ###
    if model_type == 'vgg19':
        model = models.vgg19(weights = 'DEFAULT')
        # DL Features from model
        IN_FEATURES = model.classifier[-1].in_features 
        # New classification layer
        final_fc = nn.Linear(IN_FEATURES, 2)
        # Append new classification layer
        model.classifier[-1] = final_fc
    elif model_type == 'resnet50':
        model = models.resnet50(weights = 'DEFAULT')
        # DL Features from model
        IN_FEATURES = model.fc.in_features 
        # New classification layer
        final_fc = nn.Linear(IN_FEATURES, 2)
        # Append new classification layer
        model.fc = final_fc
    elif model_type == 'resnet152':
        model = models.resnet152(weights = 'DEFAULT')
        # DL Features from model
        IN_FEATURES = model.fc.in_features 
        # New classification layer
        final_fc = nn.Linear(IN_FEATURES, 2)
        # Append new classification layer
        model.fc = final_fc
    elif model_type == 'densenet161':
        model = models.densenet161(weights = 'DEFAULT')
        # DL Features from model
        IN_FEATURES = model.classifier.in_features 
        # New classification layer
        final_fc = nn.Linear(IN_FEATURES, 2)
        # Append new classification layer
        model.classifier = final_fc
    else:
        raise Exception(f"Can not use the following model_type: {model_type}")

    ### Model: Load Weights ###
    print(f"#"*50)
    print("Total GPUs: %s" %torch.cuda.device_count() )
    ## Model saved as parallel ##
    if( list( torch.load(model_weights_path).keys() )[0].find('module.', 0, 7) != -1 ):
        ## Parallel: adds 'model.' in front of model layers ##
        model = nn.DataParallel(model)
    ## Load Trained Weights (Note: needs to be after DataParallel) ##
    model.load_state_dict( torch.load(model_weights_path), strict=True)
    ## Freeze Model ##
    for p in model.parameters():
        p.requires_grad = False
    ## Model: Send to device ##
    model.to(device);
    ## Do not Train ##
    model.eval();


    ### Transformations ###
    pretrained_size = 256
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
                                transforms.Resize(pretrained_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = pretrained_means, 
                                                     std = pretrained_stds)
                            ])

    param_dict['len(train_wsis)'] = len(train_wsis)
    param_dict['len(test_wsis)'] = len(test_wsis)

    print(f"#"*50)
    print(f"## Training Parameters ##")
    for key_ in param_dict.keys(): # create a tag in tensorboardx for every tag used.
        text = key_ + ': ' + str(param_dict[key_])
        print(text)

    pathlib.Path(f'{current_dir}{sep}param_dict{sep}').mkdir(parents=True, exist_ok=True)
    df_param_dict = pd.DataFrame.from_dict(param_dict, orient='index')
    df_param_dict.to_csv(f'{current_dir}{sep}param_dict{sep}{model_name}_param_biopsy_gen_iteration_{iteration}_seed_{random_state}.csv')


    ### Predict Training WSI tiles ###
    for wsi_idx, wsi_path in enumerate(train_wsis):
        ### WSI Name ###
        wsi_name = wsi_path.split("/")[-1]
        print(wsi_name)

        ### Progress ###
        print('\t%s / %s'% (wsi_idx,len(train_wsis) ) ) 

        ### Output File ###
        output_file = '%s/%s/%s' % (train_output_dir, wsi_name, rejection_score_filename)
        pathlib.Path(f"{train_output_dir}{wsi_name}{sep}").mkdir(parents=True, exist_ok=True)
        print('\tOutput: %s'% output_file)
        ### Skip if already Exists ###
        if( os.path.exists(output_file) ):
            print('\tSKIPPED')
            continue

        ### Train Data ###
        train_dataset = datasets.ImageFolder(wsi_path, data_transforms)
        train_iterator = torchdata.DataLoader(train_dataset, batch_size = BATCH_SIZE,
                                             num_workers=0)

        ### Predictions ###
        labels, rejection_softmax = predict(train_iterator, device);

        ### Save ###
        with open(output_file, 'wb') as f:
            np.save(f, rejection_softmax)


    ### Predict Testing WSI tiles ###
    for wsi_idx, wsi_path in enumerate(test_wsis):
        ### WSI Name ###
        wsi_name = wsi_path.split("/")[-1]
        print(wsi_name)

        ### Progress ###
        print('\t%s / %s'% (wsi_idx,len(test_wsis) ) ) 

        ### Output File ###
        output_file = '%s/%s/%s' % (test_output_dir, wsi_name, rejection_score_filename)
        pathlib.Path(f"{test_output_dir}{wsi_name}{sep}").mkdir(parents=True, exist_ok=True)
        print('Output: %s'% output_file)
        ### Skip if already Exists ###
        if( os.path.exists(output_file) ):
            print('\tSKIPPED')
            continue

        ### Test Data ###
        test_dataset = datasets.ImageFolder(wsi_path, data_transforms)
        test_iterator = torchdata.DataLoader(test_dataset, batch_size = BATCH_SIZE,
                                            num_workers=0)

        ### Predictions ###
        labels, rejection_softmax = predict(test_iterator, device);

        ### Save ###
        with open(output_file, 'wb') as f:
            np.save(f, rejection_softmax)
