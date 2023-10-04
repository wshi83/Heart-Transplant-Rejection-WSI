import numpy as np
import pandas as pd

import os
import sys
import ast
import pathlib
import random 
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as torchdata

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

# Date 
from datetime import datetime
from datetime import date

### Setup ###
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
        print(torch.cuda.get_device_properties( device ))

    ### Command Line Argument Processing ###
    debugger=False
    augment=False
    augmentation_type = 'origin' #can also be 'diffusion'
    model_type = 'vgg19' #can also be 'resnet50' or 'resnet152' or 'densenet161'

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
    BATCH_SIZE = 24
    dataset_type = 'jbi'
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
            timeStamp = (sys.argv[sys.argv.index('--timeStamp')+1])
            print(f'(User Defined) timeStamp: {timeStamp}')
        except: pass
        # BATCH_SIZE
        try:
            BATCH_SIZE = int(sys.argv[sys.argv.index('--BATCH_SIZE')+1])
            print(f'(User Defined) BATCH_SIZE: {BATCH_SIZE}')
        except: pass
        # dataset_type
        try:
            dataset_type = str(sys.argv[sys.argv.index('--dataset_type')+1])
            print(f'(User Defined) dataset_type: {dataset_type}')
        except: pass
        # use_best
        try:
            use_best = (sys.argv[sys.argv.index('--use_best')+1] == 'True')
            print(f'(User Defined) use_best: {use_best}')
        except: pass
    
    
    ## Use Debugger ##
    if(debugger):
        import pdb; pdb.set_trace() 

    if dataset_type == 'jbi':
        dataset_type_figure_str = 'JBI 2022'
    ## Model Name ##
    epoch = epoch_list[iteration-1]
    
    pathlib.Path(f'{current_dir}{sep}figures{sep}').mkdir(parents=True, exist_ok=True)
    if(augment): 
        # Augmented
        model_name = f'tile-classifier_{model_type}_augmented_{augmentation_type}_{dataset_type}_{timeStamp}_iteration_{iteration}_seed_{random_state}'
        rejection_score_filename = f"rejection_scores_{model_type}_{dataset_type}_{timeStamp}_augmented_{augmentation_type}_iteration_{iteration}_seed_{random_state}.npy"
        fig_title = f"Augmented with {augmentation_type} Classifier: {model_type} - {dataset_type_figure_str}"
        fig_roc = f"{current_dir}{sep}figures{sep}Tile_ROC_{model_type}_{dataset_type}_augmented_{augmentation_type}_iteration_{iteration}_seed_{random_state}.png"
        fig_confusion = f"{current_dir}{sep}figures{sep}Tile_ConfusionMatrix_{model_type}_{dataset_type}_augmented_{augmentation_type}_iteration_{iteration}_seed_{random_state}.png"
        tile_results_output = f"{current_dir}{sep}results{sep}Tile_Classification{sep}Tile_Results_{model_type}_{dataset_type}_augmented_{augmentation_type}.csv"

    else: 
        # Original
        model_name = f'tile-classifier_{model_type}_{dataset_type}_{timeStamp}_iteration_{iteration}_seed_{random_state}'
        rejection_score_filename = f"rejection_scores_{model_type}_{dataset_type}_{timeStamp}_iteration_{iteration}_seed_{random_state}.npy"
        fig_title = f"Classifier: {model_type} - {dataset_type_figure_str}"
        fig_roc = f"{current_dir}{sep}figures{sep}Tile_ROC_{model_type}_{dataset_type}_iteration_{iteration}_seed_{random_state}.png"
        fig_confusion = f"{current_dir}{sep}figures{sep}Tile_ConfusionMatrix_{model_type}_{dataset_type}_iteration_{iteration}_seed_{random_state}.png"
        tile_results_output = f"{current_dir}{sep}results{sep}Tile_Classification{sep}Tile_Results_{model_type}_{dataset_type}.csv"
    
    pathlib.Path(f'{current_dir}{sep}results{sep}Tile_Classification{sep}').mkdir(parents=True, exist_ok=True)
    
    ## Dataset ##
    if dataset_type == 'jbi':
        data_path = f"{sep}datadrive{sep}HeartTransplant{sep}ACM-2023_Data{sep}classifier_training_val_data{sep}"
        # data_path = f'{sep}home{sep}mainuser{sep}fast_datadrive{sep}HeartTransplantData{sep}2021_Data{sep}classifier_training_val_data{sep}'
        # data_path = f'{sep}home{sep}mainuser{sep}datadrive{sep}HeartTransplant{sep}2021_Data{sep}classifier_training_val_data{sep}'
        if(augment): 
            # Augmented
            data_dir = f"{data_path}augmented_{augmentation_type}_both{sep}"
        else:
            # Original
            data_dir = f"{data_path}original_both{sep}"

    else:
        raise Exception(f"The dataset_type has to be 'jbi'. Got: {dataset_type}")

    ## Model Path ##
    if use_best:
        model_weights_path = f'{current_dir}{sep}models{sep}{model_name}{sep}{model_name}_{epoch}_best.pt' 
    else:
        model_weights_path = f'{current_dir}{sep}models{sep}{model_name}{sep}{model_name}_{epoch}.pt' 

    ## Parameters Dictionary ##
    param_dict = {}
    param_dict['debugger'] = str(debugger)
    param_dict['augment'] = str(augment)
    param_dict['augmentation_type'] = augmentation_type
    param_dict['model_type'] = model_type

    param_dict['epoch'] = str(epoch)
    param_dict['timeStamp'] = str(timeStamp)
    param_dict['BATCH_SIZE'] = str(BATCH_SIZE)

    param_dict['model_name'] = str(model_name)
    param_dict['model_weights_path'] = str(model_weights_path)
    param_dict['dataset_type'] = dataset_type
    param_dict['data_dir'] = data_dir


    ### Run Model on Iterator ###
    def predict(data_iterator, device):
        labels = torch.Tensor([]).to(device)
        predicted = torch.Tensor([]).to(device)
        with torch.no_grad():
            total_batches = len(data_iterator)
            for batch_idx, (x, y) in enumerate(data_iterator):
                print(f'Test Batch: {batch_idx+1} / {total_batches} ({(batch_idx+1)*100/total_batches:.2f}%)'
                      , end='\r')
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

    ### Generate and plot AUROC and MCC ###
    def generate_auroc_mcc(predicted, actual, title, show_output=False, fig_save_path=False):
        # results_dict = {'Actual': (actual).astype(int), 'Predicted': (predicted > 0.5).astype(int)}
        # results_df = pd.DataFrame.from_dict(results_dict)
        # print(results_df)
        # print('Actual: %s'% actual)
        # print('Predicted: %s'% (predicted > 0.5).astype(int))

        ## AUROC ##
        auc_score = metrics.roc_auc_score(y_true=actual, y_score=predicted);
        ### FPR and TPR ###
        fpr, tpr, _ = metrics.roc_curve(y_true=actual, y_score=predicted, pos_label=1);
        ## MCC ##
        if( np.count_nonzero((predicted > 0.5).astype(int)) == 0 or np.count_nonzero(actual) == 0):
            mcc_score = 0
        else:
            mcc_score = metrics.matthews_corrcoef(y_true=actual, y_pred=(predicted > 0.5).astype(int) );

        ## Save ROC Plot with AUC and MCC Scores ## 
        if show_output:
            plt.plot(fpr,tpr,color="darkorange", label="ROC curve (AUC = %0.4f, MCC = %0.4f)" % (auc_score, mcc_score));
            plt.plot([0, 1], [0, 1], color="navy");
            plt.xlim([0.0, 1.0]);
            plt.ylim([0.0, 1.05]);
            plt.title(title, fontsize = 28);
            plt.xlabel("False Positive Rate (FPR)", fontsize = 16);
            plt.ylabel("True Positive Rate (TPR)", fontsize = 16);
            plt.legend(loc="lower right", fontsize = 14);

            ### Save ###
            if(fig_save_path):
                plt.savefig(fig_save_path, bbox_inches="tight");
            # plt.show();
            plt.close();
            plt.clf();
        return auc_score, mcc_score

    ### Generate Confusion Matrix Values ###
    def calc_conf_matrix(predicted, actual):
        tps = np.sum(np.logical_and(predicted == 1, actual == 1))
        fps = np.sum(np.logical_and(predicted == 1, actual == 0))
        fns = np.sum(np.logical_and(predicted == 0, actual == 1))
        tns = np.sum(np.logical_and(predicted == 0, actual == 0))
        return tps, fps, fns, tns

    ### Generate Performance Metrics ###
    def performance_metrics(predicted, actual, title, show_output=False, fig_save_path=False):
        tps,fps,fns,tns = calc_conf_matrix((predicted > 0.5).astype(int), actual.astype(int))
        sens  = np.NaN
        spec = np.NaN
        total = actual.shape[0]
        acc = (tps + tns)/total
        if(tps + fns > 0):
            sens = tps / (tps + fns)
        if(tns + fps > 0):
            spec = tns / (tns + fps) 
        conf_matrix = np.array([[tps, fps], [fns, tns]])
        auc_score, mcc_score = generate_auroc_mcc(predicted=predicted, actual=actual, title=title, show_output=show_output, fig_save_path=fig_save_path);

        return acc, sens, spec, auc_score, mcc_score, conf_matrix    



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


    ### Preprocessing ###
    # Normalization
    pretrained_size = 256 # Pixel dimension of ImageNet
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]
    # Standard Augmentation
    data_transforms = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                           ])

    ### Dataset ### (Note: 'val' will be used as Testing dataset)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms)
                      for x in ['train', 'val']}
    ### Test Dataset ###
    data_iterator = torchdata.DataLoader(image_datasets['val'], 
                                    batch_size = BATCH_SIZE,
                                        num_workers=0)

    param_dict['len(data_iterator.dataset)'] = len(data_iterator.dataset)

    print(f"#"*50)
    print(f"## Training Parameters ##")
    for key_ in param_dict.keys(): # create a tag in tensorboardx for every tag used.
        text = key_ + ': ' + str(param_dict[key_])
        print(text)

    pathlib.Path(f'{current_dir}{sep}param_dict{sep}').mkdir(parents=True, exist_ok=True)
    df_param_dict = pd.DataFrame.from_dict(param_dict, orient='index')
    df_param_dict.to_csv(f'{current_dir}{sep}param_dict{sep}{model_name}_param_tile_test_iteration_{iteration}_seed_{random_state}.csv')

    ### Run Model ###
    print(f"#"*50)
    true_labels, pred_labels = predict(data_iterator=data_iterator, device=device)

    print(f"#"*50)
    print(f"Plotting and saving Confusion Matrix and ROC")
    ### Generate Performance Metrics and Plot AUROC ###
    acc, sens, spec, auc_score, mcc_score, conf_matrix = performance_metrics(predicted=np.array(pred_labels),
                                                                             actual=np.array(true_labels.cpu()),
                                                                             title=fig_title,
                                                                             show_output=True,
                                                                             fig_save_path=fig_roc
                                                                            )
    tps, fps, fns, tns = conf_matrix.flatten()
    df_results = pd.DataFrame([[iteration, random_state, acc, sens, spec, auc_score, mcc_score, tps, fps, fns, tns]],
                             columns=['iteration', 'random_state', 'acc', 'sens', 'spec', 'auc_score', 'mcc_score', 'tps', 'fps', 'fns', 'tns'])

    if os.path.exists(tile_results_output):
        df_previous_results = pd.read_csv(tile_results_output)
        df_results = pd.concat([df_previous_results, df_results], axis=0)

    df_results.to_csv(tile_results_output, index=False)
    ### Normalized Confusion Matrix ###
    if( np.any(conf_matrix.sum(axis=1, keepdims=True), where=0) ):
        conf_intensity = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    else:
        conf_intensity = conf_matrix

    ### Plot: Confusion Matrix ###
    if(augment):
        s = sns.heatmap(conf_intensity, annot=conf_matrix, cmap='Blues',fmt=".0f", xticklabels=["Rejection", "Nonrejection"],
                        yticklabels=["Rejection", "Nonrejection"], cbar=False, robust=True, annot_kws={"size": 32})
    else: 
        s = sns.heatmap(conf_intensity, annot=conf_matrix, cmap='Oranges',fmt=".0f", xticklabels=["Rejection", "Nonrejection"],
                        yticklabels=["Rejection", "Nonrejection"], cbar=False, robust=True, annot_kws={"size": 32})    

    ## Axis Labels ##
    s.set_xlabel("Actual",fontsize=21)
    s.set_xticklabels(labels=s.get_xticklabels(), va='center', fontsize = 16)
    s.set_ylabel("Predicted",fontsize=21)
    s.set_yticklabels(labels=s.get_yticklabels(), va='center', fontsize = 16)

    s.set_title(fig_title, fontsize=18)
    plt.savefig(fig_confusion, bbox_inches="tight");
    plt.close();
    plt.clf();

    print('DONE')
