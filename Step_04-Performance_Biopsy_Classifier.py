import os
import sys
import ast
import pathlib
import glob

import random

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rcParams['figure.facecolor'] = 'white'

# Read Image
import PIL

# TQDM
from tqdm import tqdm, trange

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV

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
    random_seed = random_state
    ### Seed ###
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)

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
    timeStamp = str(date.today())
    if( len(sys.argv)>1 ):
        print(f"#"*50)
        print(f"## User Variables ##")
        # timeStamp
        try:
            timeStamp = str(sys.argv[sys.argv.index('--timeStamp')+1])
            print(f'(User Defined) timeStamp: {timeStamp}')
        except: pass

    ## Use Debugger ##
    if(debugger):
        import pdb; pdb.set_trace() 


    ######################################
    ############# Path Setup #############
    ######################################    
    ## Augmented ##
    if(augment): 
        model_name = f'tile-classifier_{model_type}_augmented_{augmentation_type}_{dataset_type}_{timeStamp}_iteration_{iteration}_seed_{random_state}'
        # Rejection Scores Filename
        rejection_score_filename = f"rejection_scores_{model_type}_{timeStamp}_augmented_{augmentation_type}_{dataset_type}_iteration_{iteration}_seed_{random_state}.npy"
        ### Figure: ROC ###
        figure_ROC_path = f"{current_dir}{sep}figures{sep}Biopsy_ROC_{model_type}_augmented_{augmentation_type}_{dataset_type}_iteration_{iteration}_seed_{random_state}.png"
        ### Figure: Confusion Matrix ###
        figure_CM_path = f"{current_dir}{sep}figures{sep}ConfusionMatrix_{model_type}_augmented_{augmentation_type}_{dataset_type}_iteration_{iteration}_seed_{random_state}.png"
        wsi_results_output = f"{current_dir}{sep}results{sep}WSI_Classification{sep}WSI_Results_{model_type}_{dataset_type}_augmented_{augmentation_type}.csv"
        outcome_file_path = f"{current_dir}{sep}results{sep}WSI_Classification{sep}"\
    + f"WSI_results_{model_type}_{dataset_type}_augmented_{augmentation_type}{sep}"\
    + f"WSI_results_{model_type}_{dataset_type}_augmented_{augmentation_type}_state_{random_state}_iteration_{iteration}.csv"
        pathlib.Path(f'{current_dir}{sep}results{sep}WSI_Classification{sep}WSI_results_{model_type}_{dataset_type}_augmented_{augmentation_type}{sep}').mkdir(parents=True, exist_ok=True)
    ## Original ##
    else: 
        model_name = f'tile-classifier_{model_type}_{dataset_type}_{timeStamp}_iteration_{iteration}_seed_{random_state}'
        # Output Name
        rejection_score_filename = f"rejection_scores_{model_type}_{timeStamp}_{dataset_type}_iteration_{iteration}_seed_{random_state}.npy"
        ### Figure: ROC ###
        figure_ROC_path = f"{current_dir}{sep}figures{sep}Biopsy_ROC_{model_type}_{dataset_type}_iteration_{iteration}_seed_{random_state}.png"
        ### Figure: Confusion Matrix ###
        figure_CM_path = f"{current_dir}{sep}figures{sep}ConfusionMatrix_{model_type}_{dataset_type}_iteration_{iteration}_seed_{random_state}.png"
        wsi_results_output = f"{current_dir}{sep}results{sep}WSI_Classification{sep}WSI_Results_{model_type}_{dataset_type}.csv"
        
        outcome_file_path = f"{current_dir}{sep}results{sep}WSI_Classification{sep}"\
    + f"WSI_results_{model_type}_{dataset_type}{sep}"\
    + f"WSI_results_{model_type}_{dataset_type}_state_{random_state}_iteration_{iteration}.csv"
        pathlib.Path(f'{current_dir}{sep}results{sep}WSI_Classification{sep}WSI_results_{model_type}_{dataset_type}{sep}').mkdir(parents=True, exist_ok=True)


    pathlib.Path(f'{current_dir}{sep}results{sep}WSI_Classification{sep}').mkdir(parents=True, exist_ok=True)
    #####################################
    ########## Print Variables ##########
    #####################################
    ## Parameters Dictionary ##
    param_dict = {}
    param_dict['debugger'] = str(debugger)
    param_dict['augment'] = str(augment)
    param_dict['augmentation_type'] = str(augmentation_type)
    param_dict['model_type'] = str(model_type)

    param_dict['timeStamp'] = str(timeStamp)
    
    param_dict['rejection_score_filename'] = str(rejection_score_filename)
    param_dict['figure_ROC_path'] = str(figure_ROC_path)
    param_dict['figure_CM_path'] = str(figure_CM_path)

    print(f"#"*50)
    print(f"## Training Parameters ##")
    for key_ in param_dict.keys(): # create a tag in tensorboardx for every tag used.
        text = key_ + ': ' + str(param_dict[key_])
        print(text)

    pathlib.Path(f'{current_dir}{sep}param_dict{sep}').mkdir(parents=True, exist_ok=True)
    df_param_dict = pd.DataFrame.from_dict(param_dict, orient='index')
    df_param_dict.to_csv(f'{current_dir}{sep}param_dict{sep}{model_name}_param_biopsy_prediction_iteration_{iteration}_seed_{random_state}.csv')
    
    
    num_bins = [2,10,50,100,200,300,400]

    ## Metadata ##
    metadata_path = f"{sep}home{sep}mainuser{sep}datadrive{sep}HeartTransplant{sep}Metadata{sep}Metadata_wsi_previous.csv"
    metadata_df = pd.read_csv(metadata_path)

    ## WSI Dataset Path ##
    wsi_train_path = f"{sep}home{sep}mainuser{sep}datadrive{sep}HeartTransplant{sep}2021_Data{sep}wsi_classifier_training_val_data{sep}wsi_train{sep}"
    wsi_test_path = f"{sep}home{sep}mainuser{sep}datadrive{sep}HeartTransplant{sep}2021_Data{sep}wsi_classifier_training_val_data{sep}wsi_test{sep}"

    ## Training Data ##
    train_wsis = glob.glob(wsi_train_path +'[SC]*[0-9]')

    ## Test Data ##
    test_wsis = glob.glob(wsi_test_path +'[SC]*[0-9]')

    models = {}


    #######################################
    ############## Functions ##############
    #######################################
    ### Generate Performance Metrics ###
    def generate_auroc_mcc(predicted, actual, title, show_output=False, fig_save_path=False):  
        ### Confusion Matrix Values ###
        tps,fps,fns,tns = calc_conf_matrix((predicted > 0.5).astype(int), actual.astype(int))

        ## AUROC ##
        auc_score = metrics.roc_auc_score(y_true=actual, y_score=predicted);
        ### FPR and TPR ###
        fpr, tpr, _ = metrics.roc_curve(y_true=actual, y_score=predicted, pos_label=1);
        ## MCC ##
        if(tps == 0 or tns == 0):
            mcc_score = 0
        else:
            mcc_score = metrics.matthews_corrcoef(y_true=actual, y_pred=(predicted > 0.5).astype(int) );

        ### Plot ###
        if show_output:
            plt.plot(fpr, tpr, color="darkorange", label="ROC curve (AUC = %0.4f, MCC = %0.4f)" % (auc_score, mcc_score));
            plt.plot([0, 1], [0, 1], color="navy");
            plt.xlim([0.0, 1.0]);
            plt.ylim([0.0, 1.05]);
            plt.title(title, fontsize = 20);
            plt.xlabel("False Positive Rate (FPR)", fontsize = 16);
            plt.ylabel("True Positive Rate (TPR)", fontsize = 16);
            plt.legend(loc="lower right", fontsize = 14);

            ### Save ###
            if(fig_save_path):
                plt.savefig(fig_save_path, bbox_inches="tight");
            plt.show();
            plt.close();
            plt.clf();
        return auc_score, mcc_score

    ### Generate Confusion Matrix Values ###
    def calc_conf_matrix(predicted, actual, verbose=False):
        if(verbose):
            print('predicted: %s'% predicted)
            print('actual: %s'% actual)
        tps = np.sum(np.logical_and(predicted == 1, actual == 1))
        fps = np.sum(np.logical_and(predicted == 1, actual == 0))
        fns = np.sum(np.logical_and(predicted == 0, actual == 1))
        tns = np.sum(np.logical_and(predicted == 0, actual == 0))
        return tps, fps, fns, tns

    ### Generate WSI Performance Metrics ###
    def calc_wsi_metrics(predicted, actual, title, show_output=False, fig_save_path=False):
        ### Confusion Matrix Values ###
        tps,fps,fns,tns = calc_conf_matrix((predicted > 0.5).astype(int), actual.astype(int))
        # print("TP: %f, FP: %f, FN: %f, TN: %f" % (tps, fps, fns, tns))

        ### Accuracy ###
        total = actual.shape[0]
        acc = (tps + tns)/total

        ### Sensitivity ###
        sens  = np.NaN
        if(tps + fns > 0):
            sens = tps / (tps + fns)
        ### Specificity ###
        spec = np.NaN
        if(tns + fps > 0):
            spec = tns / (tns + fps) 

        ### Confusion Matrix ###
        conf_matrix = np.array([[tps, fps], [fns, tns]])
        auc_score, mcc_score = generate_auroc_mcc(predicted=predicted, actual=actual, 
                                                  title=title, show_output=show_output, fig_save_path=fig_save_path);

        ### Return ###
        return acc, sens, spec, auc_score, mcc_score, conf_matrix

    ### Plot Confusion Matrix ###
    def plot_confusion_matrix(confusion_matrix, title, save_fig_path=False):
        conf = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        confusion_plot = sns.heatmap(conf,annot=confusion_matrix,cmap='Blues',fmt=".0f", xticklabels=["Rejection", "Nonrejection"],
        yticklabels=["Rejection", "Nonrejection"],cbar=False, robust=True)
        confusion_plot.set_xlabel("Actual",fontsize=15)
        confusion_plot.set_ylabel("Predicted",fontsize=15)
        confusion_plot.set_title(title)

        if(save_fig_path):
            if(not os.path.exists(path)):
                os.makedirs(path)
            confusion_plot.get_figure().savefig(save_fig_path)

        plt.clf()

    def calc_histogram(hist_arr, wsi_name='', bins=10, show=False):
        ### Bins ###
        hist_bins = np.linspace( 0, 1.0, bins+1 )
        ### Histogram ###
        histogram_wsi = np.histogram(hist_arr, bins=hist_bins, density=True)
        ### Plot ###
        if show:
            plt.hist(hist_arr, bins=hist_bins, density=True)
            plt.xticks(np.arange(0,1,0.1))
            plt.title(wsi_name)
            plt.xlabel("Probability")
            plt.ylabel("Count")
        return histogram_wsi

    ### Return Histogram Bin values ###
    def gen_data(bin_idx, wsi_paths, rejection_score_filename):
        input_arr = []
        labels = []
        for wsi_path in wsi_paths:
            wsi_name = wsi_path.split("/")[-1]
            # print(wsi_name)
            hist_arr = np.load(wsi_path + "/" + rejection_score_filename)

            ### Histogram with bin_idx bins ###
            hist_details = calc_histogram(hist_arr, wsi_name, bins=bin_idx, show=False)
            input_arr.append(hist_details[0])

            ### Label ###
            label = metadata_df[metadata_df["Filename"] == f"{wsi_name}"]["Label"].iloc[0]
            labels.append(label)
        return input_arr, labels
    
    
    ## Model Hyperparameter Tuning (5-Fold CV is the default for GridSearchCV)
    
    ### XGBoost Random Forest
    model_name = "XGBoost_RF"
    for loop_idx,bin_idx in enumerate(num_bins):
        print('%s / %s'%(loop_idx,len(num_bins)), end='\r')
        train_arr, train_labels = gen_data(bin_idx, wsi_paths=train_wsis, rejection_score_filename=rejection_score_filename)
        model = xgb.XGBRFClassifier(random_state=random_seed)
        model = GridSearchCV(model, {'max_depth': [2,3,4,5,6,7,8,9,10],
                                     'n_estimators': [2,3,4,5,6,10,25,50, 100, 200, 300, 400], 
                                     'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]},
                             n_jobs=-1)
        model.fit(train_arr, train_labels)
        if(not model_name in models.keys()):
            models[model_name] = {}
        models[model_name][bin_idx] = {'model':model, 'score':model.best_score_}
    print('DONE     ')
    
    ### XGBoost
    model_name = "XGBoost"
    for loop_idx,bin_idx in enumerate(num_bins):
        print('%s / %s'%(loop_idx,len(num_bins)), end='\r')
        train_arr, train_labels = gen_data(bin_idx, wsi_paths=train_wsis, rejection_score_filename=rejection_score_filename)
        model = xgb.XGBClassifier(random_state=random_seed)
        model = GridSearchCV(model, {'max_depth': [2,3,4,5,6,7,8,9,10],
                                     'n_estimators': [2,3,4,5,6,10,25,50, 100, 125, 150, 175, 200, 300, 400], 
                                     'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]},
                             n_jobs=-1)
        model.fit(train_arr, train_labels)
        if(not model_name in models.keys()):
            models[model_name] = {}
        models[model_name][bin_idx] = {'model':model, 'score':model.best_score_}
    print('DONE     ')
    
    ### Random Forest
    model_name = "Random Forest Classifier"
    for loop_idx,bin_idx in enumerate(num_bins):
        print('%s / %s'%(loop_idx,len(num_bins)), end='\r')
        train_arr, train_labels = gen_data(bin_idx, wsi_paths=train_wsis, rejection_score_filename=rejection_score_filename)
        model = RandomForestClassifier(random_state=random_seed)
        model = GridSearchCV(model, {'max_depth': [2,3,4,5,6,7,8,9,10], 
                                     'n_estimators': [2,3,4,5,6,10,25,50, 100, 125, 150, 175, 200, 300, 400]},
                             n_jobs=-1)
        model.fit(train_arr, train_labels)
        if(not model_name in models.keys()):
            models[model_name] = {}
        models[model_name][bin_idx] = {'model':model, 'score':model.best_score_}
    print('DONE     ')
    
    ### Decision Tree
    model_name = "Decision Tree Classifier"
    for loop_idx,bin_idx in enumerate(num_bins):
        print('%s / %s'%(loop_idx,len(num_bins)), end='\r')
        train_arr, train_labels = gen_data(bin_idx, wsi_paths=train_wsis, rejection_score_filename=rejection_score_filename)
        model = sklearn.tree.DecisionTreeClassifier(random_state=random_seed)
        model = GridSearchCV(model, {'min_samples_split': list( range(1,20) ),
                                     "min_samples_leaf": list( range(1,20) )},
                             n_jobs=-1)
        model.fit(train_arr, train_labels)
        if(not model_name in models.keys() ):
            models[model_name] = {}
        models[model_name][bin_idx] = {'model':model, 'score':model.best_score_}
    print('DONE     ')
    
    
    ## Performance: Test Dataset
    ### Iterate across Bins ###
    outcome_df={}
    for bin_idx in num_bins:
        ### Generate Test Dataset Features and Labels ###
        test_arr, test_labels = gen_data(bin_idx, wsi_paths=test_wsis, rejection_score_filename=rejection_score_filename)

        ### Iterate across Models ###
        for model_name, model_info in models.items():
            ### Model ###
            model = model_info[bin_idx]['model'] 

            ### Prediction ###
            probs = model.predict_proba(test_arr)[:,1]

            ### Performance ###
            acc, sens, spec, auc_score, mcc_score,conf_matrix = calc_wsi_metrics(predicted=np.array(probs),
                                                                                 actual=np.array(test_labels),
                                                                                 title='%s Bins: %s'%(model_name,bin_idx),
                                                                                 show_output=False)

            ### Performance Dictionary ###
            outcome_dict = {
                'Model':[model_name],
                'Score (Validation)': [model_info[bin_idx]['score']],
                'Score (Test)': [model.score(test_arr,test_labels)],
                'Accuracy (Test)':[acc],
                'AUROC (Test)':[auc_score],
                'MCC (Test)':[mcc_score],
                'Sensitivity (Test)':[sens],
                'Specificity (Test)': [spec],
                'Bins': [bin_idx] 
            }
            ### Append Model Performance to outcome_df as Row ###
            if( len(outcome_df) == 0):
                outcome_df = pd.DataFrame(data=outcome_dict )
            else:
                outcome_df = pd.concat([outcome_df, pd.DataFrame(data=outcome_dict)], ignore_index = True)
    print('DONE')
    
    
    ### Sort by MCC ###
    outcome_df = outcome_df.sort_values(by=['MCC (Test)'], ascending=False )

    outcome_df.to_csv(outcome_file_path)
    outcome_df
    
    
    # Final Model Performance
    
    ### Best Model ###
    bin_idx= outcome_df.iloc[0]["Bins"]
    # model_name = outcome_df.iloc[0]["Model"]
    model_name = 'Random Forest Classifier'
    ### Model ###
    model = models[model_name][bin_idx]['model']

    ### Generate Test Dataset Features and Labels ###
    test_arr, test_labels = gen_data(bin_idx, wsi_paths=test_wsis, rejection_score_filename=rejection_score_filename)

    ### Predictions ###
    probs = model.predict_proba(test_arr)[:,1]

    ### Performance and Plot ROC ###
    if(augment):
        title = f"{model_name} with {model_type} classifier and {augmentation_type} augmentation"
    else:
        title = f"{model_name} with {model_type} classifier original"
    acc, sens, spec, auc_score, mcc_score, conf_matrix = calc_wsi_metrics(np.array(probs),
                                                                       np.array(test_labels),
                                                                       title=title,
                                                                       show_output=True, fig_save_path=figure_ROC_path)
    
    tps, fps, fns, tns = conf_matrix.flatten()
    df_results = pd.DataFrame([[iteration, random_state, acc, sens, spec, auc_score, mcc_score, tps, fps, fns, tns]],
                             columns=['iteration', 'random_state', 'acc', 'sens', 'spec', 'auc_score', 'mcc_score', 'tps', 'fps', 'fns', 'tns'])

    if os.path.exists(wsi_results_output):
        df_previous_results = pd.read_csv(wsi_results_output)
        df_results = pd.concat([df_previous_results, df_results], axis=0)

    df_results.to_csv(wsi_results_output, index=False)
    
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
    s.set_xticklabels(labels=s.get_xticklabels(), va='center', fontsize = 14)
    s.set_ylabel("Predicted",fontsize=21)
    s.set_yticklabels(labels=s.get_yticklabels(), va='center', fontsize = 14)

    if(augment):
        s.set_title(f"{model_name} with {model_type} classifier and {augmentation_type} augmentation",fontsize=18)
    else:
        s.set_title(f"{model_name} with {model_type} classifier original",fontsize=18)
    plt.savefig(figure_CM_path, bbox_inches="tight")
    plt.close();
    plt.clf();
