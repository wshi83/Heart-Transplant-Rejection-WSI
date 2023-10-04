import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata

import torchvision.transforms as transforms
import torchvision.datasets as datasets

## Tensorboard ##
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

import random
import time
import os
import sys
import pathlib
import numpy as np
import pandas as pd

# Date
from datetime import datetime
from datetime import date

### Setup ###
sep = os.sep
current_dir = os.path.dirname(os.path.abspath(__file__))

### Loop Argument Processing ###
start_random_state = 1234
nb_iteration = 1

if len(sys.argv) > 1:
    print(f"#" * 50)
    print(f"User Arguments")
    # start_random_state
    try:
        start_random_state = int(sys.argv[sys.argv.index("--start_random_state") + 1])
        print(f"(User Defined) start_random_state: {start_random_state}")
    except:
        pass
    # nb_iteration
    try:
        nb_iteration = int(sys.argv[sys.argv.index("--nb_iteration") + 1])
        print(f"(User Defined) nb_iteration: {nb_iteration}")
    except:
        pass


for iteration in range(1, nb_iteration + 1):
    random_state = start_random_state + iteration - 1

    ### Seed ###
    torch.cuda.empty_cache()
    random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
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
    print(f"#" * 50)
    print(f"Device Used")
    print(device)
    ## Print Device Properties ##
    if device == torch.device("cuda"):
        print(torch.cuda.get_device_properties(device))

    ### Command Line Argument Processing ###
    debugger = False
    augment = False
    augmentation_type = "GAN"  # can also be 'diffusion'
    model_type = "vgg19"  # can also be 'resnet50' or 'resnet152' or 'densenet161'
    launch_tensorboard = False

    if len(sys.argv) > 1:
        print(f"#" * 50)
        print(f"User Arguments")
        print(sys.argv)
        # Use Debugger
        try:
            debugger = sys.argv[sys.argv.index("--debug") + 1] == "True"
            print(f"(User Defined) Use Debugger: {debugger}")
        except:
            pass
        # Augment
        try:
            augment = sys.argv[sys.argv.index("--augment") + 1] == "True"
            print(f"(User Defined) Augmentation: {augment}")
        except:
            pass
        # augmentation_type
        try:
            augmentation_type = sys.argv[sys.argv.index("--augmentation_type") + 1]
            print(f"(User Defined) augmentation_type: {augmentation_type}")
        except:
            pass
        # model_type
        try:
            model_type = sys.argv[sys.argv.index("--model_type") + 1]
            print(f"(User Defined) model_type: {model_type}")
        except:
            pass
        # launch_tensorboard
        try:
            launch_tensorboard = (
                sys.argv[sys.argv.index("--launch_tensorboard") + 1] == "True"
            )
            print(f"(User Defined) launch_tensorboard: {launch_tensorboard}")
        except:
            pass

    ### User Variables ###
    EPOCHS = 50
    VALID_RATIO = 0.75
    BATCH_SIZE = 24
    LR = 1e-5
    dataset_type = "jbi"

    if len(sys.argv) > 1:
        print(f"#" * 50)
        print(f"## User Variables ##")
        # EPOCHS
        try:
            EPOCHS = int(sys.argv[sys.argv.index("--EPOCHS") + 1])
            print(f"(User Defined) EPOCHS: {EPOCHS}")
        except:
            pass
        # VALID_RATIO
        try:
            VALID_RATIO = int(sys.argv[sys.argv.index("--VALID_RATIO") + 1])
            print(f"(User Defined) VALID_RATIO: {VALID_RATIO}")
        except:
            pass
        # BATCH_SIZE
        try:
            BATCH_SIZE = int(sys.argv[sys.argv.index("--BATCH_SIZE") + 1])
            print(f"(User Defined) BATCH_SIZE: {BATCH_SIZE}")
        except:
            pass
        # LR
        try:
            LR = float(sys.argv[sys.argv.index("--LR") + 1])
            print(f"(User Defined) LR: {LR}")
        except:
            pass
        # dataset_type
        try:
            dataset_type = str(sys.argv[sys.argv.index("--dataset_type") + 1])
            print(f"(User Defined) dataset_type: {dataset_type}")
        except:
            pass

    ## Use Debugger ##
    if debugger:
        import pdb

        pdb.set_trace()

    ## Model Name ##
    if augment:
        # Augmented
        model_name = f"tile-classifier_{model_type}_augmented_{augmentation_type}_{dataset_type}_{date.today()}_iteration_{iteration}_seed_{random_state}"
    else:
        # Original
        model_name = f"tile-classifier_{model_type}_{dataset_type}_{date.today()}_iteration_{iteration}_seed_{random_state}"

    ## Dataset ##
    if dataset_type == "jbi":
        data_path = f"{sep}datadrive{sep}HeartTransplant{sep}ACM-2023_Data{sep}classifier_training_val_data{sep}"
        # data_path = f'{sep}home{sep}mainuser{sep}datadrive{sep}HeartTransplant{sep}2021_Data{sep}classifier_training_val_data{sep}'
        if augment:
            # Augmented
            data_dir = f"{data_path}augmented_{augmentation_type}_both{sep}"
        else:
            # Original
            data_dir = f"{data_path}original_both{sep}"

    else:
        raise Exception(f"The dataset_type has to be 'jbi'. Got: {dataset_type}")

    ## Parameters Dictionary ##
    param_dict = {}
    param_dict["debugger"] = str(debugger)
    param_dict["augment"] = str(augment)

    param_dict["EPOCHS"] = str(EPOCHS)
    param_dict["VALID_RATIO"] = str(VALID_RATIO)
    param_dict["BATCH_SIZE"] = str(BATCH_SIZE)
    param_dict["LR"] = str(LR)
    param_dict["dataset_type"] = dataset_type
    param_dict["data_dir"] = data_dir
    param_dict["random_state"] = random_state
    param_dict["iteration"] = iteration

    ### Tensorboard ###
    ## Port ##
    tb_port = 6006
    ## Save data for Tensorboard ##
    pathlib.Path(f"{current_dir}{sep}runs{sep}tile{sep}{model_name}").mkdir(
        parents=True, exist_ok=True
    )
    writer = SummaryWriter(f"{current_dir}{sep}runs{sep}tile{sep}{model_name}")

    if launch_tensorboard:
        ## Start Tensorboard on port 6006 ##
        print(f"#" * 50)
        print(f"Launching TensorBoard on port: {tb_port}")
        tb = program.TensorBoard()
        tb.configure(
            argv=[
                None,
                "--logdir",
                f"{current_dir}{sep}runs/tile/",
                "--host",
                "0.0.0.0",
                "--reload_interval",
                "5",
                "--port",
                str(tb_port),
            ]
        )
        url = tb.launch()

    for key_ in param_dict.keys():  # create a tag in tensorboardx for every tag used.
        text = key_ + ": " + str(param_dict[key_])
        writer.add_text(f"Parameters/{key_}", text)

    ### Model: VGG19 with ImageNet weights ###
    if model_type == "vgg19":
        model = models.vgg19(weights="DEFAULT")
        # DL Features from model
        IN_FEATURES = model.classifier[-1].in_features
        # New classification layer
        final_fc = nn.Linear(IN_FEATURES, 2)
        # Append new classification layer
        model.classifier[-1] = final_fc
    elif model_type == "resnet50":
        model = models.resnet50(weights="DEFAULT")
        # DL Features from model
        IN_FEATURES = model.fc.in_features
        # New classification layer
        final_fc = nn.Linear(IN_FEATURES, 2)
        # Append new classification layer
        model.fc = final_fc
    elif model_type == "resnet152":
        model = models.resnet152(weights="DEFAULT")
        # DL Features from model
        IN_FEATURES = model.fc.in_features
        # New classification layer
        final_fc = nn.Linear(IN_FEATURES, 2)
        # Append new classification layer
        model.fc = final_fc
    elif model_type == "densenet161":
        model = models.densenet161(weights="DEFAULT")
        # DL Features from model
        IN_FEATURES = model.classifier.in_features
        # New classification layer
        final_fc = nn.Linear(IN_FEATURES, 2)
        # Append new classification layer
        model.classifier = final_fc
    else:
        raise Exception(f"Can not use the following model_type: {model_type}")

    ## Model: to Device ##
    if torch.cuda.device_count() > 1:
        print(f"#" * 50)
        print(f"Total GPUs: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)
    ## Model: Send to device ##
    model.to(device)

    # Freeze all layers except the new classification layer
    # for parameter in model.classifier[:-1].parameters():
    #     parameter.requires_grad = False
    ### Preprocessing ###
    # Normalization
    pretrained_size = 256  # Pixel dimension of ImageNet
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]
    # Standard Augmentation
    data_transforms = transforms.Compose(
        [
            transforms.Resize(pretrained_size),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.RandomCrop(pretrained_size, padding = 10),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means, std=pretrained_stds),
        ]
    )

    # Image Dictionary (Note: 'train' will be split into training and validation, and 'val' will be used as Testing dataset)
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms)
        for x in ["train", "val"]
    }

    # Training data (to be spit into Training/Validation 75:25 ratio)
    train_data = image_datasets["train"]
    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples
    # Create Training and Validation datasets
    train_data, valid_data = torchdata.random_split(
        train_data, [n_train_examples, n_valid_examples]
    )

    # Data Loaders
    train_iterator = torchdata.DataLoader(
        train_data, shuffle=True, batch_size=BATCH_SIZE
    )

    valid_iterator = torchdata.DataLoader(valid_data, batch_size=BATCH_SIZE)

    ## Not Used ##
    # test_iterator = torchdata.DataLoader(image_datasets['val'],
    #                                 batch_size = BATCH_SIZE)

    ### Loss ###
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Model Parameters
    # params = [
    #           {'params': model.features.parameters(), 'lr': FOUND_LR / 10},
    #           {'params': model.classifier.parameters()}
    #          ]

    ### Optimizer ###
    # optimizer = optim.Adam(model.parameters(), lr = START_LR)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    # optimizer = optim.Adam(params, lr = FOUND_LR)

    # Performance function
    def calculate_accuracy(y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred))
        incorrect = top_pred.ne(y.view_as(top_pred))

        truePositive = top_pred.eq(torch.ones(top_pred.size(), device=device)) * correct
        truePositive = truePositive.sum()

        trueNegative = (
            top_pred.eq(torch.zeros(top_pred.size(), device=device)) * correct
        )
        trueNegative = trueNegative.sum()

        # print("TN: ", trueNegative.numpy())
        # print("TP: ", truePositive.numpy())

        falsePositive = (
            top_pred.eq(torch.ones(top_pred.size(), device=device)) * incorrect
        )
        falsePositive = falsePositive.sum()

        falseNegative = (
            top_pred.eq(torch.zeros(top_pred.size(), device=device)) * incorrect
        )
        falseNegative = falseNegative.sum()

        # print("FN: ", falseNegative.numpy())
        # print("FP: ", falsePositive.numpy())

        correct = correct.sum()
        acc = correct.float() / y.shape[0]
        return acc, truePositive, trueNegative, falsePositive, falseNegative

    ### Train (Epoch) ###
    def train(model, iterator, optimizer, criterion, device):
        epoch_loss = 0
        epoch_acc = 0

        model.train()
        total_batches = len(iterator)
        # Iterate across Batches
        total_pred = None
        total_correctness = None
        for batch_idx, (x, y) in enumerate(iterator):
            print(
                f"Train Batch: {batch_idx+1} / {total_batches} ({(batch_idx+1)*100/total_batches:.2f}%)",
                end="\r",
            )
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Predict
            y_pred = model(x)
            y_pred = torch.nn.functional.softmax(y_pred, -1)
            y_p = torch.argmax(y_pred, -1)
            correctness = y_p == y
            temp_pred = [y_pred[i][y[i]] for i in range(len(y))]
            temp_pred = torch.tensor(temp_pred)

            if total_pred == None:
                total_pred = temp_pred.detach().cpu()
                total_correctness = correctness.detach().cpu()
            else:
                total_pred = torch.cat((total_pred, temp_pred.detach().cpu()), 0)
                total_correctness = torch.cat(
                    (total_correctness, correctness.detach().cpu()), 0
                )

            # Loss
            loss = criterion(y_pred, y)
            # Performance
            acc, tp, tn, tp, fn = calculate_accuracy(y_pred, y)
            # Backpropagate
            loss.backward()
            optimizer.step()
            # Record
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f"\r")
        # Return Epoch statistics (Average Batch Loss for Epoch)
        return (
            epoch_loss / len(iterator),
            epoch_acc / len(iterator),
            total_pred,
            total_correctness,
        )

    ### Evaluate (Epoch) ###
    def evaluate(model, iterator, criterion, device):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            total_batches = len(iterator)
            # Iterate across Batches
            for batch_idx, (x, y) in enumerate(iterator):
                print(
                    f"Val Batch: {batch_idx+1} / {total_batches} ({(batch_idx+1)*100/total_batches:.2f}%)",
                    end="\r",
                )
                x = x.to(device)
                y = y.to(device)

                # Predict
                y_pred = model(x)
                # Loss
                loss = criterion(y_pred, y)
                # Performance
                acc, tp, tn, fp, fn = calculate_accuracy(y_pred, y)
                # Record
                TP += tp.item()
                TN += tn.item()
                FP += fp.item()
                FN += fn.item()
                epoch_loss += loss.item()
                epoch_acc += acc.item()

            print(f"\r")
            print(f"\tTP: {TP}")
            print(f"\tTN: {TN}")
            print(f"\tFP: {FP}")
            print(f"\tFN: {FN}")

            # Sensitivity and Specificity (Epoch)
            if TP + FN > 0:
                sens = TP / (TP + FN)
            if TN + FP > 0:
                spec = TN / (TN + FP)

            print(f"\tSensitivity: {sens}")
            print(f"\tSpecificity: {spec}")

        # Return Epoch statistics (Average Batch Loss for Epoch)
        return epoch_loss / len(iterator), epoch_acc / len(iterator), sens, spec

    ### Time ###
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    print(f"#" * 50)
    print(f"Starting Training ...")
    ### Iterate across Epochs ###
    best_valid_loss = float("inf")
    best_epoch = 1
    pathlib.Path(f"{current_dir}{sep}models{sep}{model_name}{sep}").mkdir(
        parents=True, exist_ok=True
    )
    train_probs = []
    train_correctness_list = []
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1} / {EPOCHS}")
        ### Time ###
        start_time = time.monotonic()

        ### Train ###
        train_loss, train_acc, train_preds, train_correctness = train(
            model, train_iterator, optimizer, criterion, device
        )
        # train_preds = torch.nn.functional.softmax(train_preds, -1)
        train_probs.append(train_preds)
        train_correctness_list.append(train_correctness)
        ### Validate ###
        valid_loss, valid_acc, valid_sens, valid_spec = evaluate(
            model, valid_iterator, criterion, device
        )
        ### Save Model ###
        # Best Model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                f"{current_dir}{sep}models{sep}{model_name}{sep}{model_name}_{epoch+1}_best.pt",
            )
        # Every 10 Epochs
        elif (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                f"{current_dir}{sep}models{sep}{model_name}{sep}{model_name}_{epoch+1}.pt",
            )
        # Final Model
        elif epoch + 1 == EPOCHS:
            torch.save(
                model.state_dict(),
                f"{current_dir}{sep}models{sep}{model_name}{sep}{model_name}_{epoch+1}.pt",
            )

        ### Time ###
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Print Epoch Statistics
        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

        ### Tensorboard ###
        ## Loss: Training ##
        writer.add_scalar("Training loss", train_loss, epoch + 1)
        ## Loss: Validation ##
        writer.add_scalar("Validation loss", valid_loss, epoch + 1)

    param_dict["len(train_iterator.dataset)"] = len(train_iterator.dataset)
    param_dict["len(valid_iterator.dataset)"] = len(valid_iterator.dataset)

    print(f"#" * 50)
    print(f"## Training Parameters ##")
    for key_ in param_dict.keys():  # create a tag in tensorboardx for every tag used.
        text = key_ + ": " + str(param_dict[key_])
        print(text)

    pathlib.Path(f"{current_dir}{sep}param_dict{sep}").mkdir(
        parents=True, exist_ok=True
    )
    df_param_dict = pd.DataFrame.from_dict(param_dict, orient="index")
    df_param_dict.to_csv(
        f"{current_dir}{sep}param_dict{sep}{model_name}_param_tile_train_iteration_{iteration}_seed_{random_state}.csv"
    )

    print(f"#" * 50)
    print(f"Best Epoch: {best_epoch}")
    ## Empty GPU Cache ##
    torch.cuda.empty_cache()
    ## End Tensorboard Writer ##
    writer.close()

    train_probs = torch.stack(train_probs, 0)
    print(train_probs.shape)
    if augment == False:
        augmentation_type = "origin"
    torch.save(
        train_probs,
        "/home/HeartTransplant_JBI/cache/train_probs-{}-{}.pt".format(
            augmentation_type, model_type
        ),
    )
    train_correctness_list = torch.stack(train_correctness_list, 0)
    print(train_correctness_list.shape)
    torch.save(
        train_correctness_list,
        "/home/HeartTransplant_JBI/cache/train_correctness-{}-{}.pt".format(
            augmentation_type, model_type
        ),
    )
