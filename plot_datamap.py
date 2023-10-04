import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import torch

### Setup ###
sep = os.sep
current_dir = os.path.dirname(os.path.abspath(__file__))


def scatter_it(
    dataframe, hue_metric="correct.", title="", model="RoBERTa", show_hist=False
):
    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(
        n=25000 if dataframe.shape[0] > 25000 else len(dataframe)
    )

    # Normalize correctness to a value between 0 and 1.
    #     dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    #     dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    main_metric = "variability"
    other_metric = "confidence"

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        ax0 = axs
    else:
        fig = plt.figure(
            figsize=(16, 10),
        )
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])

        ax0 = fig.add_subplot(gs[0, :])

    ### Make the scatterplot.

    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(
        x=main_metric,
        y=other_metric,
        ax=ax0,
        data=dataframe,
        hue=hue,
        palette=pal,
        style=style,
        s=30,
    )

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    an1 = ax0.annotate(
        "ambiguous",
        xy=(0.9, 0.5),
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        rotation=350,
        bbox=bb("black"),
    )
    an2 = ax0.annotate(
        "easy-to-learn",
        xy=(0.27, 0.85),
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        bbox=bb("r"),
    )
    an3 = ax0.annotate(
        "hard-to-learn",
        xy=(0.35, 0.25),
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        bbox=bb("b"),
    )

    if not show_hist:
        plot.legend(
            ncol=1,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fancybox=True,
            shadow=True,
        )
    else:
        plot.legend(fancybox=True, shadow=True, ncol=1)
    plot.set_xlabel("variability")
    plot.set_ylabel("confidence")

    if show_hist:
        # plot.set_title(f"{model}-{title} Data Map", fontsize=17)
        plot.set_title("Data Map", fontsize=17)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])

        plott0 = dataframe.hist(column=["confidence"], ax=ax1, color="#622a87")
        plott0[0].set_title("")
        plott0[0].set_xlabel("confidence")
        plott0[0].set_ylabel("density")

        plott1 = dataframe.hist(column=["variability"], ax=ax2, color="teal")
        plott1[0].set_title("")
        plott1[0].set_xlabel("variability")

        plot2 = sns.countplot(x="correct.", data=dataframe, color="#86bf91", ax=ax3)
        ax3.xaxis.grid(True)  # Show the vertical gridlines

        plot2.set_title("")
        plot2.set_xlabel("correctness")
        plot2.set_ylabel("")

    fig.tight_layout()
    filename = "/home/HeartTransplant_JBI/cache/datamap-{}-{}-epoch-{}-{}.pdf".format(
        augmentation_type, model_type, epoch_start, epoch_end
    )
    fig.savefig(filename, dpi=600)

for i_model_type in ["vgg19","densenet161","resnet50","resnet152"]:
    for i_epoch in range(5):
        augmentation_type = "GAN" # origin, diffusion, GAN
        model_type = i_model_type
        epoch_start = i_epoch * 10 
        epoch_end = epoch_start + 9


        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", None)
        sns.set(style="whitegrid", font_scale=1.6, font="Georgia", context="paper")

        probs_list = torch.load(
            "/home/HeartTransplant_JBI/cache/train_probs-{}-{}.pt".format(
                augmentation_type, model_type
            )
        )
        correctness_list = torch.load(
            "/home/HeartTransplant_JBI/cache/train_correctness-{}-{}.pt".format(
                augmentation_type, model_type
            )
        )
        # print(probs_list[0])
        # input()
        mean = torch.mean(probs_list[epoch_start:epoch_end], 0).squeeze()
        std = torch.std(probs_list[epoch_start:epoch_end], 0).squeeze()
        print(correctness_list.shape)
        correctness = torch.sum(correctness_list[:10], 0).squeeze() / 10
        print(mean.shape, std.shape, correctness.shape)

        plot_data = torch.stack([mean.cpu(), std.cpu(), correctness.cpu()], 0)
        print(plot_data.shape)
        plot_data = plot_data.permute(1, 0).cpu().detach().numpy()

        td_metrics = pd.DataFrame(plot_data)
        names = ["confidence", "variability", "correct."]
        td_metrics.columns = names


        scatter_it(td_metrics, title="SNLI", show_hist=True)
