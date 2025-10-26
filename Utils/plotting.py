from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import argparse
import numpy as np
import os
import random

def plot_metrics(train_metrics, train_losses, val_metrics, val_losses):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(train_metrics, label='Train')
    ax[0].plot(val_metrics, label='Validation')
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].plot(train_losses, label='Train')
    ax[1].plot(val_losses, label='Validation')
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.show()

def plot_csv_alpha_mean_loyalty(file:str) -> Figure:
    df = pd.read_csv(file)
    fig, ax = plt.subplots()
    for name, group in df.groupby("Type"):
        ax.plot(group["Alpha"], group["Mean Loyalty"], label=name)
    ax.set_title("Mean Loyalty by Alpha")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean Loyalty")
    ax.legend()
    return fig

def plot_csv_complexity_mean_loyalty(file:str) -> Figure:
    df = pd.read_csv(file)
    representation_type = ["o", "x", '+', "|", "s"]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, group) in enumerate(df.groupby("Type")):
        scatter = ax.scatter(group["Complexity"], group["Mean Loyalty"], 
                            label=name, c=group['Alpha'], cmap='viridis', 
                            marker=representation_type[i])
    
    ax.set_title(f"Mean Loyalty")
    ax.set_xlabel("Complexity\n(Num Segments)")
    ax.set_ylabel("Mean Loyalty")
    
    min_complexity = df["Complexity"].min()
    max_complexity = df["Complexity"].max()
    num_ticks = 6
    complexity_ticks = np.linspace(min_complexity, max_complexity, num_ticks)
    
    labels = []
    for comp in complexity_ticks:
        closest_comp = df["Complexity"].iloc[(df["Complexity"] - comp).abs().argsort()[:1]].values[0]
        segments = sorted(df[df["Complexity"] == closest_comp]["Num Segments"].unique())
        segments_str = ", ".join(map(str, segments))
        labels.append(f"{comp:.1f}\n({segments_str})")
    
    ax.set_xticks(complexity_ticks)
    ax.set_xticklabels(labels)
    
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Alpha/Epsilon')
    plt.tight_layout()
    
    return fig

def plot_csv_complexity_kappa_loyalty(file:str, points:dict={}) -> Figure:
    assert isinstance(file, str), "File should be a string"
    assert os.path.exists(file), f"File {file} does not exist"
    assert file.endswith(".csv"), "File should be a CSV file"
    
    df = pd.read_csv(file)
    #representation_type = ["o", "x", '+', "|", "s", "^", "v", "D", "*"]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, group) in enumerate(df.groupby("Type")):
        if name == 'BU_1':
            continue
        elif name == 'BU_2':
            label = "BU"
        else:
            label = name
        ax.plot(group["Complexity"], group["Kappa Loyalty"], 
                            label=label, 
                            ) #c=group['Alpha'], cmap='viridis'` marker=representation_type[i]
        
    #ax.set_title(f"Kappa Loyalty")
    ax.set_xlabel("Complexity\n(Abs. Num. Segments)")
    ax.set_ylabel("Kappa Loyalty\n(% Agreement)")
    
    min_complexity = df["Complexity"].min()
    max_complexity = df["Complexity"].max()
    num_ticks = 6
    complexity_ticks = np.linspace(min_complexity, max_complexity, num_ticks)
    
    x_labels = []
    for comp in complexity_ticks:
        closest_comp = df["Complexity"].iloc[(df["Complexity"] - comp).abs().argsort()[:1]].values[0]
        segment = sorted(df[df["Complexity"] == closest_comp]["Num Segments"].unique())[0]
        segment = round(segment,2)
        x_labels.append(f"{comp:.1f}\n({segment})")

    if "Percentage Agreement" in df.columns:
        loyalty_ticks = np.linspace(df["Kappa Loyalty"].min(), df["Kappa Loyalty"].max(), num_ticks)    #Sampled points of kappa
        y_labels = []
        for loyalty in loyalty_ticks:
            closest_loyalty = df["Kappa Loyalty"].iloc[(df["Kappa Loyalty"] - loyalty).abs().argsort()[:1]].values[0]
            segments = sorted(df[df["Kappa Loyalty"] == closest_loyalty]["Percentage Agreement"].unique())[0]
            y_labels.append(f"{loyalty:.1f}\n({segments}%)")
        ax.set_yticks(loyalty_ticks)
        ax.set_yticklabels(y_labels) 
    
    #ax.set_xticks(complexity_ticks)
    ax.set_xticklabels(x_labels)

    ax.legend(loc='lower right')
    plt.tight_layout()
    
    return fig


def plot_prototipes(alpha: np.float64, pred_class: np.int8, X: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(X)
    ax.set_title(f"Prototypes for Alpha: {alpha}, Predicted Class: {pred_class}")
    return fig

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, help="File to plot")
    args = argparser.parse_args()
    file = args.file
    assert os.path.exists(file), f"File {file} does not exist"

    if file.endswith(".csv"):
        plt.rcParams.update({
            "font.size": 18,        
            "axes.titlesize": 22,   
            "axes.labelsize": 20,   
            "xtick.labelsize": 16,  
            "ytick.labelsize": 16,  
            "legend.fontsize": 16,  
        })
        fig2 = plot_csv_complexity_kappa_loyalty(file)
        plt.show()
    elif file.endswith(".npy"):
        data = np.load(file)
        rand_num = random.randint(0, len(data) - 1)
        pred_classes = [data[i][1] for i in range(len(data))]
        unique_classes = np.unique(pred_classes)
        print(f"Unique classes: {unique_classes}")
        for i in range(100):
            rand_num = random.randint(0,399)
            alpha = data[rand_num][0]; pred_class = data[rand_num][1]; X = data[rand_num][-1]
            if pred_class != 0:
                fig = plot_prototipes(alpha, pred_class, X)
                plt.show(block=False)
                plt.pause(0.5)
                plt.close()
    else:
        raise ValueError(f"File format not supported")