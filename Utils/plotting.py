from matplotlib import pyplot as plt
import pandas as pd
import argparse

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

def plot_csv_alpha_mean_loyalty(file:str) -> plt.Figure:
    df = pd.read_csv(file)
    fig, ax = plt.subplots()
    for name, group in df.groupby("Type"):
        ax.plot(group["Alpha"], group["Mean Loyalty"], label=name)
    ax.set_title("Mean Loyalty by Alpha")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean Loyalty")
    ax.legend()
    return fig

def plot_csv_complexity_mean_loyalty(file:str) -> plt.Figure:
    df = pd.read_csv(file)
    representation_type = ["o", "x", '+']
    fig, ax = plt.subplots()
    for i, (name, group) in enumerate(df.groupby("Type")):
        scatter = ax.scatter(group["Complexity"], group["Mean Loyalty"], label=name, c=group['Alpha'], cmap='viridis', marker=representation_type[i])
    ax.set_title(f"Mean Loyalty")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Mean Loyalty")
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Alpha/Epsilon')
    return fig

def plot_csv_complexity_kappa_loyalty(file:str) -> plt.Figure:
    df = pd.read_csv(file)
    representation_type = ["o", "x", '+']
    fig, ax = plt.subplots()
    for i, (name, group) in enumerate(df.groupby("Type")):
        scatter = ax.scatter(group["Complexity"], group["Kappa Loyalty"], label=name, c=group['Alpha'], cmap='viridis', marker=representation_type[i])
    ax.set_title(f"Kappa Loyalty")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Kappa Loyalty")
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Alpha/Epsilon')
    return fig


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, help="File to plot")
    args = argparser.parse_args()
    file = args.file
    fig1 = plot_csv_complexity_mean_loyalty(file)
    fig1.show()
    fig2 = plot_csv_complexity_kappa_loyalty(file)
    fig2.show()
