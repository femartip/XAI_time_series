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

def plot_csv_alpha_mean_loyalty(file:str):
    df = pd.read_csv(file)
    for name, group in df.groupby("Type"):
        plt.figure()
        plt.plot(group["Alpha"], group["Mean Loyalty"])
        plt.title(f"Mean Loyalty for {name}")
        plt.xlabel("Alpha")
        plt.ylabel("Mean Loyalty")
        plt.show()

def plot_csv_complexity_mean_loyalty(file:str):
    df = pd.read_csv(file)
    representation_type = ["o", "x"]
    plt.figure()
    for i, (name, group) in enumerate(df.groupby("Type")):
        plt.scatter(group["Complexity"], group["Mean Loyalty"], label=name, c=group['Alpha'], cmap='viridis', marker=representation_type[i])
    plt.title(f"Mean Loyalty")
    plt.xlabel("Complexity")
    plt.ylabel("Mean Loyalty")
    plt.legend()
    plt.colorbar(label='Alpha/Epsilon')
    plt.show()

def plot_csv_complexity_kappa_loyalty(file:str):
    df = pd.read_csv(file)
    representation_type = ["o", "x"]
    plt.figure()
    for i, (name, group) in enumerate(df.groupby("Type")):
        plt.scatter(group["Complexity"], group["Kappa Loyalty"], label=name, c=group['Alpha'], cmap='viridis', marker=representation_type[i])
    plt.title(f"Kappa Loyalty")
    plt.xlabel("Complexity")
    plt.ylabel("Kappa Loyalty")
    plt.legend()
    plt.colorbar(label='Alpha/Epsilon')
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, help="File to plot")
    args = argparser.parse_args()
    file = args.file
    plot_csv_complexity_mean_loyalty(file)
    plot_csv_complexity_kappa_loyalty(file)
