from matplotlib import pyplot as plt

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