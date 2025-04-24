import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(conf_mat, outdir, current_epoch):
    # normalize the confusion matrix so that the sum of the matrix is 100
    total = conf_mat.flatten().sum()
    conf_mat = conf_mat / total * 100
    # Calculate accuracy by summing diagonal elements (correct predictions) and dividing by total
    accuracy = np.sum(np.diag(conf_mat)) / 100

    plt.figure(figsize=(6, 6))
    plt.imshow(conf_mat, cmap='Blues')
    plt.colorbar()
    plt.ylabel('Expected')
    plt.xlabel('Predicted')
    plt.title(f"Confusion Matrix - Epoch {current_epoch}, accuracy {accuracy:.3f}")
    
    # Set integer ticks only
    n_classes = conf_mat.shape[0]
    plt.xticks(np.arange(n_classes))
    plt.yticks(np.arange(n_classes))
    
    # Add text annotations to show the values
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, f"{conf_mat[i, j]:.1f}", ha='center', va='center', color='black')
    
    plt.savefig(f'{outdir}/confusion_matrix_epoch{current_epoch}.png')
    plt.close()

import os
def plot_predictions(output_path, input, target, pred):
    cmap='rainbow'
    fig, axs = plt.subplots(1,3)
    axs[0].title.set_text('Input')
    axs[1].title.set_text('Target')
    axs[2].title.set_text('Prediction')
    x = input.flatten()
    y = target.flatten()
    vmin, vmax = min(x.min(), y.min()), max(x.max(), y.max())
    fig.colorbar(axs[0].imshow(input,cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax ), ax=axs[0], shrink=0.6, orientation='horizontal')
    fig.colorbar(axs[1].imshow(target,cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax ), ax=axs[1], shrink=0.6, orientation='horizontal')
    fig.colorbar(axs[2].imshow(pred,cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax ), ax=axs[2], shrink=0.6, orientation='horizontal')
    plt.savefig(output_path)
    # turn off y-axis labels
    plt.yticks([])
    plt.show()
    plt.close()
