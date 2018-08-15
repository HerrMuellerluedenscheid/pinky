import matplotlib.pyplot as plt
import math
import numpy as num


def flatten(items):
    return [ax for _ax in items for ax in _ax]


def clear_ax(ax):
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])


def adjust(fig):
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.02, hspace=0.07)


def show_data(model, shuffle=False):
    yscale = 3.
    n = 9
    n_rows = 3

    
    boxstyle = dict(boxstyle='round', facecolor='white', alpha=0.5)
    fig, axs = plt.subplots(math.ceil(n/n_rows), n_rows)
    axs = flatten(axs)

    fig_w, axs_w = plt.subplots(math.ceil(n/n_rows), n_rows)
    axs_w = flatten(axs_w)
    model.data_generator.shuffle = shuffle
    for i, (chunk, label) in enumerate(
            model.data_generator.generate()):

        if i == n:
            break

        axs[i].imshow(chunk, aspect='auto', cmap='gist_gray')
        axs[i].text(0, 0, label, transform=axs[i].transAxes, bbox=boxstyle)
        n_channels, n_samples = chunk.shape
        xdata = num.arange(n_samples)
        for irow, row in enumerate(chunk):
            axs_w[i].plot(xdata, irow+yscale*row, color='grey')

    [clear_ax(ax) for ax in axs_w]

    adjust(fig)
    adjust(fig_w)
    fig.savefig('pink_image.png')
    fig_w.savefig('pink_waves.png')

    plt.show()
