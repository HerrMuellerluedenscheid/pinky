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
        left=0.1, right=0.99, top=0.98, bottom=0.02, wspace=0.02, hspace=0.07)


def get_left_axs(axs_grid):
    return [ax[0] for ax in axs_grid]


def show_data(model, shuffle=False):
    yscale = 10.
    n = 9
    n_rows = 3
    figsize = (10, 8)
    boxstyle = dict(boxstyle='round', facecolor='white', alpha=0.7)
    fig, axs_grid = plt.subplots(math.ceil(n/n_rows), n_rows, figsize=figsize)
    bottom_axs = axs_grid[-1]
    axs = flatten(axs_grid)

    fig_w, axs_w_grid = plt.subplots(math.ceil(n/n_rows), n_rows, figsize=figsize)
    axs_w = flatten(axs_w_grid)
    model.data_generator.shuffle = shuffle
    for i, (chunk, label) in enumerate(
            model.data_generator.generate()):

        if i == n:
            break

        axs[i].imshow(chunk, aspect='auto', cmap='gist_gray')
        string = ' '.join([str(l) for l in label])
        string += '\nminmax= %1.1f| %1.1f' %(num.min(chunk), num.max(chunk))
        axs[i].text(
                0, 0, string, size=7,
                transform=axs[i].transAxes, bbox=boxstyle)

        n_channels, n_samples = chunk.shape
        xdata = num.arange(n_samples)
        for irow, row in enumerate(chunk):
            row -= num.mean(row)
            axs_w[i].plot(xdata, irow+yscale*row, color='grey', linewidth=0.5)

    [clear_ax(ax) for ax in axs_w]

    locs = []
    labels = []
    for nslc, i in model.data_generator.nslc_to_index.items():
        locs.append(i)
        labels.append('.'.join(nslc))

    left_axs = []
    for axs in (get_left_axs(x) for x in (axs_grid, axs_w_grid)):
        for ax in axs:
            ax.set_yticks(locs)
            ax.set_yticklabels(labels, size=7)
            left_axs.append(ax)

    for ax in axs:
        if ax in left_axs:
            continue
        ax.set_yticks([])

    adjust(fig)
    adjust(fig_w)
    fig.savefig('pink_image.pdf', dpi=400)
    fig_w.savefig('pink_waves.pdf', dpi=400)

    plt.show()
