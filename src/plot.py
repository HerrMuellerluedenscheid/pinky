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
    '''Returns a list of left most axes objects from 2 dimensional grid.'''
    return [ax[0] for ax in axs_grid]


def plot_labels(labels, color, title, axs=None):
    fig = None
    if axs is None:
        fig, axs = plt.subplots(1, 2)

    nlabels = len(labels)
    nlabel_components = len(labels[0])
    labels_array = num.empty((nlabels, nlabel_components))
    for i, l in enumerate(labels):
        labels_array[i, :] = l

    labels_array = num.transpose(labels_array)
    axs[0].scatter(labels_array[0], labels_array[2], c=color, s=1., alpha=0.5)
    axs[1].scatter(labels_array[1], labels_array[2], c=color, s=1., alpha=0.5,
            label=title)

    for ax in axs:
        ax.set_aspect('equal')

    fig = fig if fig is not None else plt.gcf()
    plt.legend()

    return fig, axs


def show_data(model, shuffle=False):
    '''Plot 2 dimensional feature images and waveform sections.'''
    yscale = 10.  # Use this to tune amplitudes of waveform plots
    n = 9  # total number of plots
    n_rows = 3
    figsize = (10, 8)
    boxstyle = dict(boxstyle='round', facecolor='white', alpha=0.7)
    fig, axs_grid = plt.subplots(math.ceil(n/n_rows), n_rows, figsize=figsize)
    bottom_axs = axs_grid[-1]
    axs = flatten(axs_grid)

    fig_w, axs_w_grid = plt.subplots(math.ceil(n/n_rows), n_rows, figsize=figsize)
    axs_w = flatten(axs_w_grid)

    model.config.data_generator.shuffle = shuffle
    for i, (chunk, label) in enumerate(
            model.config.data_generator.generate()):

        if i == n:
            break

        axs[i].imshow(chunk, aspect='auto', cmap='gist_gray')
        string = ' '.join([' %1.2f |'% l for l in label])
        string += '\nminmax= %1.1f| %1.1f' %(num.nanmin(chunk), num.nanmax(chunk))
        axs[i].text(
                0, 0, string, size=7,
                transform=axs[i].transAxes, bbox=boxstyle)

        _, n_samples = chunk.shape
        xdata = num.arange(n_samples)

        for irow, row in enumerate(chunk):
            row -= num.mean(row)
            axs_w[i].plot(xdata, irow+yscale*row, color='grey', linewidth=0.5)

    [clear_ax(ax) for ax in axs_w]

    labels_eval = list(model.config.evaluation_data_generator.iter_labels())
    labels_train = list(model.config.data_generator.iter_labels())

    locs = []
    labels = []
    for nslc, i in model.config.data_generator.nslc_to_index.items():
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

    fig_labels, axs_labels = plot_labels(
            labels_eval, 'red', title='eval')

    fig_labels, axs_labels = plot_labels(
            labels_train, 'blue', title='train', axs=axs_labels)

    fig_labels.savefig('pinky_labels.pdf', dpi=400)
    fig.savefig('pinky_image.pdf', dpi=400)
    fig_w.savefig('pinky_waves.pdf', dpi=400)

    plt.show()


def show_kernels_dense(weights, name=None):
    '''2 dimensional images of dense weights.'''
    fig, axs = plt.subplots(1, 1)

    axs.imshow(weights, cmap='gray')
    axs.axis('off')
    axs.set_yticks([])
    axs.set_xticks([])

    if name:
        fig.savefig(name)
    else:
        plt.show()


def show_kernels(weights, name=None):
    n_columns = 8
    n_weights = weights.shape[-1]
    n_rows = int(n_weights // n_columns)
    fig, axs = plt.subplots(n_rows, n_columns)

    axs = [ax for iax in axs for ax in iax]
    
    for iweight in range(n_weights):
        axs[iweight].imshow(weights[..., iweight], cmap='gray')
    
    for ax in axs:
        ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

    if name:
        fig.savefig(name)
    else:
        plt.show()


def getActivations(sess, layer, stimuli):
    '''Plot activations for a certain stimulus.'''
    units = sess.run(
        layer, feed_dict={x: num.reshape(stimuli,
		[1,784], order='F'), keep_prob:1.0})
    plotNNFilter(units)


def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

