import matplotlib.pyplot as plt
import math
import numpy as num
import logging
from scipy import stats

from matplotlib import rc
#plt.rc('text', usetex=True)


logger = logging.getLogger('pinky.plot')
POINT_SIZE = 2.
FIG_SUF = '.pdf'
NPOINTS = 110


def save_figure(fig, name=None):
    '''Saves figure `fig` if `name` is defined. Closes the figure after
    saving.'''
    if not name:
        return

    fig.savefig(name+FIG_SUF)
    plt.close()


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


def plot_locations(locations, color, title, axs=None):
    fig = None
    if axs is None:
        fig, axs = plt.subplots(1, 2)

    nlocations = len(locations)
    nlabel_components = len(locations[0])
    locations_array = num.empty((nlocations, nlabel_components))
    for i, l in enumerate(locations):
        locations_array[i, :] = l

    locations_array = num.transpose(locations_array)

    axs[0].scatter(
            locations_array[0], locations_array[2], c=color, s=1., alpha=0.5)

    axs[1].scatter(
            locations_array[1], locations_array[2], c=color, s=1., alpha=0.5,
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
            # model.config.prediction_data_generator.generate()):
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

    fig_labels, axs_labels = plot_locations(
            labels_eval, 'red', title='eval')

    fig_labels, axs_labels = plot_locations(
            labels_train, 'blue', title='train', axs=axs_labels)

    save_figure(fig_labels, 'pinky_labels')
    save_figure(fig, 'pinky_image')
    save_figure(fig_w, 'pinky_waves.pdf')


def show_kernels_dense(weights, name=None):
    '''2 dimensional images of dense weights.'''
    fig, axs = plt.subplots(1, 1)

    axs.imshow(weights, cmap='gray')
    axs.axis('off')
    axs.set_yticks([])
    axs.set_xticks([])

    save_figure(fig, name)


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

    save_figure(fig, name)


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


def confidence(data, rate=0.95):
    return stats.t.interval(
            rate, len(data)-1, loc=num.mean(data), scale=stats.sem(data))


def hist_with_stats(data, ax):
    '''Plot a histogram of `data` into `ax` and label median and errors.'''
    nbins = 71
    ax.hist(data, bins=nbins)
    med = num.mean(data)
    ax.axvline(med, color='black')
    ax.text(0.1, 0.99,
            r'$\mu = %1.1f\pm %1.1f$' % (med, num.std(data)),
            fontsize=9,
            horizontalalignment='left',
            transform=ax.transAxes)


def mislocation_hist(predictions, labels, name=None):
    '''Plot statistics on mislocations in 3 dimensions and absolute errors.'''
    predictions = num.array(predictions)
    labels = num.array(labels)
    errors = predictions - labels
    errors_abs = num.sqrt(num.sum(errors**2, axis=1))
    
    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)

    hist_with_stats(errors.T[0], axs[0][0])
    axs[0][0].set_xlabel('Error (North) [m]')
    
    hist_with_stats(errors.T[1], axs[0][1])
    axs[0][1].set_xlabel('Error (East) [m]')
    
    hist_with_stats(errors.T[2], axs[1][0])
    axs[1][0].set_xlabel('Error (Depth) [m]')

    # hist_with_stats(errors_abs, axs[1][1])
    # axs[1][1].set_title('Absolute errors [m]')
    xticks = range(-1000, 1200, 200)
    xtick_labels = ((-1000, 0, 1000))
    for ax in flatten(axs[:3]):
        ax.set_yticks([])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save_figure(fig, name)


def error_map(prediction, label, ax):
    px, py = prediction
    lx, ly = label

    ax.plot((px, lx), (py, ly), color='grey', alpha=0.5, linewidth=0.1)
    # ax.arrow(px, lx, px-py, lx-ly, color='grey', alpha=0.5)

    ax.scatter(px, py, color='red', s=POINT_SIZE, linewidth=0)
    ax.scatter(lx, ly, color='blue', s=POINT_SIZE, linewidth=0)


def error_contourf(predictions, labels, ax):
    '''Make a smoothed contour plot showing absolute errors along the slab.'''
    errors = num.sqrt(num.sum((predictions-labels)**2, axis=1))
    med = num.median(errors)
    vmin = 0.
    vmax = med + 1.5 * num.std(errors)
    s = ax.scatter(predictions.T[0], -predictions.T[2], s=6, c=errors, linewidth=0,
            vmin=vmin, vmax=vmax)
    ax.set_xlabel('N-S')
    ax.set_ylabel('Z')
    plt.gcf().colorbar(s)
    # ax.contourf(predictions, errors)


def plot_predictions_and_labels(predictions, labels, name=None):
    if NPOINTS:
        predictions = predictions[: NPOINTS]
        labels = labels[: NPOINTS]
        logger.warn('limiting number of points in scatter plot to %s' % NPOINTS)
    logger.debug('plot predictions and labels')

    predictions = num.array(predictions)
    labels = num.array(labels)
    
    fig, axs = plt.subplots(2, 2)
    for (px, py, pz), (lx, ly, lz) in zip(predictions, labels):
        # top left
        error_map((px, py), (lx, ly), axs[0][0])
        axs[0][0].set_xlabel('N-S')
        axs[0][0].set_ylabel('E-W')

        # bottom left 
        error_map((px, -pz), (lx, -lz), axs[1][0])
        axs[1][0].set_xlabel('N-S')
        axs[1][0].set_ylabel('Z')

        # top right
        error_map((-pz, py), (-lz, ly), axs[0][1])
        axs[0][1].set_xlabel('Z')
        axs[0][1].set_ylabel('E-W')

    error_contourf(predictions, labels, axs[1][1])
    save_figure(fig, name)



