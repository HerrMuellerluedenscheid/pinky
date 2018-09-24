import matplotlib
font = {'font.size': 12}
matplotlib.rcParams.update(font)

import matplotlib.pyplot as plt
import math
import numpy as num
import logging
from scipy import stats

logger = logging.getLogger('pinky.plot')

FIG_SIZE = (8.5/2., 11./3.)
POINT_SIZE = 2.
FIG_SUF = '.pdf'
NPOINTS = 110

logger.debug('setting figsize to: %s x %s' % (FIG_SIZE))

def save_figure(fig, name=None):
    '''Saves figure `fig` if `name` is defined. Closes the figure after
    saving.'''
    if not name:
        return

    print('saving figure: %s' % name)
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
        left=0.14, right=0.98, top=0.98, bottom=0.15, wspace=0.01, hspace=0.01)


def get_notleft_axs(axs_grid):
    '''Returns a list of left most axes objects from 2 dimensional grid.'''
    x = []
    for axs in axs_grid:
        x.extend(axs[1:])
    return x


def get_left_axs(axs_grid):
    '''Returns a list of left most axes objects from 2 dimensional grid.'''
    return [ax[0] for ax in axs_grid]


def get_bottom_axs(axs_grid):
    '''Returns a list of bottom most axes objects from 2 dimensional grid.'''
    return axs_grid[-1]


def get_notbottom_axs(axs_grid):
    '''Returns a list of every but bottom most axes objects from 2 dimensional grid.'''
    if len(axs_grid) > 1:
        return axs_grid[:-1][0]
    else:
        return []


def plot_locations(locations, color, title, axs=None):
    fig = None
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE)

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


def show_data(model, n=9, nskip=0, shuffle=False):
    '''Plot 2 dimensional feature images and waveform sections.

    :param model `pinky.model.Model` instance:
    :param n: number of plots to produce
    :param shuffle: if `True` randomly select the `n` samples'''

    yscale = 2.  # Use this to tune amplitudes of waveform plots
    n_rows = int(max(num.sqrt(n), 1))
    boxstyle = dict(boxstyle='round', facecolor='white', alpha=0.7)
    fig, axs_grid = plt.subplots(math.ceil(n/n_rows), n_rows, figsize=FIG_SIZE,
            squeeze=False)

    debug = logger.getEffectiveLevel() == logging.DEBUG

    axs = flatten(axs_grid)

    fig_w, axs_w_grid = plt.subplots(math.ceil(n/n_rows), n_rows,
            figsize=FIG_SIZE, squeeze=False)
    axs_w = flatten(axs_w_grid)

    model.config.data_generator.shuffle = shuffle
    for i, (chunk, label) in enumerate(
            model.config.data_generator.generate()):

        if i<nskip:
            continue
        elif i == n+nskip:
            break

        i -= nskip
        axs[i].imshow(chunk, aspect='auto', cmap='gist_gray', origin='lower')
        string = ' '.join([' %1.2f |'% l for l in label])
        string += '\nminmax= %1.1f| %1.1f' %(num.nanmin(chunk), num.nanmax(chunk))
        if debug:
            axs[i].text(
                0, 0, string, transform=axs[i].transAxes, bbox=boxstyle)

        _, n_samples = chunk.shape
        xdata = num.arange(n_samples)

        for irow, row in enumerate(chunk):
            row -= num.mean(row)
            axs_w[i].plot(xdata, irow+yscale*row, color='black', linewidth=0.5)

    [clear_ax(ax) for ax in axs_w]

    labels_eval = list(model.config.evaluation_data_generator.iter_labels())
    labels_train = list(model.config.data_generator.iter_labels())

    locs = []
    labels = []
    for nslc, i in model.config.data_generator.nslc_to_index.items():
        locs.append(i)
        labels.append('.'.join(nslc))

    for axs in (get_left_axs(x) for x in (axs_grid, axs_w_grid)):
        for ax in axs:
            ax.set_yticks(locs)
            ax.set_yticklabels(labels)

    for axs in (get_notleft_axs(x) for x in (axs_grid, axs_w_grid)):
        for ax in axs:
            ax.set_yticks([])

    for axs in (get_bottom_axs(x) for x in (axs_grid, axs_w_grid)):
        for ax in axs:
            ax.set_xlabel('Sample')

    for axs in (get_notbottom_axs(x) for x in (axs_grid, axs_w_grid)):
        for ax in axs:
            ax.set_xticks([])

    adjust(fig)
    adjust(fig_w)

    fig_labels, axs_labels = plot_locations(
            labels_eval, 'red', title='eval')

    fig_labels, axs_labels = plot_locations(
            labels_train, 'blue', title='train', axs=axs_labels)

    save_figure(fig_labels, 'pinky_labels')
    save_figure(fig, 'pinky_features')
    save_figure(fig_w, 'pinky_waves')


def show_kernels_dense(weights, name=None):
    '''2 dimensional images of dense weights.'''
    fig, axs = plt.subplots(1, 1, figsize=FIG_SIZE)

    axs.imshow(weights, cmap='gray')
    axs.axis('off')
    axs.set_yticks([])
    axs.set_xticks([])

    save_figure(fig, name)


def show_kernels(weights, name=None):
    n_columns = 8
    n_weights = weights.shape[-1]
    n_rows = int(n_weights // n_columns)
    fig, axs = plt.subplots(n_rows, n_columns, figsize=FIG_SIZE)

    axs = [ax for iax in axs for ax in iax]
    
    for iweight in range(n_weights):
        axs[iweight].imshow(weights[..., iweight], cmap='gray')
    
    for ax in axs:
        ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

    save_figure(fig, name)


def confidence(data, rate=0.95):
    return stats.t.interval(
            rate, len(data)-1, loc=num.mean(data), scale=stats.sem(data))


def hist_with_stats(data, ax, bins=31):
    '''Plot a histogram of `data` into `ax` and label median and errors.'''
    ax.hist(data, bins=bins)
    med = num.mean(data)
    ax.axvline(med, color='black')
    xlim = 1000
    ax.text(0.1, 0.99,
            r'$\mu = %1.1f\pm %1.1f$' % (med, num.std(data)),
            horizontalalignment='left',
            transform=ax.transAxes)
    logger.warn('%s datapoints outside xlim [-1000, 1000]' %
            len(num.where(num.logical_or(xlim>data, -xlim<data)[0])))
    ax.set_xlim([-xlim, xlim])


def mislocation_hist(predictions, labels, name=None):
    '''Plot statistics on mislocations in 3 dimensions and absolute errors.'''
    predictions = num.array(predictions)
    labels = num.array(labels)
    errors = predictions - labels
    errors_abs = num.sqrt(num.sum(errors**2, axis=1))
    xlim = 1000.
    fig = plt.figure(figsize=FIG_SIZE)
    ax1 = fig.add_subplot(221, figsize=FIG_SIZE)
    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1, figsize=FIG_SIZE)
    ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1, figsize=FIG_SIZE)
    ax4 = fig.add_subplot(224, figsize=FIG_SIZE)
    axs = [[ax1, ax2], [ax3, ax4]]
    bins = num.linspace(-xlim, xlim, 71)
    hist_with_stats(errors.T[0], axs[0][0], bins=bins)
    axs[0][0].set_xlabel('Error (North) [m]')
    
    hist_with_stats(errors.T[1], axs[0][1], bins=bins)
    axs[0][1].set_xlabel('Error (East) [m]')
    
    hist_with_stats(errors.T[2], axs[1][0], bins=bins)
    axs[1][0].set_xlabel('Error (Depth) [m]')

    for ax in [ax1, ax2, ax3]:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[1][1].hist(errors_abs, cumulative=True, bins=71, density=True,
    histtype='step')
    axs[1][1].set_xlabel('Distance [m]')
    axs[1][1].set_xlim([0, 1000])
    axs[1][1].spines['top'].set_visible(False)
    axs[1][1].spines['right'].set_visible(False)

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
    
    fig, axs = plt.subplots(2, 2, figsize=FIG_SIZE)
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

