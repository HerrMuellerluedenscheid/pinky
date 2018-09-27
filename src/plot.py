import matplotlib
MAIN_FONT_SIZE = 10
font = {'font.size': MAIN_FONT_SIZE}
matplotlib.rcParams.update(font)
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as num
import logging
from scipy import stats

logger = logging.getLogger('pinky.plot')

FIG_SIZE = (5., 4.230769230769231)
# FIG_SIZE = (8.5, 11.)
POINT_SIZE = 2.
FIG_SUF = '.pdf'
NPOINTS = 200

logger.debug('setting figsize to: %s x %s' % (FIG_SIZE))

def save_figure(fig, name=None):
    '''Saves figure `fig` if `name` is defined. Closes the figure after
    saving.'''
    if not name:
        return

    name = name + FIG_SUF
    logger.info('saving figure: %s' % name)
    fig.savefig(name)
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
                0, 0, string, size=MAIN_FONT_SIZE-1,
                transform=axs[i].transAxes, bbox=boxstyle)

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
    ax.hist(data, bins=bins, histtype='stepfilled')
    med = num.mean(data)
    ax.axvline(med, linestyle='dashed', color='black', alpha=0.8)
    xlim = 500
    ax.text(0.99, 0.99,
            r'$\mu = %1.1f\pm %1.1f$ m' % (med, num.std(data)),
            size=MAIN_FONT_SIZE-1,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes)
    logger.warn('%s datapoints outside xlim [-1000, 1000]' %
            len(num.where(num.logical_or(xlim>data, -xlim<data)[0])))
    ax.set_xlim([-xlim, xlim])


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(int(100 * y))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def mislocation_hist(predictions, labels, name=None):
    '''Plot statistics on mislocations in 3 dimensions and absolute errors.'''
    predictions = num.array(predictions)
    labels = num.array(labels)
    errors = predictions - labels
    errors_abs = num.sqrt(num.sum(errors**2, axis=1))
    xlim = 500.
    fig = plt.figure(figsize=FIG_SIZE)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(224)
    axs = [[ax1, ax2], [ax3, ax4]]
    bins = num.linspace(-xlim, xlim, 71)
    hist_with_stats(errors.T[0], axs[0][0], bins=bins)
    axs[0][0].set_xlabel('North [m]')
    
    hist_with_stats(errors.T[1], axs[0][1], bins=bins)
    axs[0][1].set_xlabel('East [m]')
    
    hist_with_stats(errors.T[2], axs[1][0], bins=bins)
    axs[1][0].set_xlabel('Depth [m]')

    for ax in [ax1, ax2, ax3]:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[1][1].hist(errors_abs, cumulative=True, bins=71, density=True,
    	histtype='step')

    formatter = FuncFormatter(to_percent)

    # Set the formatter
    axs[1][1].yaxis.set_major_formatter(formatter)

    axs[1][1].set_xlabel('Distance [m]')
    axs[1][1].set_xlim([0, 1000])
    axs[1][1].spines['top'].set_visible(False)
    axs[1][1].spines['right'].set_visible(False)

    fig.suptitle('Deviations from DD catalog')

    add_char_labels(flatten(axs))

    fig.subplots_adjust(
        left=0.08, right=0.96, top=0.88, bottom=0.1, wspace=0.35, hspace=0.35)

    save_figure(fig, name)

    n = len(errors_abs)

    e100 = len(num.where(errors_abs<100.)[0])/n
    e200 = len(num.where(errors_abs<200.)[0])/n

    print('Fraction of solutions with error < 100.', e100)
    print('Fraction of solutions with error < 200.', e200)


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
    s = ax.scatter(predictions.T[0], predictions.T[2], s=6, c=errors, linewidth=0,
            vmin=vmin, vmax=vmax)
    ax.set_xlabel('N-S')
    ax.set_ylabel('Z')
    # colorbar
    cax = inset_axes(ax,
		     width="2%",  # width = 10% of parent_bbox width
		     height="50%",  # height : 50%
		     loc='lower left',
		     bbox_to_anchor=(0., 0.05, 1, 1),
		     bbox_transform=ax.transAxes,
		     borderpad=0,
		     )
    cbar = colorbar(s, cax=cax)
    cbar.ax.tick_params(labelsize=MAIN_FONT_SIZE-2)


def rotate(locations, degrees):
    r = degrees * num.pi / 180.
    rotmat = num.array(((num.cos(r), -num.sin(r), 0.),
          (num.sin(r), num.cos(r), 0.),
          (0., 0., 1.)))
    return num.dot(locations, rotmat.T)


def add_char_labels(axes, chars='abcdefghijklmnopqstuvwxyz'):
    for label, ax in zip(chars, axes):
        ax.text(-0.05, 1.05, '(%s)' % label, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='bottom')


def plot_predictions_and_labels(predictions, labels, name=None):

    if NPOINTS:
        predictions = predictions[: NPOINTS]
        labels = labels[: NPOINTS]
        logger.warn('limiting number of points in scatter plot to %s' % NPOINTS)

    predictions /= 1000.
    labels /= 1000.

    logger.debug('plot predictions and labels')

    predictions = num.array(predictions)
    labels = num.array(labels)
    fig = plt.figure(figsize=(5, 5))

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])
    top_left = fig.add_subplot(gs[0])
    top_right = fig.add_subplot(gs[1])
    bottom_left = fig.add_subplot(gs[2])
    bottom_right = fig.add_subplot(gs[3], sharey=top_right)

    top_right.yaxis.tick_right()
    bottom_right.yaxis.tick_right()
    top_right.yaxis.set_label_position('right')
    bottom_right.yaxis.set_label_position('right')

    max_range = num.max(num.abs(num.min(predictions, axis=0) - \
            num.max(predictions, axis=0)))

    def _update_axis_lim(axis_data):
        dmax = num.max(axis_data)
        dmin = num.min(axis_data)
        r = dmax-dmin
        delta = (max_range - r) / 2.
        return dmin - delta, dmax + delta

    px, py, pz = predictions.T
    lx, ly, lz = labels.T

    error_map((py, px), (ly, lx),  top_left)
    top_left.set_ylabel('N-S [km]')
    top_left.set_xlabel('E-W [km]')
    top_left.set_aspect('equal')
    top_left.set_xlim((-0.5, 1.0))

    # with rotation
    degrees = 12
    predictions = rotate(predictions, degrees=degrees)
    labels = rotate(labels, degrees=degrees)
    px, py, pz = predictions.T
    lx, ly, lz = labels.T

    eshift = 0.5
    error_map((py+eshift, pz), (ly+eshift, lz), bottom_left)
    bottom_left.set_xlabel('E-W (rot.) [km]')
    bottom_left.set_ylabel('Depth [km]')
    bottom_left.set_aspect('equal')
    bottom_left.set_xlim((-0.5, 0.5))
    bottom_left.invert_yaxis()

    error_map((px, pz), (lx, lz), bottom_right)
    bottom_right.set_xlabel('N-S (rot.) [km]')
    bottom_right.set_ylabel('Depth [km]')
    bottom_right.invert_yaxis()

    error_contourf(predictions, labels, top_right)
    top_right.set_ylabel('Depth [km]')
    top_right.set_xlabel('N-S (rot.) [km]')

    fig.subplots_adjust(
        left=0.097, right=0.87, top=0.95, bottom=0.1, wspace=0.02, hspace=0.35)

    add_char_labels([top_left, top_right, bottom_left, bottom_right])

    save_figure(fig, name)


def plot_predictions_and_labels_automatic(predictions, labels, name=None):

    if NPOINTS:
        predictions = predictions[: NPOINTS]
        labels = labels[: NPOINTS]
        logger.warn('limiting number of points in scatter plot to %s' % NPOINTS)
    logger.debug('plot predictions and labels')

    predictions = num.array(predictions)
    labels = num.array(labels)
    fig = plt.figure(figsize=FIG_SIZE)

    top_left = fig.add_subplot(2, 2, 1)
    top_right = fig.add_subplot(2, 2, 2, sharey=top_left)
    bottom_left = fig.add_subplot(2, 2, 3, sharex=top_left)
    bottom_right = fig.add_subplot(2, 2, 4)

    max_range = num.max(num.abs(num.min(predictions, axis=0) - \
            num.max(predictions, axis=0)))

    def _update_axis_lim(axis_data):
        dmax = num.max(axis_data)
        dmin = num.min(axis_data)
        r = dmax-dmin
        delta = (max_range - r) / 2.
        return dmin - delta, dmax + delta

    for (px, py, pz), (lx, ly, lz) in zip(predictions, labels):
        error_map((px, py), (lx, ly), top_left)
        top_left.set_xlabel('N-S')
        top_left.set_ylabel('E-W')
        # top_left.set_xlim(*_update_axis_lim(px))
        # top_left.set_ylim(*_update_axis_lim(py))
        top_left.set_xlim(*_update_axis_lim(px+lx))
        top_left.set_ylim(*_update_axis_lim(py+ly))

        error_map((-pz, py), (-lz, ly), top_right)
        top_right.set_xlabel('Z')
        top_right.set_ylabel('E-W')
        top_right.set_xlim(*_update_axis_lim(-pz))
        top_right.set_ylim(*_update_axis_lim(py))

        error_map((px, -pz), (lx, -lz), bottom_left)
        bottom_left.set_xlabel('N-S')
        bottom_left.set_ylabel('Z')
        bottom_left.set_xlim(*_update_axis_lim(px))
        bottom_left.set_ylim(*_update_axis_lim(-pz))

    error_contourf(predictions, labels, bottom_right)
    save_figure(fig, name)


def evaluate_errors(all_predictions, labels, name=None):
    '''
    first set of predictions in `all_predictions` is expected to be the
    predictions of the network.
    '''
    errors_true = num.sqrt(num.sum((all_predictions[0]-labels)**2, axis=1))
    errors_from_prediction = num.sqrt(num.sum((all_predictions[1:]-all_predictions[0])**2,
        axis=2))
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    max_error = num.max(errors_true)
    max_error_prediction = num.max(errors_from_prediction)
    # errors_true /= max_error
    # errors_from_prediction /= max_error
    std_error = num.std(errors_from_prediction, axis=0)
    ax.scatter(errors_true, std_error, alpha=0.9,
            s=POINT_SIZE)
    ax.set_xlabel('deviation from catalog [m]')
    ax.set_ylabel('$\mu(X_i)$')
    save_figure(fig, name)

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.scatter(errors_true, std_error, alpha=0.9,
            s=POINT_SIZE)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('deviation from catalog [m]')
    ax.set_ylabel('$\mu(X_i)$')
    save_figure(fig, name+'_log')

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.scatter(errors_true, std_error, alpha=0.9,
            s=POINT_SIZE)
    ax.set_xscale('log')
    ax.set_xlabel('deviation from catalog [m]')
    ax.set_ylabel('$\mu(X_i)$')
    save_figure(fig, name+'_semi_log')

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.scatter(errors_true/max_error, std_error/num.max(std_error), alpha=0.9,
            s=POINT_SIZE)
    # ax.set_ylim((1E-2, 0))
    # ax.set_xlim((1E-2, 0))
    ax.set_aspect('equal')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('deviation from catalog [m]')
    ax.set_ylabel('$\mu(X_i)$')
    save_figure(fig, name+'_loglog_norm')
