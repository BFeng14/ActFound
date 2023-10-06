import numpy as np
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from collections import defaultdict
import pandas as pd

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.markers import MarkerStyle
import matplotlib as mpl

import plot_settings


def format_ax(ax):
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def format_legend(fig_legend, handles, labels, legendmarker=20, loc='center', ncols=1):
    fig_legend.legend(handles, labels, loc=loc, scatterpoints=1, ncol=ncols,
                      frameon=False, markerscale=legendmarker)


def put_legend_outside_plot(ax, anchorage=(1.1, 1.05)):
    ax.legend(bbox_to_anchor=anchorage)


def align_axes_ticks(ax, use_y=True, ticks_to_use=None):
    if use_y:
        yticks = ax.get_yticks()
        print(yticks)
        ax.set_xticks(yticks)
    elif ticks_to_use is None:  # use xticks
        xticks = ax.get_xticks()
        ax.set_yticks(xticks)
    else:
        ax.set_xticks(ticks_to_use)
        ax.set_yticks(ticks_to_use)


def scatter_plot(ax, xs, ys, xlabel, ylabel, xscale='linear', yscale='linear',
                 invert_axes=False, color='tab:blue', alpha=1, size=5, style=MarkerStyle('o'),
                 label=None, edge_color=None):
    for i, x in enumerate(xs):
        if x != x:  # got a nan -> set to min / max value
            xs[i] = 1
    for i, y in enumerate(ys):
        if y != y:  # got a nan -> set to min / max value
            ys[i] = 1

    if edge_color is None:
        edge_color = color
    scatter = ax.scatter(xs, ys, s=size, c=color, alpha=alpha, marker=style, label=label, edgecolors=edge_color)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if invert_axes:
        ax.invert_xaxis()
        ax.invert_yaxis()
    return scatter


def show_x_equals_y(ax, domain=(0, 1), color='tab:orange', set_max=1.0):
    new_min, new_max = min(*domain, 1e-2), max(*domain, 1e-2)

    line = np.arange(new_min, new_max, 0.01)
    ax.plot(line, line, c=color, linestyle='dashed')


def line_plot(ax, ydata, xlabel, ylabel, xdata=None, xscale='linear', yscale='linear',
              max_time=None, invert_axes=False, color='tab:blue', linestyle='solid', label_marker=None, linewidth=1):
    if xdata is None:
        xdata = range(len(ydata))
    ax.plot(xdata, ydata, c=color, linestyle=linestyle, label=label_marker, linewidth=linewidth)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if invert_axes:
        ax.invert_xaxis()
        ax.invert_yaxis()


def horizontal_line(ax, yval, linestyle='solid', linewidth=1, color='tab:blue'):
    ax.axhline(y=yval, linestyle=linestyle, linewidth=linewidth, c=color)


def bar_plot(ax, data, data_labels, xlabel, ylabel, xscale='linear', yscale='linear',
             min_val=0, invert_axes=False, color='tab:blue', errs=None, edge_color=None,
             rotangle=0, anchor='center'):
    if invert_axes:
        ax.bar(x=range(len(data)), height=[min_val - d for d in data], bottom=data, color=color,
               yerr=errs, edgecolor=color if edge_color is None else edge_color, ecolor='k')
        ax.invert_yaxis()
    else:
        ax.bar(x=range(len(data)), height=[d - min_val for d in data], bottom=min_val, color=color, yerr=errs,
               ecolor='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xticks(range(len(data_labels)))
    ax.set_xticklabels(data_labels, rotation=rotangle, ha=anchor)


def horizontal_bar_plot(ax, data, data_labels, xlabel, ylabel, xscale='linear', yscale='linear',
                        min_val=0, invert_axes=False, color='tab:blue', edge_color=None):
    if invert_axes:
        ax.barh(y=range(len(data)), width=[min_val - d for d in data], left=data, color=color,
                edgecolor=color if edge_color is None else edge_color)
        ax.invert_xaxis()
    else:
        ax.barh(y=range(len(data)), width=data, left=min_val, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_yticks(range(len(data_labels)))
    ax.set_yticklabels(data_labels)


def grouped_barplot(ax, nested_data, data_labels, xlabel, ylabel, xscale='linear', yscale='linear',
                    min_val=0, invert_axes=False, nested_color='tab:blue', color_legend=None, nested_errs=None,
                    tickloc_top=True, rotangle=45, legend_loc='upper right', anchorpoint='right', scale=1.5):
    xs = [scale * (i + j * 0.9 / len(nested_data[0]) - 0.5) for i in range(len(nested_data)) for j in
          range(len(nested_data[0]))]
    # print(xs)
    heights = []
    bottoms = []
    colors = []
    labels = []
    errs = []
    for idx, item in enumerate(nested_data):
        heights.extend([it - min_val if not invert_axes else min_val - it for it in item])
        bottoms.extend([min_val if not invert_axes else it for it in item])
        colors.extend(nested_color if not isinstance(nested_color, str) else [nested_color for _ in range(len(item))])
        if nested_errs is not None:
            errs.extend(nested_errs[idx])
        if color_legend is not None:
            labels.extend(color_legend)

    for it in range(len(nested_data[0])):
        ax.bar(x=[xs[i] for i in range(it, len(xs), len(nested_data[0]))],
               height=[heights[i] for i in range(it, len(heights), len(nested_data[0]))],
               bottom=[bottoms[i] for i in range(it, len(heights), len(nested_data[0]))],
               color=[colors[i] for i in range(it, len(colors), len(nested_data[0]))],
               width=scale * 0.9 / len(nested_data[0]), label=labels[it] if color_legend is not None else '',
               align='edge', ecolor='k',
               yerr=None if nested_errs is None else [errs[i] for i in range(it, len(errs), len(nested_data[0]))])
    if invert_axes:
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xticks([scale * i for i in range(len(data_labels))])
    ax.set_xticklabels(data_labels)
    ax.tick_params(top=tickloc_top, bottom=not tickloc_top,
                   labeltop=tickloc_top, labelbottom=not tickloc_top)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=rotangle, ha=anchorpoint,
             rotation_mode="anchor")
    if color_legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        format_legend(plt, handles, labels, loc=legend_loc)


def show_image(ax, data, xlabel, ylabel, aspect=None, cmap='bwr', xticks=[], yticks=[]):
    c = ax.imshow(data, aspect=aspect, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    return c


def histogram(ax, data, xlabel, ylabel, color=None, rotangle=None, check_upper=True, N_bins=10,
              xscale='linear', yscale='linear', edge_color=None, label=None):
    N, bins, patches = plt.hist(data, bins=N_bins, color=color, rwidth=1,
                                edgecolor=color if edge_color is None else edge_color, label=None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def sorted_histogram(ax, data, sorted_data, xlabel, ylabel, yscale='linear', call_out_labels=None,
                     base_color=None, call_out_color=None, rotangle=None, check_upper=True, add_padding=0,
                     extra_name_translation=None, edge_color=None, edge_width=1, override_equals=False, anchor='right'):
    N, bins, patches = plt.hist(sorted([c for c in data], key=lambda x: sorted_data.index(x)), bins=len(sorted_data),
                                color=base_color, rwidth=1,
                                edgecolor=base_color if edge_color is None else edge_color, linewidth=edge_width)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if call_out_labels is None:
        ax.set_xticks([i + 0.5 for i in range(len(sorted_data))])
        ax.set_xticklabels(sorted_data, rotation=rotangle)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        for x in call_out_labels:
            xcheck = x.upper() if check_upper else x
            ibase = sorted_data.index(xcheck)
            lenx = len([m for m in data if m == xcheck])
            for i in range(ibase - add_padding, ibase + 1 + add_padding):
                # make sure it's the same first!
                icheck = sorted_data[i].upper() if check_upper else sorted_data[i]
                len_i = len([n for n in data if n == icheck])
                if not override_equals and len_i != lenx:
                    continue
                patches[i].set_facecolor(call_out_color)
        ax.set_xticks([sorted_data.index(x.upper() if check_upper else x) + 0.5 for x in call_out_labels])
        ax.set_xticklabels([extra_name_translation[x] if extra_name_translation is not None else x
                            for x in call_out_labels], rotation=rotangle, ha=anchor)


def box_plot(ax, data, xlabel, ylabel, xticks=None, yticks=None, xticklabels=None, yticklabels=None,
             xscale='linear', yscale='linear', box_colors=None, alpha=1, widths=1.0, zorder=0, positions=None):
    bplot = ax.boxplot(
        data, patch_artist=box_colors is not None, widths=widths, zorder=zorder,
        positions=positions)

    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for idx, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(box_colors[idx])
        patch.set_alpha(alpha)


def violin_plot(ax, data, xlabel, ylabel, xticks=None, yticks=None, xscale='linear', yscale='linear',
                violin_color=None, violin_line_color=None, alpha=1.0):
    violins = ax.violinplot(data)
    for pc in violins['bodies']:
        pc.set_color(violin_color)
        pc.set_alpha(alpha)
    for partname in ['cbars', 'cmins', 'cmaxes']:
        vp = violins[partname]
        vp.set_edgecolor(violin_line_color)
        vp.set_linewidth(1)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(range(1, len(xticks) + 1))
        ax.set_xticklabels(xticks)
    if yticks is not None:
        ax.set_yticks(range(1, len(yticks) + 1))
        ax.set_yticklabels(yticks)


def kaplan_meier_curve(ax, times, observations, labels, xlabel, ylabel, showCI=True,
                       max_time=None, usePercentageX=False, usePercentageY=True, colors=None):
    kmf1 = KaplanMeierFitter()  # instantiate the class to create an object

    if type(colors) == str:
        colors = [colors for _ in range(len(times))]  # all the same color
    for cohort in range(len(times)):
        dt = np.nan_to_num(times[cohort])
        kmf1.fit(dt, np.nan_to_num(observations[cohort]))
        kmf1.plot(ax=ax, label=labels[cohort], ci_show=showCI, color=colors[cohort] if colors is not None else None)

    if usePercentageX:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(
            xmax=1, decimals=None, symbol='%', is_latex=False))
    if usePercentageY:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(
            xmax=1, decimals=None, symbol='%', is_latex=False))

    if max_time is not None:
        ax.set_xlim(ax.get_xlim()[0], max_time)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    format_legend(plt, handles, labels, loc='lower left')
