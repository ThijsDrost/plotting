from __future__ import annotations

from typing import Sequence
import math
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from ttools.ttools.itertools import product3
from ttools.ttools.sort import sort_by


def _linestyles_maker():
    dash = (4, 1.5)
    dot = (1, 1.5)
    return [(0, x) for x in (
        (),
        (*dot,),
        (*dash,),
        (*dash, *dot),
        (*dash, *dot, *dot),
        (*dash, *dot, *dot, *dot),
        (*dash, *dash, *dot),
        (*dash, *dash, *dot, *dot),
    )]
    # return ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 3, 1, 1, 1))]


_linestyles = _linestyles_maker()
_marker_list = ['o', 's', 'v', 'D', '^', '<', '>', 'p', '*', 'H', 'd', 'P', 'X']


def marker_cycler(index, markers=None):
    if markers is None:
        markers = _marker_list
    return markers[index % len(markers)]


def color_cycler(index, colors=None):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return colors[index % len(colors)]


def linestyle_cycler(index, linestyles=None):
    if linestyles is None:
        linestyles = _linestyles
    return linestyles[index % len(linestyles)]


def _set_cycler_value(value, default, name):
    if value is True:
        return default
    elif (value is False) or (value is None):
        return ['none']
    elif isinstance(value, (str, tuple)):
        return [value]
    elif isinstance(value, dict):  # maybe remove this
        return value
    else:
        try:
            value[0]
        except TypeError as e:
            raise IndexError(f'{name} seems not to be indexable') from e
    return value


def linelook_by(values: Sequence, *, markers: Sequence | bool | str = False, linestyles: Sequence | bool | str = False,
                colors: Sequence | bool | str = False):
    """
    Returns a list of dictionaries with the markers, linestyles, and colors for each value in `values`. If a value or a sequence
    of values is given for `markers`, `linestyles`, or `colors`, the values are used. If `True` is given, the default cyclers are
    used.

    Parameters
    ----------
    values
        The values to get the linelook for, each value should be hashable. The order of the values is preserved in the output.
        Same values will have the same linelook.
    markers
    linestyles
    colors

    Returns
    -------
    list[dict]

    Notes
    -----
    A string or a tuple value for `markers`, `linestyles`, or `colors` are interpreted as a single value. All other values are
    interpreted as a sequence of values.

    The marker, linestyle, and color are varied simultaneously. If the number of values is greater than the number of markers,
    linestyles, or colors, the values are repeated.

    """
    markers = _set_cycler_value(markers, _marker_list, 'markers')
    if isinstance(markers, dict):
        raise ValueError('Markers should not be a dict.')
    linestyles = _set_cycler_value(linestyles, _linestyles, 'linestyles')
    if isinstance(linestyles, dict):
        raise ValueError('Linestyles should not be a dict.')
    colors = _set_cycler_value(colors, plt.rcParams['axes.prop_cycle'].by_key()['color'], 'colors')
    if isinstance(colors, dict):
        raise ValueError('Colors should not be a dict.')

    if math.lcm(len(markers), len(linestyles), len(colors)) < len(values):
        warnings.warn('Values will have non unique linelooks.')

    given_values = {}
    iterator = product3(markers, linestyles, colors)
    for value in values:
        if value not in given_values:
            try:
                vals = next(iterator)
            except StopIteration:
                iterator = product3(markers, linestyles, colors)
                vals = next(iterator)

            given_values[value] = {}
            if vals[0] is not None:
                given_values[value]['marker'] = vals[0]
            if vals[1] is not None:
                given_values[value]['linestyle'] = vals[1]
            if vals[2] is not None:
                given_values[value]['color'] = vals[2]

    return [given_values[value].copy() for value in values]


def linelooks_by(*, color_values: Sequence = None, linestyle_values: Sequence = None, marker_values: Sequence = None,
                 markers: Sequence | bool | str | dict = True, linestyles: Sequence | bool | str | dict = True,
                 colors: Sequence | bool | str | dict = True) -> list[dict[str, any]]:
    """
    Returns a list of dictionaries with the markers, linestyles, and colors for each value in `values`. If a value or a sequence
    of values is given for `markers`, `linestyles`, or `colors`, the values are used. If `True` is given, the default cyclers are
    used.
    """
    def linelooks(values, cycle, name):
        if isinstance(cycle, dict):
            try:
                return [cycle[value] for value in values]
            except KeyError as e:
                for value in values:
                    if value not in cycle:
                        raise KeyError(f'Label {value} not found in {name} dict.') from e
                raise e  # should be unreachable

        given_values = {}
        for value in values:
            if value not in given_values:
                given_values[value] = cycle[len(given_values) % len(cycle)]
        if len(given_values) > len(cycle):
            warnings.warn(f'Lines will have non unique {name}.')
        return [given_values[value] for value in values]

    length = -1
    if color_values is not None:
        length = len(color_values)
    if linestyle_values is not None:
        if length != -1:
            if length != len(linestyle_values):
                raise ValueError('The lengths of the values are not the same.')
        else:
            length = len(linestyle_values)
    if marker_values is not None:
        if length != -1:
            if length != len(marker_values):
                raise ValueError('The lengths of the values are not the same.')
        else:
            length = len(marker_values)

    if length == -1:
        raise ValueError('No values given.')

    markers = _set_cycler_value(markers, _marker_list, 'markers')
    linestyles = _set_cycler_value(linestyles, _linestyles, 'linestyles')
    colors = _set_cycler_value(colors, plt.rcParams['axes.prop_cycle'].by_key()['color'], 'colors')

    result = [{} for _ in range(length)]
    if color_values is not None:
        c_val = linelooks(color_values, colors, 'color')
        for i in range(length):
            result[i]['color'] = c_val[i]
    if linestyle_values is not None:
        l_val = linelooks(linestyle_values, linestyles, 'linestyle')
        for i in range(length):
            result[i]['linestyle'] = l_val[i]
    if marker_values is not None:
        m_val = linelooks(marker_values, markers, 'marker')
        for i in range(length):
            result[i]['marker'] = m_val[i]

    return result


def legend_linelooks_combines(labels, title=None, *, colors=None, linestyles=None, markers=None, sort=True):
    def set_value(value, default, name):
        if (value is None) or (value is False):
            value = ['none']*len(labels)
        elif value is True:
            value = [default[i % len(default)] for i in range(len(labels))]
            if len(labels) > len(default):
                warnings.warn(f'Values will have non unique {name}.')
        elif len(value) == 1:
            value = value*len(labels)
        elif len(value) != len(labels):
            raise ValueError(rf'The number of {name} should be the same as the number of labels.')
        return value

    markers = set_value(markers, _marker_list, 'markers')
    linestyles = set_value(linestyles, _linestyles, 'linestyles')
    colors = set_value(colors, plt.rcParams['axes.prop_cycle'].by_key()['color'], 'colors')

    line_kwargs_iter = [{'color': c, 'linestyle': l, 'marker': m} for c, l, m in zip(colors, linestyles, markers)]

    line_handles = []
    label_values = []
    if title is not None:
        line_handles.append(Patch(visible=False))
        title = r'$\bf{' + title.replace(' ', r'}$ $\bf{') + r'}$'
        label_values.append(title)

    try:
        sorter = [float(i) for i in labels]
    except ValueError:
        sorter = labels

    temp_line_handles = []
    temp_labels = []
    for label, kwargs in zip(labels, line_kwargs_iter):
        temp_line_handles.append(plt.Line2D([], [], label=label, **kwargs))
        temp_labels.append(label)

    label, line = sort_by(sorter, temp_labels, temp_line_handles) if sort else (temp_labels, temp_line_handles)
    line_handles.extend(line)
    label_values.extend(label)

    return line_kwargs_iter, {'handles': line_handles, 'labels': label_values}


def legend_linelooks_by(color_labels: Sequence[str|float|int] = None, linestyle_labels: Sequence[str|float|int] = None,
                        marker_labels: Sequence[str|float|int] = None, color_values: Sequence = None, linestyle_values: Sequence = None,
                        marker_values: Sequence = None, no_color='k', no_marker=True, no_linestyle=True,
                        color_title=None, marker_title=None, linestyle_title=None, sort=True):
    # make the line_kwargs_iter
    def make_values(labels, values, name, default):
        if labels is None:
            labels = [None]
            values = [None]
        else:
            if (values is None) or (values is True):
                unique_labels = list(set(labels))
                value_dict = {unique_labels[i]: default[i % len(default)] for i in range(len(unique_labels))}
                values = [deepcopy(value_dict[label]) for label in labels]
            if isinstance(values, dict):
                try:
                    values = [deepcopy(values[label]) for label in labels]
                except KeyError as e:
                    for label in labels:
                        if label not in values:
                            raise KeyError(f'Label {label} not found in {name} dict.') from e
                    raise e
            if not isinstance(values, (Sequence, np.ndarray)):
                raise TypeError(f'Values should be a `Sequence`, `dict`, or `None`, not {type(values)}.')
            if len(labels) != len(values):
                raise ValueError(f'The number of {name} labels should be the same as the number of {name} values, not {len(labels)} and {len(values)}.')
        return labels, values

    color_labels, color_values = make_values(color_labels, color_values, 'color', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    linestyle_labels, linestyle_values = make_values(linestyle_labels, linestyle_values, 'linestyle', _linestyles)
    marker_labels, marker_values = make_values(marker_labels, marker_values, 'marker', _marker_list)
    # color_labels = color_labels or [None]
    # color_values = color_values or [None]
    # if len(color_labels) != len(color_values):
    #     raise ValueError('The number of color labels should be the same as the number of color values.')

    # linestyle_labels = linestyle_labels or [None]
    # linestyle_values = linestyle_values or [None]
    # if len(linestyle_labels) != len(linestyle_values):
    #     raise ValueError('The number of linestyle labels should be the same as the number of linestyle values.')
    #
    # marker_labels = marker_labels or [None]
    # marker_values = marker_values or [None]
    # if len(marker_labels) != len(marker_values):
    #     raise ValueError('The number of marker labels should be the same as the number of marker values.')

    line_kwargs_iter = []
    c_vals = []
    l_vals = []
    m_vals = []
    for color_label, color_value in zip(color_labels, color_values):
        for linestyle_label, linestyle_value in zip(linestyle_labels, linestyle_values):
            for marker_label, marker_value in zip(marker_labels, marker_values):
                line_kwargs_iter.append({})
                if color_label is not None:
                    line_kwargs_iter[-1]['color'] = color_value
                    c_vals.append(color_label)
                if linestyle_label is not None:
                    line_kwargs_iter[-1]['linestyle'] = linestyle_value
                    l_vals.append(linestyle_label)
                if marker_label is not None:
                    line_kwargs_iter[-1]['marker'] = marker_value
                    m_vals.append(marker_label)

    c_vals = c_vals or None
    l_vals = l_vals or None
    m_vals = m_vals or None
    return line_kwargs_iter, legend_linelooks(line_kwargs_iter, color_labels=c_vals, linestyle_labels=l_vals, marker_labels=m_vals,
                no_color=no_color, no_marker=no_marker, no_linestyle=no_linestyle, color_title=color_title,
                marker_title=marker_title, linestyle_title=linestyle_title, sort=sort)


def legend_linelooks(line_kwargs_iter, /, *, color_labels: Sequence[str|float|int] = None, linestyle_labels: Sequence[str|float|int] = None,
                     marker_labels: Sequence[str|float|int] = None, no_color='k', no_marker=True, no_linestyle=True,
                     color_title=None, marker_title=None, linestyle_title=None, sort=True)\
        -> dict[str, list[plt.Line2D | LegendHandle] | list[str]]:
    line_handles = []
    label_values = []

    no_marker = no_marker or None
    no_linestyle = no_linestyle or None

    if not line_kwargs_iter:
        raise ValueError('No line_kwargs given.')

    def make(labels, name, no_marker, no_linestyle) -> tuple[list[plt.Line2D], list[str]]:
        temp_line_handles = []
        temp_labels = []
        been = {}
        if len(labels) != len(line_kwargs_iter):
            raise ValueError(f'The number of {name} labels should be the same as the number of line_kwargs.')

        for line_kwargs, label in zip(line_kwargs_iter, labels):
            if label in been:
                if isinstance(been[label], np.ndarray):
                    if not np.array_equal(been[label], line_kwargs.get(name, 'none')):
                        raise ValueError(f'The same label has different {name}.')
                elif been[label] != line_kwargs.get(name, 'none'):
                    raise ValueError(f'The same label has different {name}.')
            else:
                this_no_marker = None
                this_no_linestyle = None
                if no_marker:
                    if 'marker' in line_kwargs:
                        if no_marker is True:
                            this_no_marker = 'o'
                        else:
                            this_no_marker = no_marker
                if no_linestyle:
                    if 'linestyle' in line_kwargs:
                        if no_linestyle is True:
                            this_no_linestyle = '-'
                        else:
                            this_no_linestyle = no_linestyle
                this_no_marker = this_no_marker if this_no_marker is not None else 'none'
                this_no_linestyle = this_no_linestyle if this_no_linestyle is not None else 'none'

                been[label] = line_kwargs.get(name, 'none')

                if name in ('linestyle', 'marker'):
                    kwargs = {'color': no_color}
                elif name == 'color':
                    kwargs = {'linestyle': this_no_linestyle, 'marker': this_no_marker}
                else:
                    raise NotImplementedError(f'Unknown name: {name}')

                kwargs[name] = line_kwargs.get(name, 'none')
                temp_line_handles.append(plt.Line2D([], [], label=label, **kwargs))
                temp_labels.append(label)
        return temp_line_handles, temp_labels

    for (labels, name, title) in ((color_labels, 'color', color_title),
                                  (linestyle_labels, 'linestyle', linestyle_title),
                                  (marker_labels, 'marker', marker_title)):
        if labels is not None:
            if title is not None:
                line_handles.append(Patch(visible=False))
                title = r'$\bf{' + title.replace(' ', r'}$ $\bf{') + r'}$'
                label_values.append(title)
            line, label = make(labels, name, no_marker, no_linestyle)

            try:
                sorter = [float(i) for i in label]
            except ValueError:
                sorter = label

            label, line = sort_by(sorter, label, line) if sort else (label, line)
            line_handles.extend(line)
            label_values.extend(label)

    return {'handles': line_handles, 'labels': label_values}


class LegendHandle(plt.Line2D):
    pass
