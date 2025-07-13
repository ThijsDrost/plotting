from __future__ import annotations

from typing import Sequence, Hashable, Any, Callable, Literal
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from .cbar import cbar_norm_colors
from ttools.ttools.itertools import sim_product
from ttools.ttools.sort import sort_together
from checking.checking import Validator


_color_type = str | float | Sequence[float | str | Sequence]
_marker_type = str
_linestyle_type = str | tuple[float, tuple[float, ...]] | Sequence[float | Sequence]


_dash = (4, 1.5)
_dot = (1, 1.5)
_linestyles = [(0, x) for x in (
        (),
        (*_dot,),
        (*_dash,),
        (*_dash, *_dot),
        (*_dash, *_dot, *_dot),
        (*_dash, *_dot, *_dot, *_dot),
        (*_dash, *_dash, *_dot),
        (*_dash, *_dash, *_dot, *_dot),
    )]
_marker_list = ['o', 's', 'v', 'D', '^', '<', '>', 'p', '*', 'H', 'd', 'P', 'X']
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def marker_cycler(index, markers=None):
    """
    Cycles through a list of markers and returns the marker at the given index.

    Parameters
    ----------
    index : int
        The index to retrieve the marker for.
    markers : Sequence[str], optional
        A list of markers to cycle through. If None, a default list of markers is used.

    Returns
    -------
    str
        The marker at the given index.
    """
    if markers is None:
        markers = _marker_list
    return markers[index % len(markers)]


def color_cycler(index: int, colors: Sequence[_color_type] = None) -> _color_type:
    """
    Cycles through a list of colors and returns the color at the given index.

    Parameters
    ----------
    index : int
        The index to retrieve the color for.
    colors
        A list of colors to cycle through. If None, the default matplotlib color cycle is used.

    Returns
    -------
    any
        The color at the given index.
    """
    if colors is None:
        colors = _colors
    return colors[index % len(colors)]


def linestyle_cycler(index, linestyles: Sequence[tuple[float, ...] | str] =None):
    """
    Cycles through a list of linestyles and returns the linestyle at the given index.

    Parameters
    ----------
    index : int
        The index to retrieve the linestyle for.
    linestyles : Sequence[tuple[float, ...] | str], optional
        A list of linestyles to cycle through. If None, a default list of linestyles is used.

    Returns
    -------
    tuple[float, ...] | str
        The linestyle at the given index.
    """
    if linestyles is None:
        linestyles = _linestyles
    return linestyles[index % len(linestyles)]


def _set_cycler_value(value: Sequence | str | tuple | bool | dict | None, default: Sequence, name: str):
    """
    Set the cycler value based on the input `value`. If `value` is `True`, the default cycler is used. If `value` is `False`
    or `None`, a list with a single value 'none' is returned. If `value` is a string or a tuple, it is returned as a single
    value in a list. If `value` is a dictionary or sequence, it is returned as is.

    Parameters
    ----------
    value : Sequence, str, tuple, bool, dict, or None
        The input value to process.
    default : Sequence
        The default value to use if `value` is True.
    name : str
        The name of the value, used for error messages.

    Returns
    -------
    Sequence | dict
        The processed cycler value.

    Raises
    -------
    TypeError
        If `value` is neither a `Sequence`, `str`, `tuple`, `bool`, `dict`, or None.
    """
    if value is True:
        return default
    elif (value is False) or (value is None):
        return ['none']
    elif isinstance(value, (str, tuple)):
        return [value]
    elif isinstance(value, dict):
        return value
    else:
        if not isinstance(value, Sequence):
            raise TypeError(f'{name} should be a `Sequence`, `str`, `tuple`, `bool`, `dict`, or None, not {type(value)}.')
        return value


def linelooks(values: Sequence[Hashable], *, markers: Sequence[str] | bool | str = False,
              linestyles: Sequence | bool | str = False, colors: Sequence | bool | str = False):
    """
    Returns a list of dictionaries with the markers, linestyles, and colors for each value in `values`. If a value or a sequence
    of values is given for `markers`, `linestyles`, or `colors`, the values are used. If `True` is given, the default cyclers are
    used. . The order of the values is preserved in the output. Same values will have the same linelook.

    Parameters
    ----------
    values
        The values to get the linelook for, each value should be hashable.
    markers : Sequence, bool, or str, optional
        Marker values or a flag to use default markers.
    linestyles : Sequence, bool, or str, optional
        Linestyle values or a flag to use default linestyles.
    colors : Sequence, bool, or str, optional
        Color values or a flag to use default colors.


    Returns
    -------
    list[dict]

    Notes
    -----
    A string or a tuple value for `markers`, `linestyles`, or `colors` are interpreted as a single value. All other values are
    interpreted as a sequence of values.

    The marker, linestyle, and color are varied simultaneously. If the number of values is greater than the product of
    the number of markers, linestyles, and colors, combinations are repeated.
    """
    is_sequence = Validator.is_sequence()

    markers = _set_cycler_value(markers, _marker_list, 'markers')
    is_sequence(markers, "markers")

    linestyles = _set_cycler_value(linestyles, _linestyles, 'linestyles')
    is_sequence(linestyles, "linestyles")

    colors = _set_cycler_value(colors, _colors, 'colors')
    is_sequence(colors, "colors")

    if len(markers)*len(linestyles)*len(colors) < len(values):
        warnings.warn('Values will have non unique linelooks.')

    given_values = {}
    iterator = sim_product(markers, linestyles, colors, stop=False)
    for value in values:
        if value not in given_values:
            vals = next(iterator)

            given_values[value] = {}
            if vals[0] is not None:
                given_values[value]['marker'] = vals[0]
            if vals[1] is not None:
                given_values[value]['linestyle'] = vals[1]
            if vals[2] is not None:
                given_values[value]['color'] = vals[2]

    return [given_values[value].copy() for value in values]


def linelooks_sections(*, color_values: Sequence = None, linestyle_values: Sequence = None, marker_values: Sequence = None,
                       markers: Sequence | bool | str | dict = True, linestyles: Sequence | bool | str | dict = True,
                       colors: Sequence | bool | str | dict = True) -> list[dict[str, Any]]:
    """
    Returns a list of dictionaries with the markers, linestyles, and colors for each value in `values`. If a value or a sequence
    of values is given for `markers`, `linestyles`, or `colors`, the values are used. If `True` is given, the default cyclers are
    used.

    Parameters
    ----------
    color_values : Sequence, optional
        Values for colors. The same value will get the same color.
    linestyle_values : Sequence, optional
        Values for linestyles. The same value will get the same linestyle.
    marker_values : Sequence, optional
        Values for markers.  The same value will get the same marker.
    markers : Sequence, bool, or str, optional
        Marker cycler. If True, the default linestyle cycler is used.
    linestyles : Sequence, bool, or str, optional
        Linestyle cycler. If True, the default linestyle cycler is used.
    colors : Sequence, bool, or str, optional
        Color cycler. If True, the default marker cycler is used.

    Returns
    -------
    list of dict
        A list of dictionaries containing marker, linestyle, and color for each value.
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
        return [deepcopy(given_values[value]) for value in values]

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
        raise ValueError('One of `color_values`, `linestyle_values`, or `marker_values` must be provided.')

    markers = _set_cycler_value(markers, _marker_list, 'markers')
    linestyles = _set_cycler_value(linestyles, _linestyles, 'linestyles')
    colors = _set_cycler_value(colors, _colors, 'colors')

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

def _mk_legend_title(title: str) -> tuple:
    title = r'$\bf{' + title.replace(' ', r'}$ $\bf{') + r'}$'
    return Patch(visible=False), title


def _add_legend_section(line_styles, line_labels, title: str | None,
                        sort: bool = True, sort_key: Callable | None = None):
    """
    Creates a legend section with the given line styles and labels. If `title` is provided, it will be added as the first
    entry in the legend. The lines and labels can be sorted based on the `sort` flag and `sort_key`.
    If `sort` is True, the lines and labels will be sorted based on the `sort_key`. If `sort_key` is None, the labels will
    be sorted as strings.

    Parameters
    ----------
    line_styles: Sequence[dict[str, Any]]
    line_labels: Sequence[str]
    title: str | None
    sort: bool
    sort_key: Callable | None

    Returns
    -------
    tuple[list[plt.Line2D | LegendHandle], list[str]]
        A tuple containing the line handles and label values for the legend.
    """
    line_handles = []
    label_values = []
    if title is not None:
        line, label = _mk_legend_title(title)
        line_handles.append(line)
        label_values.append(label)

    lines = []
    labels = []
    for label, kwargs in zip(line_labels, line_styles):
        lines.append(plt.Line2D([], [], label=label, **kwargs))
        labels.append(label)

    if sort:
        labels, lines = sort_together(labels, lines, key=sort_key)

    line_handles.extend(lines)
    label_values.extend(labels)
    return line_handles, label_values


def legend_section(line_styles, line_labels, title: str | None, sort: bool = True, sort_key: Callable | None = None):
    """
    Creates a legend section with the given line styles and labels. If `title` is provided, it will be added as the first
    entry in the legend. The lines and labels can be sorted based on the `sort` flag and `sort_key`.
    If `sort` is True, the lines and labels will be sorted based on the `sort_key`. If `sort_key` is None, the labels will
    be sorted as strings.

    Parameters
    ----------
    line_styles: Sequence[dict[str, Any]]
    line_labels: Sequence[str]
    title: str | None
    sort: bool
    sort_key: Callable | None

    Returns
    -------
    dict
        A dictionary containing 'handles' and 'labels' for the legend.
    """

    line_handles, label_values = _add_legend_section(line_styles, line_labels, title, sort, sort_key)
    return {'handles': line_handles, 'labels': label_values}


def linelooks_plus_legend(labels, title=None, *, colors=None, linestyles=None, markers=None, sort=True,
                          sort_key: Callable | None = None):
    """
    Creates the line kwargs and legend configuration for a legend based on the provided labels and their associated
    cyclers for colors, linestyles, and markers.

    Parameters
    ----------
    labels : Sequence
        The labels for the legend.
    title : str, optional
        The title of the legend.
    colors : Sequence, optional
        Color values for the legend.
    linestyles : Sequence, optional
        Linestyle values for the legend.
    markers : Sequence, optional
        Marker values for the legend.
    sort : bool, optional
        Whether to sort the labels.
    sort_key : Callable, optional
        A function to use for sorting the labels. If None, the labels are sorted as strings.

    Returns
    -------
    tuple
        A tuple containing line kwargs and legend configuration.
    """
    line_kwargs_list = linelooks(labels, markers=markers, linestyles=linestyles, colors=colors)
    legend = legend_section(line_kwargs_list, labels, title, sort, sort_key)

    return line_kwargs_list, legend


def linelooks_sections_plus_legend(*, color_labels: Sequence[str | float | int] = None, linestyle_labels: Sequence[str | float | int] = None,
                                   marker_labels: Sequence[str|float|int] = None, color_values: Sequence = None, linestyle_values: Sequence = None,
                                   marker_values: Sequence = None, no_color='k', no_marker=True, no_linestyle=True,
                                   color_title=None, marker_title=None, linestyle_title=None, sort=True):
    """
    Creates a legend with linelooks for given labels and values.

    Parameters
    ----------
    color_labels : Sequence, optional
       Labels for colors.
    linestyle_labels : Sequence, optional
       Labels for linestyles.
    marker_labels : Sequence, optional
       Labels for markers.
    color_values : Sequence, optional
       Values for colors.
    linestyle_values : Sequence, optional
       Values for linestyles.
    marker_values : Sequence, optional
       Values for markers.
    no_color : str, optional
       Default color for missing values.
    no_marker : bool, optional
       Default marker for missing values.
    no_linestyle : bool, optional
       Default linestyle for missing values.
    color_title : str, optional
       Title for the color legend.
    marker_title : str, optional
       Title for the marker legend.
    linestyle_title : str, optional
       Title for the linestyle legend.
    sort : bool, optional
       Whether to sort the labels.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
       A tuple containing line kwargs and legend configuration.
   """
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

    color_labels, color_values = make_values(color_labels, color_values, 'color', _colors)
    linestyle_labels, linestyle_values = make_values(linestyle_labels, linestyle_values, 'linestyle', _linestyles)
    marker_labels, marker_values = make_values(marker_labels, marker_values, 'marker', _marker_list)

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
    return line_kwargs_iter, legend_from_linelooks_sections(line_kwargs_iter, color_labels=c_vals, linestyle_labels=l_vals, marker_labels=m_vals,
                                                            no_color=no_color, no_marker=no_marker, no_linestyle=no_linestyle, color_title=color_title,
                                                            marker_title=marker_title, linestyle_title=linestyle_title, sort=sort)


def legend_from_linelooks_sections(line_kwargs_iter, /, *, color_labels: Sequence[str | float | int] = None, linestyle_labels: Sequence[str | float | int] = None,
                                   marker_labels: Sequence[str|float|int] = None, no_color='k', no_marker=True, no_linestyle=True,
                                   color_title=None, marker_title=None, linestyle_title=None, sort=True, sort_key: Callable | None = None)\
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
                    if not (been[label] == line_kwargs.get(name, 'none')):
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
                line, label = _mk_legend_title(title)
                line_handles.append(line)
                label_values.append(label)
            lines, labels = make(labels, name, no_marker, no_linestyle)

            if sort:
                labels, lines = sort_together(labels, lines, key=sort_key)

            line_handles.extend(lines)
            label_values.extend(labels)

    return {'handles': line_handles, 'labels': label_values}


class LegendHandle(plt.Line2D):
    pass


def _sm_plus_colordict(values, cmap, sort_key, vmin=None, vmax=None, norm='linear'):
    """
    Creates a ScalarMappable object with a colormap and a color dictionary for the given values.

    Parameters
    ----------
    values : Sequence[Hashable]
        The values to get the colormap for.
    cmap : str | Callable
        A colormap name or a colormap function to use for coloring the lines.
    vmin : float, optional
        The minimum value for the colormap normalization. If None, the minimum of `values` is used.
    vmax : float, optional
        The maximum value for the colormap normalization. If None, the maximum of `values` is used.
    norm : str, optional
        The normalization method to use for the colormap. Can be 'linear' or 'log'. Defaults to 'linear'.

    Returns
    -------
    tuple[matplotlib.cm.ScalarMappable, dict[Hashable, tuple[float, float, float, float]]]
        A ScalarMappable object for the colormap normalization and a dictionary mapping each value to its corresponding RGBA color.
    """
    if sort_key is None:
        sort_key = lambda x: x
    elif not isinstance(sort_key, Callable):
        raise TypeError(f'sort_key should be a callable, not {type(sort_key)}.')

    unique_values = list(set(values))
    new_values = list(map(sort_key, unique_values))
    key_values, unique_values = sort_together(new_values, unique_values)
    _, sm = cbar_norm_colors(key_values, cmap, min_value=vmin, max_value=vmax, norm=norm)
    colors = sm.get_rgba(key_values)
    color_dict = {val: color for val, color in zip(unique_values, colors)}

    return sm, color_dict


def linelooks_with_cmap(values: Sequence[Hashable], *, cmap: str | Callable = "turbo",
                        markers: Sequence[str] | bool | str = False,
                        linestyles: Sequence | bool | str = False, sort_key: Callable | None = None, vmin=None,
                        vmax=None, norm="linear"):
    """
    Returns a list of dictionaries with the markers, linestyles, and colors for each value in `values`. If a value or a sequence
    of values is given for `markers`, `linestyles`, or `colors`, the values are used. If `True` is given, the default cyclers are
    used. The order of the values is preserved in the output. Same values will have the same linelook.

    Parameters
    ----------
    values : Sequence[Hashable]
        The values to get the linelook for, each value should be hashable.
    cmap : str | Callable | None, optional
        A colormap name or a colormap function to use for coloring the lines.
    markers : Sequence[str], bool, or str, optional
        Marker values or a flag to use default markers.
    linestyles : Sequence, bool, or str, optional
        Linestyle values or a flag to use default linestyles.]
    sort_key : Callable, optional
        A function to use for sorting the values. If None, the values are sorted as they are.
    vmin : float, optional
        The minimum value for the colormap normalization. If None, the minimum of `values` is used.
    vmax : float, optional
        The maximum value for the colormap normalization. If None, the maximum of `values` is used.
    norm : str, optional
        The normalization method to use for the colormap. Can be 'linear' or 'log'. Defaults to 'linear'.

    Returns
    -------
    sm : tuple[matplotlib.cm.ScalarMappable, list[dict[str, Any]]]
        A ScalarMappable object for the colormap normalization and a list of dictionaries containing marker, linestyle,
         and color for each value.
    """
    sm, color_dict = _sm_plus_colordict(values, cmap, sort_key, vmin=vmin, vmax=vmax, norm=norm)
    return sm, linelooks(values, markers=markers, linestyles=linestyles, colors=color_dict)


def linelooks_sections_with_cmap(*, color_values: Sequence = None, linestyle_values: Sequence = None,
                                 marker_values: Sequence = None, markers: Sequence | bool | str | dict = True,
                                 linestyles: Sequence | bool | str | dict = True, cmap: str | Callable = "turbo",
                                 sort_key: Callable | None = None, vmin: float | None = None,
                                 vmax: float | None = None, norm: Literal["lin", "log"] = "lin") \
        -> tuple[plt.cm.ScalarMappable, list[dict[str, Any]]]:
    """
    Creates a legend with linelooks for given labels and values, using a colormap for colors.

    Parameters
    ----------
    color_values: Sequence, optional
        Values for colors. The same value will get the same color.
    linestyle_values: Sequence, optional
        Values for linestyles. The same value will get the same linestyle.
    marker_values: Sequence, optional
        Values for markers. The same value will get the same marker.
    markers: Sequence, bool, or str, optional
        Marker cycler. If True, the default marker cycler is used.
    linestyles: Sequence, bool, or str, optional
        Linestyle cycler. If True, the default linestyle cycler is used.
    cmap: str | Callable, optional
        A colormap name or a colormap function to use for coloring the lines.
    sort_key: Callable, optional
        A function to use for sorting the values. If None, the values are sorted as they are.
    vmin: float, optional
        The minimum value for the colormap normalization. If None, the minimum of `color_values` is used.
    vmax: float, optional
        The maximum value for the colormap normalization. If None, the maximum of `color_values` is used.
    norm: Literal["lin", "log"], optional
        The normalization method to use for the colormap. Can be 'lin' or 'log'. Defaults to 'lin'.

    Returns
    -------
    tuple[plt.cm.ScalarMappable, list[dict[str, Any]]]
        A ScalarMappable object for the colormap normalization and a list of dictionaries containing marker, linestyle,
         and color for each value.
    """
    sm, color_dict = _sm_plus_colordict(color_values, cmap, sort_key, vmin=vmin, vmax=vmax, norm=norm)
    line_kwargs = linelooks_sections(color_values=color_values, linestyle_values=linestyle_values, marker_values=marker_values,
                                    markers=markers, linestyles=linestyles, colors=color_dict)
    return sm, line_kwargs


def linelooks_sections_plus_legend_with_cmap(*,
        color_labels: Sequence[str | float | int] = None, linestyle_labels: Sequence[str | float | int] = None,
        marker_labels: Sequence[str|float|int] = None, color_values: Sequence = None, linestyle_values: Sequence = None,
        marker_values: Sequence = None, no_color='k', no_marker=True, no_linestyle=True,
        color_title=None, marker_title=None, linestyle_title=None, sort=True,
        cmap: str | Callable = "turbo", sort_key: Callable | None = None, vmin: float | None = None,
        vmax: float | None = None, norm: Literal["lin", "log"] = "lin") \
            -> tuple[plt.cm.ScalarMappable, list[dict[str, Any]]]:
    """
    Creates a legend with linelooks for given labels and values using a colormap for colors.

    Parameters
    ----------
    color_labels : Sequence, optional
       Labels for colors.
    linestyle_labels : Sequence, optional
       Labels for linestyles.
    marker_labels : Sequence, optional
       Labels for markers.
    cmap : Callable, str
       Values for colors.
    sort_key: Callable, optional
        A function to use for sorting the values. If None, the values are sorted as they are.
    vmin: float, optional
        The minimum value for the colormap normalization. If None, the minimum of `color_values` is used.
    vmax: float, optional
        The maximum value for the colormap normalization. If None, the maximum of `color_values` is used.
    norm: Literal["lin", "log"], optional
        The normalization method to use for the colormap. Can be 'lin' or 'log'. Defaults to 'lin'.
    linestyle_values : Sequence, optional
       Values for linestyles.
    marker_values : Sequence, optional
       Values for markers.
    no_color : str, optional
       Default color for missing values.
    no_marker : bool, optional
       Default marker for missing values.
    no_linestyle : bool, optional
       Default linestyle for missing values.
    color_title : str, optional
       Title for the color legend.
    marker_title : str, optional
       Title for the marker legend.
    linestyle_title : str, optional
       Title for the linestyle legend.
    sort : bool, optional
       Whether to sort the labels.

    Returns
    -------
    tuple[plt.cm.ScalarMappable, list[dict[str, Any]], dict[str, Any]]
        A ScalarMappable object for the colormap normalization and a list of dictionaries containing marker, linestyle,
         and color for each value and legend configuration.
    """
    sm, color_dict = _sm_plus_colordict(color_labels, cmap, sort_key, vmin=vmin, vmax=vmax, norm=norm)
    line_kwargs, legend_kwargs = linelooks_sections_plus_legend(
        color_values=color_values, linestyle_values=linestyle_values, marker_values=marker_values, no_marker=no_marker,
        color_labels=color_labels, linestyle_labels=linestyle_labels, marker_labels=marker_labels, no_color=no_color,
        no_linestyle=no_linestyle, color_title=color_title, marker_title=marker_title, linestyle_title=linestyle_title,
        sort=sort)
    return sm, line_kwargs, legend_kwargs