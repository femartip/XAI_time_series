import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as patches

ColorType = Union[Optional[str]], Tuple[float, float, float, float]
LineStyle = Union[Optional[str], Tuple[float, Tuple[float, float]]]


class EllipseParams:
    x_pos: int
    y_pos: float
    radius: float
    color: ColorType
    fill: bool

    def __init__(self, x_pos: int, y_pos: float, radius: float, color: ColorType, fill: bool):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = radius
        self.color = color
        self.fill = fill


class ScatterParams:
    x_values: List[int]
    y_values: List[float]
    color: ColorType
    marker: Optional[str]

    def __init__(self, x_values: List[int], y_values: List[float], color: ColorType = None,
                 marker: Optional[str] = None):
        self.x_values = x_values
        self.y_values = y_values
        self.color = color
        self.marker = marker


class TSParam:
    x_values: List[int] | np.ndarray
    y_values: List[float] | np.ndarray
    fmat: Optional[str]
    linestyle: LineStyle
    linewidth: Optional[float]
    color: ColorType

    def __init__(self, x_values: List[int] | np.ndarray, y_values: List[float] | np.ndarray, fmat: str = None,
                 linestyle: LineStyle = None, linewidth: Optional[float] = None,
                 color: ColorType = None):
        self.x_values = x_values
        self.y_values = y_values
        self.fmat = fmat
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.color = color


class PlotParams:
    ts_params: [TSParam]
    scatter_params: [ScatterParams]
    ellipse_params: [EllipseParams]
    title: Optional[str]
    folder: Optional[str]
    save_file: Optional[str]
    display: bool
    x_lim: Optional[Tuple[int, int]]
    y_lim: Optional[Tuple[float, float]]

    def __init__(self, ts_params: List[TSParam] = None, scatter_params: List[ScatterParams] = None,
                 ellipse_params: List[EllipseParams] = None, title: Optional[str] = None,
                 save_file: Optional[str] = None, folder: Optional[str] = None,
                 display: bool = False,
                 x_lim: Optional[Tuple[int, int]] = None, y_lim: Optional[Tuple[float, float]] = None):
        if ts_params is None:
            ts_params = []
        if scatter_params is None:
            scatter_params = []
        if ellipse_params is None:
            ellipse_params = []

        if x_lim is None:
            min_x = min([min([x for x in ts_param.x_values]) for ts_param in ts_params])
            max_x = max([max([x for x in ts_param.x_values]) for ts_param in ts_params])
            x_lim = (min_x, max_x)
            self.x_lim = x_lim
        if y_lim is None:
            min_y = min([min([y for y in ts_param.y_values]) for ts_param in ts_params])
            max_y = max([max([y for y in ts_param.y_values]) for ts_param in ts_params])
            y_lim = (min_y, max_y)
            self.y_lim = y_lim

        self.ts_params = ts_params
        self.scatter_params = scatter_params
        self.ellipse_params = ellipse_params
        self.title = title
        self.save_file = save_file
        self.folder = folder
        self.display = display
        self.x_lim = x_lim
        self.y_lim = y_lim

    def make_plot(self):
        make_plot(plotParam=self)


def get_args_and_kwargs(params: TSParam | ScatterParams):
    plot_args = [params.x_values, params.y_values]
    plot_kwargs = {}
    if params.color is not None:
        plot_kwargs['color'] = params.color

    if isinstance(params, TSParam):
        if params.fmat is not None:
            plot_args.append(params.fmat)
        if params.linestyle is not None:
            plot_kwargs['linestyle'] = params.linestyle
        if params.linewidth is not None:
            plot_kwargs['linewidth'] = params.linewidth

    if isinstance(params, ScatterParams):
        if params.marker is not None:
            plot_kwargs['marker'] = params.marker
    return plot_args, plot_kwargs


def make_plot(plotParam: PlotParams):
    # Always clear
    plt.clf()
    fig, ax = plt.subplots()
    ax.set_xlim(plotParam.x_lim)
    ax.set_ylim(plotParam.y_lim)
    for scatterParam in plotParam.scatter_params:
        args, kwargs = get_args_and_kwargs(scatterParam)
        ax.scatter(*args, **kwargs, zorder=1)
    for tsParam in plotParam.ts_params:
        args, kwargs = get_args_and_kwargs(tsParam)
        ax.plot(*args, **kwargs, zorder=2)

    for ellipseParam in plotParam.ellipse_params:
        x_axsis_r = ellipseParam.radius
        y_axsis_r = ellipseParam.radius * (
                abs(plotParam.y_lim[0] - plotParam.y_lim[1]) / abs(plotParam.x_lim[0] - plotParam.x_lim[1]))
        ellipse = patches.Ellipse((ellipseParam.x_pos, ellipseParam.y_pos), x_axsis_r, y_axsis_r,
                                  fill=ellipseParam.fill, color=ellipseParam.color, zorder=3)
        ax.add_patch(ellipse)

    if plotParam.title is not None:
        plt.title(plotParam.title)

    if plotParam.save_file is not None:
        folder = plotParam.folder
        if folder is None:
            folder = "PyPlots"
        plt.savefig(f"{folder}/{plotParam.save_file}.png")

    if plotParam.display:
        plt.show()

    plt.close(fig)


def run():
    color = 'red'
    color = mcolors.to_rgba(color, alpha=0.9)
    fmat = None  # "--"
    linestyle = (0, (2, 0.5))
    linewidth = 5
    x_value = [1, 2]
    y_value = [4, 2]

    ts_param = TSParam(x_values=x_value, y_values=y_value, fmat=fmat, color=color, linestyle=linestyle,
                       linewidth=linewidth)

    color = 'blue'
    marker = "*"
    x_value = [3, 4]
    y_value = [5, 3]
    sc_param = ScatterParams(x_values=x_value, y_values=y_value, color=color, marker=marker)
    location = "testImg"
    x_lim = (0, 6)
    y_lim = (0, 6)

    color = "grey"
    x_pos = 3
    y_pos = 3
    fill = True
    radius = 0.1
    ellipse_param_inner = EllipseParams(x_pos=x_pos, y_pos=y_pos, radius=radius, color=color, fill=fill)
    color = "black"
    x_pos = 3
    y_pos = 3
    fill = False
    radius = 0.15
    ellipse_param_outer = EllipseParams(x_pos=x_pos, y_pos=y_pos, radius=radius, color=color, fill=fill)
    ellipse_params = [ellipse_param_inner, ellipse_param_outer]

    plot_params = PlotParams(ts_params=[ts_param], scatter_params=[sc_param], ellipse_params=ellipse_params,
                             save_file=location, display=True,
                             x_lim=x_lim, y_lim=y_lim)

    make_plot(plot_params)


if __name__ == "__main__":
    run()
