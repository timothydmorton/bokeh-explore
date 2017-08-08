import numpy as np
import holoviews as hv
import pandas as pd

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.io import show
from bokeh.layouts import layout
from bokeh.plotting import curdoc
from bokeh.models.widgets import Panel, Tabs, Select, RadioButtonGroup, TextInput, PreText
from bokeh.palettes import Spectral4, Category10, Dark2

import holoviews as hv

import datashader as ds
from holoviews.operation.datashader import aggregate, shade, datashade, dynspread
from holoviews.operation import decimate, histogram

from holoviews.plotting.bokeh.callbacks import callbacks, Callback

hv.extension('bokeh')

class BoundsCallback(Callback):
    """
    Returns the bounds of a box_select tool.
    """

    models = ['box_select']
    extra_models = ['plot']
    code="""
    frame = plot.plot_canvas.frame;
    xscale = frame.xscales['default']
    yscale = frame.yscales['default']
    data['x0'] = xscale.invert(cb_obj.overlay.left)
    data['y0'] = yscale.invert(cb_obj.overlay.bottom)
    data['x1'] = xscale.invert(cb_obj.overlay.right)
    data['y1'] = yscale.invert(cb_obj.overlay.top)
    """

    def _process_msg(self, msg):
        if all(c in msg for c in ['x0', 'y0', 'x1', 'y1']):
            return {'bounds': (msg['x0'], msg['y0'], msg['x1'], msg['y1'])}
        else:
            return {}

callbacks[hv.streams.BoundsXY] = BoundsCallback

import param
import parambokeh

from utils import Mag, CustomFunctor, DeconvolvedMoments, RAColumn, DecColumn, StarGalaxyLabeller
from dataset import Dataset


catalog = pd.read_hdf('data/forced_big.h5')

xFuncs = {'base_PsfFlux' : Mag('base_PsfFlux'),
          'modelfit_CModel' : Mag('modelfit_CModel')}
yFuncs = {'modelfit_CModel - base_PsfFlux' : CustomFunctor('mag(modelfit_CModel) - mag(base_PsfFlux)'),
          'Deconvolved Moments' : DeconvolvedMoments()}

xFunc = xFuncs['base_PsfFlux']
yFunc = yFuncs['modelfit_CModel - base_PsfFlux']
labeller = StarGalaxyLabeller()
data = Dataset(catalog, xFunc, yFunc, labeller)

class QAExplorer(hv.streams.Stream):

    hist_nbins = param.Integer(default=20, bounds=(10,100))
    hist_xbin_range = param.Range(default=data.points.range('x'), bounds=data.points.range('x'))
    hist_ybin_range = param.Range(default=data.points.range('y'), bounds=data.points.range('y'))

    x_data = param.ObjectSelector(default='base_PsfFlux', 
                                objects=list(xFuncs.keys()))
    y_data = param.ObjectSelector(default='modelfit_CModel - base_PsfFlux', 
                                objects=list(yFuncs.keys()))

    output = parambokeh.view.Plot()

    def __init__(self, data, *args, **kwargs):
        super(QAExplorer, self).__init__(*args, **kwargs)

        self.data = data

    @property
    def labels(self):
        return self.data.labels

    @property
    def points(self):
        return self.data.points

    @property
    def points_list(self):
        return self.data.points_list

    def _make_hist(self, dimension, num_bins, rng):
        hists = []
        for lbl, pts, c in zip(self.labels, self.points_list, self.data.color_list):
            vals = pts.dimension_values(dimension)
            lo = np.percentile(vals, 0.5)
            hi = np.percentile(vals, 99.5)
            # print(lbl, dimension, lo, hi)
            if rng[0] > lo:
                lo = rng[0]
            if rng[1] < hi:
                hi = rng[1]
            # print('now', lo, hi)

            opts = 'Histogram [yaxis=None] (alpha=0.3, cmap=[{}])'.format(c) + \
                   ' {+framewise +axiswise} '
            h = hv.operation.histogram(pts, num_bins=num_bins,
                            normed='height', dimension=dimension,
                            bin_range=(lo, hi)).opts(opts)
            hists.append(h)

        return hv.Overlay(hists)

    def make_xhist(self, hist_nbins, hist_xbin_range, x_data, y_data, **kwargs):
        self.set_catalog(x_data=x_data, y_data=y_data)
        return self._make_hist('x', num_bins=hist_nbins,
                                rng=hist_xbin_range)
        
    def make_yhist(self, hist_nbins, hist_ybin_range, x_data, y_data, **kwargs):
        self.set_catalog(x_data=x_data, y_data=y_data)
        return self._make_hist('y', num_bins=hist_nbins,
                                rng=hist_ybin_range)

    def make_datashaded(self, *args, **kwargs):
        d = datashade(self.points, normalization='log', aggregator=ds.count_cat('label'),
                      cmap=self.data.color_map)
        d = dynspread(d).opts(plot=dict(width=600, height=400, tools=['box_select']))
        return d

    def make_bounds_box(self, bounds, **kwargs):
        return hv.Bounds(lbrt=bounds)

    def set_catalog(self, x_data, y_data, **kwargs):
        self.data.xFunc = xFuncs[x_data]
        self.data.yFunc = yFuncs[y_data]
        # self._make()

    def view(self):
        xhist = hv.DynamicMap(self.make_xhist, kdims=[], streams=[self])
        yhist = hv.DynamicMap(self.make_yhist, kdims=[], streams=[self])

        # plot_2d = hv.DynamicMap(self.make_datashaded, kdims=[], streams=[self])        
        plot_2d = self.make_datashaded()

        box = hv.streams.BoundsXY(source=self.points, bounds=(20,0,20,0))
        bounds_box = hv.DynamicMap(self.make_bounds_box, streams=[self, box])

        l = (plot_2d * bounds_box + hv.Empty() + yhist + xhist).cols(2)
        return l

        # l = plot_2d << yhist << xhist
        # return l.opts(plot=dict(shared_axes=False), norm=dict(framewise=True, axiswise=True))

explorer = QAExplorer(data)
explorer.output = explorer.view()

doc = parambokeh.Widgets(explorer, continuous_update=True, callback=explorer.event, 
                        view_position='right', mode='server')

