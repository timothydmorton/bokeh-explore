import numpy as np
import holoviews as hv
import pandas as pd

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.io import show
from bokeh.layouts import layout
from bokeh.plotting import curdoc

import datashader as ds
from holoviews.operation.datashader import aggregate, shade, datashade, dynspread
from holoviews.operation import decimate, histogram

import param
import parambokeh

hv.extension('bokeh')

class HistExplorer(hv.streams.Stream):
    nbins = param.Integer(default=20, bounds=(10,100))
    normed = param.Boolean(default=True)    
    # xbin_range = param.Range(default=x_range, bounds=x_range)
    # ybin_range = param.Range(default=y_range, bounds=y_range)

    logN = param.Number(default=5, bounds=(3,8))
    x_std = param.Number(default=1, bounds=(1,5)) 
    y_std = param.Number(default=50, bounds=(50,200))

    output = parambokeh.view.Plot()

    def __init__(self, *args, **kwargs):
        super(HistExplorer, self).__init__(*args, **kwargs)

        self._data = None
        self._points = None

        self._last_props = {k:None for k in ['logN', 'x_std', 'y_std']}

    @property
    def data(self):
        if self._data is None:
            self.generate_data(self.logN, self.x_std, self.y_std)
        return self._data

    def generate_data(self, logN, x_std, y_std, **kwargs):
        generate = False
        for k in ['logN', 'x_std', 'y_std']:
            if self._last_props[k] != eval(k):
                generate = True
                break

        if generate:
            print('Generating data...')
            data = pd.DataFrame({'x' : np.random.randn(int(10**logN))*x_std,
                                 'y' : np.random.randn(int(10**logN))*y_std,
                                 'label' : np.random.randint(2, size=int(10**logN))})
            self._data = data
            self._points = self._make_points()
            self._last_props = dict(logN=logN, x_std=x_std, y_std=y_std)

    def _make_points(self):
        return hv.Points(self.data, kdims=['x', 'y'], vdims=['label'])

    @property
    def points(self):
        if self._points is None:
            self._points = self._make_points()
        return self._points

    def make_xhist(self, nbins, normed, **kwargs):
        hist = hv.operation.histogram(self.points, num_bins=nbins, normed=normed, dimension='x')
        return hist.opts(norm=dict(framewise=True, axiswise=True),
                         style=dict(alpha=0.3))
        
    def make_yhist(self, nbins, normed, **kwargs):
        hist = hv.operation.histogram(self.points, num_bins=nbins, normed=normed, dimension='y')
        return hist.opts(norm=dict(framewise=True, axiswise=True),
                         style=dict(alpha=0.3))

    def make_datashaded(self, logN=None, x_std=None, y_std=None, x_range=None, y_range=None, **kwargs):
        self.generate_data(logN=logN, x_std=x_std, y_std=y_std)

        d = datashade(self.points, normalization='log', aggregator=ds.count_cat('label'),
                      cmap={0:'red', 1:'blue'}, dynamic=False, x_range=x_range, y_range=y_range)
        d = dynspread(d).opts(plot=dict(width=600, height=400, tools=['box_select']))
        return d

    def make_scatter(self, *args, **kwargs):
        return self.points.opts(plot=dict(tools=['box_select']))

    def make_bounds_box(self, bounds, **kwargs):
        return hv.Bounds(lbrt=bounds)

    def view(self):
        # scatter = self.make_datashaded(x_std=self.x_std, y_std=self.y_std)
        scatter = hv.DynamicMap(self.make_datashaded, kdims=[], streams=[self, hv.streams.RangeXY()])

        xhist = hv.DynamicMap(self.make_xhist, kdims=[], streams=[self])
        yhist = hv.DynamicMap(self.make_yhist, kdims=[], streams=[self])
        
        box = hv.streams.BoundsXY(source=self.points, bounds=(0,0,0,0))
        bounds_box = hv.DynamicMap(self.make_bounds_box, streams=[self, box])

        return hv.Layout([scatter * bounds_box, hv.Empty(), xhist, yhist]).cols(2)
    
explorer = HistExplorer()
explorer.output = explorer.view()

doc = parambokeh.Widgets(explorer, continuous_update=True, callback=explorer.event, 
                        view_position='right', mode='server')

