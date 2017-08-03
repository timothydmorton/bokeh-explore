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

import datashader as ds
from holoviews.operation.datashader import aggregate, shade, datashade, dynspread
from holoviews.operation import decimate, histogram

import param
import parambokeh

hv.extension('bokeh')

pts = np.random.randn(10000,2)
pts[:,0] *= 50
points = hv.Points(pts)


class HistExplorer(hv.streams.Stream):
    nbins = param.Integer(default=20, bounds=(10,100))
    normed = param.Boolean(default=True)
    xbin_range = param.Range(default=points.range('x'), bounds=points.range('x'))
    ybin_range = param.Range(default=points.range('y'), bounds=points.range('y'))
    
    output = parambokeh.view.Plot()

    def __init__(self, data, *args, **kwargs):
        super(HistExplorer, self).__init__(*args, **kwargs)

        self.data = data

        self._points = None

    @property
    def points(self):
        if self._points is None:
            self._points = hv.Points(self.data)
        return self._points

    def make_xhist(self, nbins, normed, xbin_range, **kwargs):
        hist = hv.operation.histogram(self.points, num_bins=nbins, normed=normed, dimension='x',
                                      bin_range=xbin_range)
        return hist.opts(norm=dict(framewise=True, axiswise=True))
        
    def make_yhist(self, nbins, normed, ybin_range, **kwargs):
        hist = hv.operation.histogram(self.points, num_bins=nbins, normed=normed, dimension='y',
                                      bin_range=ybin_range)
        return hist.opts(norm=dict(framewise=True, axiswise=True))


    def view(self):
        xhist = hv.DynamicMap(self.make_xhist, kdims=[], streams=[self])
        yhist = hv.DynamicMap(self.make_yhist, kdims=[], streams=[self])
        
        l = hv.Layout([xhist, yhist])
        return l
    
explorer = HistExplorer(pts)
explorer.output = explorer.view()

doc = parambokeh.Widgets(explorer, continuous_update=True, callback=explorer.event, 
                        view_position='right', mode='server')

