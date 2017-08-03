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

import param
import parambokeh

from utils import Mag, CustomFunctor, DeconvolvedMoments, RAColumn, DecColumn, StarGalaxyLabeller
from dataset import Dataset

hv.extension('bokeh')

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
    hist_normed = param.Boolean(default=True)
    hist_xbin_range = param.Range(default=data.points.range('x'), bounds=data.points.range('x'))
    hist_ybin_range = param.Range(default=data.points.range('y'), bounds=data.points.range('y'))
    
    output = parambokeh.view.Plot()

    def __init__(self, data, *args, **kwargs):
        super(QAExplorer, self).__init__(*args, **kwargs)

        self.data = data
        self._points = None

    @property
    def points(self):
        if self._points is None:
            self._points = self.data.points
        return self._points

    def make_xhist(self, hist_nbins, hist_normed, hist_xbin_range, **kwargs):
        hist = hv.operation.histogram(self.points, num_bins=hist_nbins, 
                                      normed=hist_normed, dimension='x',
                                      bin_range=hist_xbin_range)
        return hist.opts(norm=dict(framewise=True, axiswise=True))
        
    def make_yhist(self, hist_nbins, hist_normed, hist_ybin_range, **kwargs):
        hist = hv.operation.histogram(self.points, num_bins=hist_nbins, 
                                      normed=hist_normed, dimension='y',
                                      bin_range=hist_ybin_range)
        return hist.opts(norm=dict(framewise=True, axiswise=True))

    def make_2d(self, *args, **kwargs):
        d = datashade(self.points, normalization='log', aggregator=ds.count_cat('label'),
                      cmap=self.data.color_map)
        return dynspread(d).opts(plot=dict(width=600, height=400))

    def view(self):
        xhist = hv.DynamicMap(self.make_xhist, kdims=[], streams=[self])
        yhist = hv.DynamicMap(self.make_yhist, kdims=[], streams=[self])
        
        plot_2d = self.make_2d()

        return hv.Layout([plot_2d, xhist, yhist]).cols(2)

explorer = QAExplorer(data)
explorer.output = explorer.view()

doc = parambokeh.Widgets(explorer, continuous_update=True, callback=explorer.event, 
                        view_position='right', mode='server')

