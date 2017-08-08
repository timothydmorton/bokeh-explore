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

from holoviews.plotting.bokeh.callbacks import callbacks, Callback

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

N = 10000
data = pd.DataFrame({'x' : np.random.randn(N),
                     'y' : np.random.randn(N)*50,
                     'label' : np.random.randint(2, size=N)})
x_range = (data.x.min(), data.x.max())
y_range = (data.y.min(), data.y.max())

class HistExplorer(hv.streams.Stream):
    nbins = param.Integer(default=20, bounds=(10,100))
    normed = param.Boolean(default=True)
    xbin_range = param.Range(default=x_range, bounds=x_range)
    ybin_range = param.Range(default=y_range, bounds=y_range)

    output = parambokeh.view.Plot()

    def __init__(self, data, *args, **kwargs):
        super(HistExplorer, self).__init__(*args, **kwargs)

        self.data = data
        self._points = None

    @property
    def points(self):
        if self._points is None:
            self._points = hv.Points(self.data, kdims=['x', 'y'], vdims=['label'])
        return self._points

    def make_xhist(self, nbins, normed, xbin_range, **kwargs):
        hist = hv.operation.histogram(self.points, num_bins=nbins, normed=normed, dimension='x',
                                      bin_range=xbin_range)
        return hist.opts(norm=dict(framewise=True, axiswise=True),
                         style=dict(alpha=0.3))

    def make_yhist(self, nbins, normed, ybin_range, **kwargs):
        hist = hv.operation.histogram(self.points, num_bins=nbins, normed=normed, dimension='y',
                                      bin_range=ybin_range)
        return hist.opts(norm=dict(framewise=True, axiswise=True),
                         style=dict(alpha=0.3))

    def make_datashaded(self, *args, **kwargs):
        d = datashade(self.points, normalization='log', aggregator=ds.count_cat('label'),
                      cmap={0:'red', 1:'blue'})
        d = dynspread(d).opts(plot=dict(width=600, height=400, tools=['box_select']))
        return d

    def make_bounds_box(self, bounds, **kwargs):
        return hv.Bounds(lbrt=bounds)

    def view(self):
        xhist = hv.DynamicMap(self.make_xhist, kdims=[], streams=[self])
        yhist = hv.DynamicMap(self.make_yhist, kdims=[], streams=[self])

        box = hv.streams.BoundsXY(source=self.points, bounds=(20,0,20,0))
        bounds_box = hv.DynamicMap(self.make_bounds_box, streams=[self, box])

        datashaded = self.make_datashaded()

        return hv.Layout([datashaded, hv.Empty(), xhist, yhist]).cols(2)

explorer = HistExplorer(data)
explorer.output = explorer.view()

doc = parambokeh.Widgets(explorer, continuous_update=True, callback=explorer.event, 
                        view_position='right', mode='server')