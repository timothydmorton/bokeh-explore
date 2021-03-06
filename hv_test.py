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
from holoviews.operation import decimate
decimate.max_samples=5000
from datashader.colors import Sets1to3

from utils import Mag, CustomFunctor, DeconvolvedMoments, RAColumn, DecColumn, StarGalaxyLabeller


class Dataset(object):
    """
    Holds x, y and position data for interactive dashboard

    X and Y data are computed from underlying catalog
    """
    def __init__(self, catalog, xFunc, yFunc, labeller):
        self._catalog = catalog
        self._xFunc = xFunc
        self._yFunc = yFunc
        self._labeller = labeller

        self._df = None
        self._color_list = None

        # Holoviews objects
        self._xdim = None
        self._ydim = None
        self._labeldim = None
        self._points = None
        self._datashaded = None
        self._decimated = None

    def _reset(self):
        self._df = None
        self._reset_hv()

    def _reset_hv(self):
        self._xdim = None
        self._ydim = None
        self._labeldim = None
        self._points = None
        self._datashaded = None

    @property
    def catalog(self):
        return self._catalog

    @catalog.setter
    def catalog(self, new):
        self._catalog = new
        self._reset()

    @property
    def xFunc(self):
        return self._xFunc

    @xFunc.setter
    def xFunc(self, new):
        self._xFunc = new
        self._reset()

    @property
    def yFunc(self):
        return self._yFunc

    @yFunc.setter
    def yFunc(self, new):
        self._yFunc = new
        self._reset()

    @property
    def labeller(self):
        return self._labeller

    @labeller.setter
    def labeller(self, new):
        self._labeller = new
        self._reset()

    def _generate_df(self):
        x = self.xFunc(self.catalog)
        y = self.yFunc(self.catalog)
        label = self.labeller(self.catalog)
        ra = np.rad2deg(self.catalog['coord_ra'])
        dec = np.rad2deg(self.catalog['coord_dec'])

        df = pd.DataFrame({'x' : x,
                           'y' : y,
                           'label' : label,
                           'ra' : ra,
                           'dec' : dec})

        df = df.replace([np.inf, -np.inf], np.nan).dropna(how='all')
        self._df = df

    @property
    def df(self):
        if self._df is None:
            self._generate_df()
        return self._df

    def _get_default_range(self):
        x = self.df['x'].dropna()
        y = self.df['y'].dropna()
        xMed = np.median(x)
        yMed = np.median(y)
        xMAD = np.median(np.absolute(x - xMed))
        yMAD = np.median(np.absolute(y - yMed))

        ylo = yMed - 10*yMAD
        yhi = yMed + 10*yMAD

        xlo, xhi = x.quantile([0., 0.99])
        xBuffer = xMAD/4.
        xlo -= xBuffer
        xhi += xBuffer

        # print(xlo, xhi, ylo, yhi)
        return (xlo, xhi), (ylo, yhi)

    def _make_dims(self):
        xRange, yRange = self._get_default_range()
        self._xdim = hv.Dimension('x', label=self.xFunc.name, range=xRange)
        self._ydim = hv.Dimension('y', label=self.yFunc.name, range=yRange)
        self._labeldim = hv.Dimension('label', label='Object Type', values=self.df.label.unique())            

    @property
    def xdim(self):
        if self._xdim is None:
            self._make_dims()
        return self._xdim

    @property
    def ydim(self):
        if self._ydim is None:
            self._make_dims()
        return self._ydim

    @property
    def labeldim(self):
        if self._labeldim is None:
            self._make_dims()
        return self._labeldim

    @property
    def points(self):
        if self._points is None:
            # Version to make points a dictionary
            # self._points = {}
            # for l in self.labels:
            #     qdf = self.df.query('label=="{}"'.format(l))
            #     self._points[l] = hv.Points(qdf, kdims=[self.xdim, self.ydim], label=l)

            # Make points a single thing
            self._points = hv.Points(self.df, kdims=[self.xdim, self.ydim], 
                                     vdims=[self.labeldim])
        return self._points

    # @property
    # def pointsDict(self):
    #     if self._pointsDict is None:
    #         self._pointsDict = {}
    #         for l in self.labels:
    #             qdf = self.df.query('label=="{}"'.format(l))
    #             self._pointsDict[l] = hv.Points(qdf, kdims=[self.xdim, self.ydim], label=l)


    @property
    def labels(self):
        return self.labeldim.values

    @property
    def color_map(self):
        return {l:c for l,c in zip(self.labels, Sets1to3)}

    @property
    def color_list(self):
        """List of colors in same order that datashader plots them, hopefully
        """
        if self._color_list is None:
            labels = list(self.df.groupby('label').count().index)
            self._color_list = [self.color_map[l] for l in self.labels][::-1]
        return self._color_list

    @property
    def datashaded(self):
        if self._datashaded is None:
            # This is a solution to overlay two images
            # dlist = []
            # for l,pts in self.points.items():
            #     d = datashade(pts, normalization='log', aggregator=ds.count_cat('label'), cmap=self.color_map)
            #     dlist.append(dynspread(d).opts(plot=dict(width=600, height=400)))
            # self._datashaded = hv.Overlay(dlist)

            # This just colors by category a single image
            d = datashade(self.points, normalization='log', aggregator=ds.count_cat('label'),
                          cmap=self.color_map)
            self._datashaded = dynspread(d).opts(plot=dict(width=600, height=400))
        return self._datashaded

    @property
    def decimated(self):
        if self._decimated is None:
            self._decimated = decimate(self.points.opts(plot=dict(color_index='label'),
                                                        style=dict(cmap=self.color_list)))

catalog = pd.read_hdf('data/forced_big.h5')

xFuncs = {'base_PsfFlux' : Mag('base_PsfFlux'),
          'modelfit_CModel' : Mag('modelfit_CModel')}
yFuncs = {'modelfit_CModel - base_PsfFlux' : CustomFunctor('mag(modelfit_CModel) - mag(base_PsfFlux)'),
          'Deconvolved Moments' : DeconvolvedMoments()}

xFunc = xFuncs['base_PsfFlux']
yFunc = yFuncs['modelfit_CModel - base_PsfFlux']
labeller = StarGalaxyLabeller()
data = Dataset(catalog, xFunc, yFunc, labeller)

scatter_all = data.datashaded

# dmap = dynspread(datashade(data.points, normalization='log', aggregator=ds.count_cat('label')))
# dmap = dmap.opts(plot=dict(width=1000, height=800))

renderer = hv.renderer('bokeh').instance(mode='server')

def modify_doc(doc):
    # Create HoloViews plot and attach the document
    hvplot = renderer.get_plot(scatter_all, doc)

    x_select = Select(title="X-axis:", value='base_PsfFlux', options=list(xFuncs.keys()))
    y_select = Select(title="Y-axis:", value='modelfit_CModel - base_PsfFlux', options=list(yFuncs.keys()))

    def update_plot():
        new_scatter_all = data.datashaded
        new_plot = renderer.get_plot(new_scatter_all, doc)
        l.children[0] = new_plot.state

    def update_xFunc(attr, old, new):
        try:
            data.xFunc = xFuncs[new]
        except KeyError:
            data.xFunc = CustomFunctor(new)
            if new not in x_select.options:
                x_select.options.append(new)
            x_select.value = new
        update_plot()

    def update_yFunc(attr, old, new):
        try:
            data.yFunc = yFuncs[new]
        except KeyError:
            data.yFunc = CustomFunctor(new)
            if new not in y_select.options:
                y_select.options.append(new)
            y_select.value = new
        update_plot()

    x_select.on_change('value', update_xFunc)
    y_select.on_change('value', update_yFunc)

    l = layout([[hvplot.state], 
                [x_select, y_select]], sizing_mode='fixed')
    


    doc.add_root(l)
    return doc

# To display in the notebook
# handler = FunctionHandler(modify_doc)
# app = Application(handler)
# show(app, notebook_url='localhost:8888')

# To display in a script
doc = modify_doc(curdoc()) 
