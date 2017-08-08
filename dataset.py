import numpy as np
import pandas as pd
import holoviews as hv

from bokeh.layouts import layout
from bokeh.plotting import curdoc
from bokeh.models.widgets import Panel, Tabs, Select, RadioButtonGroup, TextInput, PreText
from bokeh.palettes import Spectral4, Category10, Dark2

import datashader as ds
from holoviews.operation.datashader import aggregate, shade, datashade, dynspread
from holoviews.operation import decimate, histogram
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
        self._default_range = None
        self._color_list = None

        # Holoviews objects
        self._points = None
        self._points_list = None
        self._xdim = None
        self._ydim = None
        self._labeldim = None

    def _reset(self):
        self._df = None
        self._default_range = None
        self._color_list = None
        self._points = None
        self._points_list = None
        self._xdim = None
        self._ydim = None
        self._labeldim = None

    @property
    def catalog(self):
        return self._catalog

    @catalog.setter
    def catalog(self, new):
        if new is self._catalog:
            return
        self._catalog = new
        self._reset()

    @property
    def xFunc(self):
        return self._xFunc

    @xFunc.setter
    def xFunc(self, new):
        if new is self._xFunc:
            return
        self._xFunc = new
        self._reset()

    @property
    def yFunc(self):
        return self._yFunc

    @yFunc.setter
    def yFunc(self, new):
        if new is self._yFunc:
            return
        self._yFunc = new
        self._reset()

    @property
    def labeller(self):
        return self._labeller

    @labeller.setter
    def labeller(self, new):
        if new is self._labeller:
            return
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

        df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
        self._df = df

    @property
    def df(self):
        if self._df is None:
            self._generate_df()
        return self._df

    def _calc_default_range(self):
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

    @property
    def default_range(self):
        if self._default_range is None:
            self._default_range = self._calc_default_range()
        return self._default_range

    def _make_dims(self):
        xRange, yRange = self.default_range
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

    @property
    def points_list(self):
        if self._points_list is None:
            self._points_list = [hv.Points(self.df.query('label=="{}"'.format(l)),
                                            kdims=[self.xdim, self.ydim],
                                            vdims=[self.labeldim])
                                    for l in self.labels]
        return self._points_list

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
