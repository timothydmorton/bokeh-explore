''' Present a demo of what QA plots may look like.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve qa_prototype.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/qa_prototype

in your browser.

'''

from __future__ import division, print_function

import numpy as np
import pandas as pd

from bokeh.layouts import row, column
from bokeh.models import (BoxSelectTool, LassoSelectTool, 
                        Spacer, ColumnDataSource, Range1d, HoverTool)
from bokeh.charts import Scatter
from bokeh.plotting import figure, curdoc
from bokeh.palettes import Spectral4, Category10, Dark2

from utils import Mag, MagDiff, StarGalaxyLabeller, makeDataSource, Sizer
from utils import RAColumn, DecColumn

from bokeh.plotting.figure import Figure
from bokeh.models import CategoricalColorMapper, ColumnDataSource, LinearInterpolator
from bokeh.models import Range1d, Circle
from bokeh.layouts import widgetbox, layout, row, column
from bokeh.models.widgets import Slider

class QAPlot(object):
    def __init__(self, catalog, labeller=StarGalaxyLabeller(), palette=Category10[10]):
        self.catalog = catalog
        self.palette = palette

        self._df = None
        
        # This will be a dict, with keys as labels
        self._sources = None

        # Figures, etc.
        self._figure = None
        self._renderers = None
        
        # Functors
        self._labeller = labeller

    def _reset(self):
        self._sources = None
        self._df = None
        
    @property
    def labeller(self):
        return self._labeller
    
    @labeller.setter
    def labeller(self, f):
        self._labeller = f
        self._reset()
                
    def _make_sources(self):
        
        return {k:ColumnDataSource(self.df.query('label=="{0}"'.format(k))) 
                 for k in self.labels}            

    @property
    def df(self):
        if self._df is None:
            self._df = self._make_df()
        return self._df
    
    @property
    def labels(self):
        return np.unique(self.df.label)        
    
    @property
    def sources(self):
        """Dictionary of ColumnDataSources, keyed by labels
        """
        if self._sources is None:
            self._sources = self._make_sources()
        return self._sources

    @property
    def figure(self):
        if self._figure is None:
            self._make_figure()
        return self._figure

    @property
    def renderers(self):
        if self._renderers is None:
            self._make_figure()
        return self._renderers

    def _make_df(self):
        raise NotImplementedError

    def _make_figure(self):
        raise NotImplementedError


class QAScatterPlot(QAPlot):
    figure_kwargs = dict(active_scroll='wheel_zoom', 
                         active_drag='box_select',
                         plot_width=600, plot_height=400,
                         toolbar_location="below",
                         toolbar_sticky=False)

    def __init__(self, catalog, xFunc=None, yFunc=None, sizer=Sizer(1), alpha=0.8,
                 **kwargs):

        self._xFunc = xFunc
        self._yFunc = yFunc
        self._sizer = sizer

        self.alpha = alpha

        super(QAScatterPlot, self).__init__(catalog, **kwargs)

    @property
    def xFunc(self):
        return self._xFunc
    
    @xFunc.setter
    def xFunc(self, f):
        self._xFunc = f
        self._reset()
        
    @property
    def yFunc(self):
        return self._yFunc
    
    @yFunc.setter
    def yFunc(self, f):
        self._yFunc = f
        self._reset()

    @property
    def sizer(self):
        return self._sizer
    
    @sizer.setter
    def sizer(self, f):
        self._sizer = f
        self._reset()

    def selected_inds(self, label):
        return self.renderers[label].data_source.selected['1d']['indices']

    def selected_column(self, label, col):
        inds = self.selected_inds[label]
        return self.df.iloc[inds, col]

    def _make_df(self, q=None):
        if q:
            cat = self.catalog.query(q)
        else:
            cat = self.catalog
        x = self.xFunc(cat)
        y = self.yFunc(cat)
        label = self.labeller(cat)
        size = self.sizer(cat)
        
        ok = (np.isfinite(x) & np.isfinite(y))
        x = x[ok]
        y = y[ok]
        label = label[ok]
        size = size[ok]
        sourceId = np.array(catalog['id'])[ok]

        d = {'x':x, 'y':y, 'sourceId':sourceId, 'label':label, 'size':size}        
        return pd.DataFrame(d)    

    def _make_figure(self):
        df = self.df
        
        hover = HoverTool(
                        tooltips=[
                            ("index", "@index"),
                            ("(x,y)", "(@{0}, @{1})".format('x', 'y')),
                            ("label", "@label")
                        ]
                    )

        TOOLS=['pan','wheel_zoom','box_zoom','box_select','reset', hover]
        
        xlo, xhi = df.x.quantile([0., 0.99])
        ylo, yhi = df.y.quantile([0.001, 0.999])
        fig = figure(tools=TOOLS, x_range=(xlo, xhi), y_range=(ylo, yhi),
                        **self.figure_kwargs)
    
        size_scale = LinearInterpolator(x=[min(df['size']), max(df['size'])],
                                        y=[1,1])

        renderers = {}
        nonselected_circle = Circle(fill_alpha=0.1, fill_color="black", line_color=None)
        for label, color in zip(self.labels, self.palette):
            src = self.sources[label]        
            r = fig.circle('x', 'y', source=src, line_color=None,
                     color=color, legend=label,
                     size=dict(field='size', transform=size_scale), alpha=self.alpha)

            selected_circle = Circle(fill_alpha=0.6, fill_color=color, line_color=None)
            r.selection_glyph = selected_circle
            r.nonselection_glyph = nonselected_circle
            renderers[label] = r

        fig.legend.location='bottom_left'
        fig.legend.click_policy = 'hide'    

        fig.xaxis.axis_label = self.xFunc.name
        fig.yaxis.axis_label = self.yFunc.name

        self._renderers = renderers
        self._figure = fig

class QAHistogram(QAPlot):

    figure_kwargs = dict(toolbar_location=None, plot_width=300, plot_height=300,
                         y_axis_location=None, active_drag='xpan', active_scroll='xwheel_zoom')

    def __init__(self, scatter, axis, bins=20, **kwargs):
        """
        axis: 'x' or 'y'
        """

        self.scatter = scatter
        self.axis = axis

        self._bins = bins

        self._hist_cache = {k:None for k in self.labels}
        self._hist_cache_key = {k:None for k in self.labels}

        self._selected_cache_key = None


        super(QAHistogram, self).__init__(scatter.catalog, palette=scatter.palette,
                                          **kwargs)

    @property
    def sources(self):
        return self.scatter.sources

    @property
    def labels(self):
        return self.scatter.labels

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, new):
        self._bins = new
        self._reset()

    @property
    def df(self):
        return self.scatter.df

    @property
    def histograms(self):
        bins = self.bins

        hists = {}
        # none_selected = sum([len(self.scatter.selected_inds(l))
        #                          for l in self.labels]) == 0 
        # print(none_selected, [len(self.scatter.selected_inds(l))
        #                          for l in self.labels])

        for label in self.labels:
            inds = self.scatter.selected_inds(label)
            src = self.sources[label]

            if len(inds)==0:
                data = src.data[self.axis]
            else:
                data = src.data[self.axis][inds]
            ok = np.isfinite(data)
            data = data[ok]

            h, edges = np.histogram(data, bins=self.bins)
            h = h/h.max()
            hists[label] = (h, edges)

        return hists

    def _make_figure(self):
        df = self.df

        tools = ['xpan', 'xwheel_zoom']
        fig = figure(tools=tools, **self.figure_kwargs)
        

        renderers = {}
        hists = self.histograms
        for label, color in zip(self.labels, self.palette):
            hist, edges = hists[label]
            zeros = np.zeros(len(edges)-1)
            histmax = max(hist)*1.1
            h = fig.quad(bottom=0, left=edges[:-1], right=edges[1:], top=hist, color=color, 
                    line_color="#3A5785", legend=label, alpha=0.4)
            renderers[label] = h

        fig.legend.location='top_left'
        fig.legend.click_policy = 'hide'    

        if self.axis == 'x':
            axis_label = self.scatter.xFunc.name
        else:
            axis_label = self.scatter.yFunc.name
        fig.xaxis.axis_label = axis_label

        self._renderers = renderers
        self._figure = fig

    def update_histogram(self, label):
        hist, edges = self.histograms[label]
        hist = hist / hist.max()
        self.renderers[label].data_source.data['top'] = hist
        self.renderers[label].data_source.data['left'] = edges[:-1]
        self.renderers[label].data_source.data['right'] = edges[1:]


class QASkyPlot(QAScatterPlot):
    def __init__(self, catalog, **kwargs):
        super(QASkyPlot, self).__init__(catalog, xFunc=RAColumn(), yFunc=DecColumn(), **kwargs)


catalog = pd.read_hdf('data/forced.h5', 'df')
xFunc = Mag('base_PsfFlux')
yFunc = MagDiff('modelfit_CModel', 'base_PsfFlux')

s = QAScatterPlot(catalog, xFunc=xFunc, yFunc=yFunc)
hx = QAHistogram(s, 'x')
hy = QAHistogram(s, 'y')

sky = QASkyPlot(catalog)

l = layout([[s.figure, sky.figure], [hx.figure, hy.figure]])

curdoc().add_root(l)

def update_histograms(label):
    def update(attr, old, new):
        for h in [hx, hy]:
            h.update_histogram(label)
    return update

for label, r in s.renderers.items():
    r.data_source.on_change('selected', update_histograms(label))

