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
from bokeh.models.widgets import Slider, Button, DataTable, TableColumn, NumberFormatter
from bokeh.models.widgets import Panel, Tabs, Select
from bokeh.charts import Scatter
from bokeh.plotting import figure, curdoc
from bokeh.palettes import Spectral4, Category10, Dark2

from utils import Mag, MagDiff, StarGalaxyLabeller, Sizer
from utils import RAColumn, DecColumn
from utils import DeconvolvedMoments

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
        self._df = None
        self._update_sources()
        self._update_figure()
        
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

    def _update_sources(self):
        for k, src in self.sources.items():
            src.data = src.from_df(self.df.query('label=="{0}"'.format(k)))

    def _update_figure(self):
        raise NotImplementedError

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

    @property
    def figure_kwargs(self):
        try:
            kws = super(type(self), self)._figure_kwargs
        except AttributeError:
            kws = {}

        kws.update(self._figure_kwargs)
        return kws

class QAScatterPlot(QAPlot):
    _figure_kwargs = dict(active_scroll='wheel_zoom', 
                         active_drag='box_select',
                         plot_width=600, plot_height=400,
                         toolbar_location="right",
                         toolbar_sticky=False)
    _xCol = 'x'
    _yCol = 'y'
    _default_columns = ('x', 'y', 'id', 'label', 'ra', 'dec', 'size')


    def __init__(self, catalog, xFunc=None, yFunc=None, sizer=Sizer(1), 
                alpha=0.8, unselected_alpha=0.1, size=1., 
                columns=None,
                **kwargs):

        self._xFunc = xFunc
        self._yFunc = yFunc
        self._sizer = sizer

        if columns is None:
            self._columns = list(self._default_columns)

        self.alpha = alpha
        self.unselected_alpha = unselected_alpha
        self.size = size

        super(QAScatterPlot, self).__init__(catalog, **kwargs)

    @property
    def columns(self):
        return self._columns

    @property
    def minSize(self):
        try:
            return self.size[0]
        except:
            return self.size

    @property
    def maxSize(self):
        try:
            return self.size[1]
        except:
            return self.size

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
    def xLabel(self):
        if self.xFunc is None:
            return self._xCol
        else:
            return self.xFunc.name

    @property
    def yLabel(self):
        if self.yFunc is None:
            return self._yCol
        else:
            return self.yFunc.name

    @property
    def sizer(self):
        return self._sizer
    
    @sizer.setter
    def sizer(self, f):
        self._sizer = f
        self._reset()

    def selected_inds(self, label):
        return self.renderers[label].data_source.selected['1d']['indices']

    def selected_column(self, label, col, allow_empty=False):
        inds = self.selected_inds(label)
        src = self.renderers[label].data_source
        if len(inds)==0 and not allow_empty:
            return src.data[col]
        else:
            return src.data[col][inds]

    def _make_df(self, q=None):
        if q:
            cat = self.catalog.query(q)
        else:
            cat = self.catalog.copy()
        if self.xFunc is None:
            x = np.ones(len(cat))
            y = np.ones(len(cat))
        else:
            x = self.xFunc(cat)
            y = self.yFunc(cat)
        label = self.labeller(cat)
        size = self.sizer(cat)
        ra = RAColumn()(cat)
        dec = DecColumn()(cat)

        cat['x'] = x
        cat['y'] = y
        cat['label'] = label
        cat['size'] = size
        cat['ra'] = ra
        cat['dec'] = dec

        return cat[np.isfinite(cat['x']) & np.isfinite(cat['y'])][self.columns]

        # ok = (np.isfinite(x) & np.isfinite(y))
        # x = x[ok]
        # y = y[ok]
        # ra = ra[ok]
        # dec = dec[ok]
        # label = label[ok]
        # size = size[ok]
        # sourceId = np.array(catalog['id'])[ok]

        # d = {'x':x, 'y':y, 'sourceId':sourceId, 'label':label, 'size':size,
        #      'ra':ra, 'dec':dec}        
        # return pd.DataFrame(d)    

    def _get_default_range(self):
        xlo, xhi = self.df[self._xCol].quantile([0., 0.99])
        ylo, yhi = self.df[self._yCol].quantile([0.01,0.99])        
        return (xlo, xhi), (ylo, yhi)

    def _make_figure(self):
        df = self.df
        
        hover = HoverTool(
                        tooltips=[
                            ("index", "@index"),
                            ("(x,y)", "(@{0}, @{1})".format('x', 'y')),
                            ("label", "@label")
                        ]
                    )

        TOOLS=['pan','wheel_zoom','box_zoom','box_select','lasso_select','reset', hover]
        
        x_range, y_range = self._get_default_range()
        fig = figure(tools=TOOLS, x_range=x_range, y_range=y_range,
                        **self.figure_kwargs)
    
        size_scale = LinearInterpolator(x=[min(df['size']), max(df['size'])],
                                        y=[self.minSize, self.maxSize])

        renderers = {}
        nonselected_circle = Circle(fill_alpha=self.unselected_alpha, 
                                    fill_color="black", line_color=None)
        for label, color in zip(self.labels, self.palette):
            src = self.sources[label]        
            r = fig.circle(self._xCol, self._yCol, source=src, line_color=None,
                     color=color, legend=label,
                     # size=dict(field='size', transform=size_scale), 
                     size='size',
                     alpha=self.alpha)

            selected_circle = Circle(fill_alpha=0.6, fill_color=color, line_color=None)
            r.selection_glyph = selected_circle
            r.nonselection_glyph = nonselected_circle
            renderers[label] = r

        fig.legend.location='bottom_left'
        fig.legend.click_policy = 'hide'    

        fig.xaxis.axis_label = self.xLabel
        fig.yaxis.axis_label = self.yLabel

        self._renderers = renderers
        self._figure = fig

    def _update_figure(self):
        self._figure.xaxis.axis_label = self.xLabel
        self._figure.xaxis.axis_label = self.xLabel
        x_range, y_range = self._get_default_range()
        self._figure.x_range.start = x_range[0]
        self._figure.x_range.end = x_range[1]
        self._figure.y_range.start = y_range[0]
        self._figure.y_range.end = y_range[1]


class ChildQAPlot(QAPlot):
    def __init__(self, parent, **kwargs):
        self.parent = parent
        super(ChildQAPlot, self).__init__(parent.catalog, **kwargs)

    @property
    def sources(self):
        return self.parent.sources

    @property
    def df(self):
        return self.parent.df

    @property
    def labels(self):
        return self.parent.labels


class QAHistogram(ChildQAPlot):

    _figure_kwargs = dict(toolbar_location=None, plot_width=300, plot_height=300,
                         y_axis_location=None, active_drag='xpan', active_scroll='xwheel_zoom')

    def __init__(self, parent, axis, bins='auto', **kwargs):
        """
        axis: 'x' or 'y'
        """

        self.parent = parent
        self.axis = axis

        self._bins = bins

        super(QAHistogram, self).__init__(parent, **kwargs)

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, new):
        self._bins = new
        self._reset()

    @property
    def histograms(self):
        bins = self.bins

        hists = {}
        # none_selected = sum([len(self.scatter.selected_inds(l))
        #                          for l in self.labels]) == 0 
        # print(none_selected, [len(self.scatter.selected_inds(l))
        #                          for l in self.labels])
        n_selected = np.array([len(self.parent.selected_inds(l))
                        for l in self.labels])
        allow_empty = False
        # print(n_selected, np.any(n_selected==0), np.all(n_selected==0))
        if np.any(n_selected==0) and not np.all(n_selected==0):
            allow_empty = True
            # print('should allow empty...')

        for label in self.labels:
            data = self.parent.selected_column(label, self.axis, allow_empty=allow_empty)
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
            axis_label = self.parent.xLabel
        else:
            axis_label = self.parent.yLabel
        fig.xaxis.axis_label = axis_label

        self._renderers = renderers
        self._figure = fig

    def update_histogram(self, label):
        hist, edges = self.histograms[label]
        hist = hist / hist.max()
        self.renderers[label].data_source.data['top'] = hist
        self.renderers[label].data_source.data['left'] = edges[:-1]
        self.renderers[label].data_source.data['right'] = edges[1:]

class QASkyPlot(ChildQAPlot, QAScatterPlot):
    _figure_kwargs = dict(plot_width=400, plot_height=400)
    _xCol = 'ra'
    _yCol = 'dec'

class QATable(object):
    def __init__(self, parent, skip=('size',)):
        self.parent = parent # QAScatterPlot

        self.skip = skip

        self._sources = None
        self._tabs = None

    @property
    def labels(self):
        return self.parent.labels

    @property
    def sources(self):
        if self._sources is None:
            self._sources = {l: ColumnDataSource(self.parent.sources[l].data) for l in self.labels}
        return self._sources

    def update_sources(self, label):
        inds = self.parent.selected_inds(label)
        src = self.parent.sources[label]

        self._sources[label].data = {k:np.array(src.data[k])[inds] for k in src.data.keys()}

    def _make_tabs(self):
        columns = [TableColumn(field=c, title=c) for c in self.parent.columns if c not in self.skip]
        data_tables = {l: DataTable(source=self.sources[l], columns=columns, width=800)
                        for l in self.labels}
        self._tabs = Tabs(tabs=[Panel(child=data_tables[l], title=l) for l in self.labels])


    @property
    def tabs(self):
        if self._tabs is None:
            self._make_tabs()
        return self._tabs


catalog = pd.read_hdf('data/forced.h5', 'df')

xFuncs = {'base_PsfFlux' : Mag('base_PsfFlux'),
          'modelfit_CModel' : Mag('modelfit_CModel')}
yFuncs = {'modelfit_CModel - base_PsfFlux' : MagDiff('modelfit_CModel', 'base_PsfFlux'),
          'Deconvolved Moments' : DeconvolvedMoments()}
xFunc = xFuncs['base_PsfFlux']
yFunc = yFuncs['modelfit_CModel - base_PsfFlux']

s = QAScatterPlot(catalog, xFunc=xFunc, yFunc=yFunc)
hx = QAHistogram(s, 'x')
hy = QAHistogram(s, 'y')
sky = QASkyPlot(s, unselected_alpha=0., size=2)
table = QATable(s)

size_slider = Slider(start=1, end=10, step=1, value=1, title="Circle Size")
alpha_slider = Slider(start=0, end=1, step=0.01, value=0.8, title='alpha')

x_select = Select(title="X-axis:", value='base_PsfFlux', options=xFuncs.keys())
y_select = Select(title="Y-axis:", value='modelfit_CModel - base_PsfFlux', options=yFuncs.keys())

def update_radius(attr, old, new):
    for _, src in s.sources.items():
        src.data['size'] = np.ones(len(src.data['size']))*new

def update_alpha(attr, old, new):
    for r in s.renderers.values() + sky.renderers.values():
        r.glyph.fill_alpha = new

def update_yFunc(attr, old, new):
    s.yFunc = yFuncs[new]
    for label in s.labels:
        for h in [hx, hy]:
            h.update_histogram(label)
        table.update_sources(label)

def update_xFunc(attr, old, new):
    s.xFunc = xFuncs[new]
    for label in s.labels:
        for h in [hx, hy]:
            h.update_histogram(label)
        table.update_sources(label)

size_slider.on_change('value', update_radius)
alpha_slider.on_change('value', update_alpha)
y_select.on_change('value', update_yFunc)
x_select.on_change('value', update_xFunc)

# skip = ('x', 'y', 'size')
# columns = [TableColumn(field=c, title=c) for c in s.columns if c not in skip]
# data_tables = {l: DataTable(source=s.sources[l], columns=columns, width=800)
#             for l in s.labels}

# tabs = Tabs(tabs=[Panel(child=data_tables[l], title=l) for l in s.labels])

l = layout([[s.figure, sky.figure, column([size_slider, alpha_slider])], 
            [hx.figure, hy.figure, column([x_select, y_select])], 
            [table.tabs]])

curdoc().add_root(l)


def update_histograms(label):
    def update(attr, old, new):
        for h in [hx, hy]:
            h.update_histogram(label)
        table.update_sources(label)
    return update


for label, r in s.renderers.items():
    r.data_source.on_change('selected', update_histograms(label))

