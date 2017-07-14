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
from bokeh.models import (BoxSelectTool, LassoSelectTool, ColorBar,
                        Spacer, ColumnDataSource, Range1d, HoverTool)
from bokeh.models.widgets import Slider, Button, DataTable, TableColumn, NumberFormatter
from bokeh.models.widgets import Panel, Tabs, Select, RadioButtonGroup, TextInput, PreText
from bokeh.models.mappers import LinearColorMapper
# from bokeh.models.tickers import LinearTicker
from bokeh.charts import Scatter
from bokeh.plotting import figure, curdoc
from bokeh.palettes import Spectral4, Category10, Dark2

from utils import Mag, MagDiff, StarGalaxyLabeller, Sizer
from utils import RAColumn, DecColumn
from utils import DeconvolvedMoments, TestFunctor

from bokeh.plotting.figure import Figure
from bokeh.models import CategoricalColorMapper, ColumnDataSource, LinearInterpolator
from bokeh.models import Range1d, Circle
from bokeh.layouts import widgetbox, layout, row, column
from bokeh.models.widgets import Slider

import logging

class QAPlot(object):
    def __init__(self, catalog, labeller=StarGalaxyLabeller(), palette=Category10[10],
                 include_labels=None, **kwargs):
        self._catalog = catalog
        self._full_catalog = catalog.copy()

        self.palette = palette
        self._include_labels = include_labels
        self.children = []

        self._df = None
        self._query = None
        
        # This will be a dict, with keys as labels
        self._sources = None

        # Figures, etc.
        self._figure = None
        self._renderers = None
        self._empty_selections = None

        # Functors
        self._labeller = labeller

        self._kwargs = kwargs

    @property
    def catalog(self):
        return self._catalog

    @catalog.setter
    def catalog(self, new):
        self._catalog = new
        self._reset()

    def query_catalog(self, query):
        if not query:
            self.reset_catalog()
        else:
            self.catalog = self._full_catalog.query(query)
            self._query = query

    def reset_catalog(self):
        self.catalog = self._full_catalog.copy()

    def _reset(self):
        self._df = None
        self._query = None
        self.clear_selections()
        self._update_sources()
        try:
            self._update_figure()
        except NotImplementedError: # clean up so this is not necessary!
            pass
        for p in self.children:
            p._reset()

        # for p in self.children:
        #     try:
        #         p._update_figure()
        #     except NotImplementedError: # temporary...
        #         pass
        
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
    def include_labels(self):
        if self._include_labels is None:
            return self.labels
        else:
            return self._include_labels

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

    def clear_selections(self):
        if self._empty_selections is not None:
            for l, sel in self._empty_selections.items():
                self._renderers[l].data_source.selected = sel

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
        kws.update(self._kwargs)
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
        inds = self.renderers[label].data_source.selected['1d']['indices']
        logging.debug('{}: {} indices selected'.format(label, len(inds)))
        return inds

    def selected_column(self, label, col, allow_empty=False):
        logging.debug('{}: getting {} column...'.format(label, col))
        if label=='all':
            data = np.array([])
            for l in self.labels:
                inds = self.selected_inds(l)
                src = self.renderers[l].data_source
                if len(inds)==0:
                    data = np.concatenate([data, np.array(src.data[col])])
                else:   
                    data = np.concatenate([data, np.array(src.data[col])[inds]])
        else:
            inds = self.selected_inds(label)
            src = self.renderers[label].data_source
            data = np.array(src.data[col])

            # There's a bug with 'allow_empty', so cutting it for now.
            if len(inds) > 0:# or allow_empty:
                # if allow_empty:
                #     logging.info('{}: allowing empty selection for {}'.format(label, col))
                data = data[inds]

        # logging.debug(data)
        return data

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
        x = self.df[self._xCol]
        y = self.df[self._yCol]
        xlo, xhi = x.quantile([0., 0.99])
        ylo, yhi = y.quantile([0.01,0.99])        
        xBuffer = np.std(x)/4.
        yBuffer = np.std(y)/4.
        xlo -= xBuffer
        xhi += xBuffer
        ylo -= yBuffer
        yhi += yBuffer
        return (xlo, xhi), (ylo, yhi)

    def _make_figure(self, title=True):
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
            if label not in self.include_labels:
                continue
            src = self.sources[label]        
            r = fig.circle(self._xCol, self._yCol, source=src, line_color=None,
                     color=color, legend=label,
                     # size=dict(field='size', transform=size_scale), 
                     size='size',
                     alpha=self.alpha)

            selected_circle = Circle(fill_alpha=self.alpha, fill_color=color, line_color=None)
            r.selection_glyph = selected_circle
            r.nonselection_glyph = nonselected_circle
            renderers[label] = r

        fig.legend.location='bottom_left'
        fig.legend.click_policy = 'hide'    

        if len(renderers) == 1:
            fig.legend.visible = False

        fig.xaxis.axis_label = self.xLabel
        fig.yaxis.axis_label = self.yLabel

        self._empty_selections = {l:r.data_source.selected for l,r in renderers.items()}

        self._renderers = renderers
        self._figure = fig
        if title:
            self.update_title()

    def update_title(self):
        n_sources = [len(self.selected_column(l, 'y')) for l in self.include_labels]
        total = sum(n_sources)
        title = '{} objects selected: '.format(total)
        for l,n in zip(self.labels, n_sources):
            title += '{0} {1}, '.format(n, l)
        title = title[:-2]
        self.figure.title.text = title

    def _update_figure(self):
        self._figure.xaxis.axis_label = self.xLabel
        self._figure.xaxis.axis_label = self.xLabel
        x_range, y_range = self._get_default_range()
        self._figure.x_range.start = x_range[0]
        self._figure.x_range.end = x_range[1]
        self._figure.y_range.start = y_range[0]
        self._figure.y_range.end = y_range[1]
        self.update_title()

class ChildQAPlot(QAPlot):
    def __init__(self, parent, **kwargs):
        self.parent = parent
        super(ChildQAPlot, self).__init__(parent.catalog, **kwargs)
        self.parent.children.append(self)

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

    _figure_kwargs = dict(toolbar_location=None, plot_width=400, plot_height=300,
                         y_axis_location=None, active_drag='xpan', active_scroll='xwheel_zoom')

    def __init__(self, parent, axis, bins='auto', **kwargs):
        """
        axis: 'x' or 'y'
        """

        self.parent = parent
        self.axis = axis

        self._bins = bins
        self._hists = None

        super(QAHistogram, self).__init__(parent, **kwargs)

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, new):
        self._bins = new
        self._reset()

    def _calc_histogram(self, label):
        if self._hists is None:
            self._hists = {}

        n_selected = np.array([len(self.parent.selected_inds(l))
                        for l in self.labels if l != 'all'])
        allow_empty = False
        # print(n_selected, np.any(n_selected==0), np.all(n_selected==0))
        if np.any(n_selected==0) and not np.all(n_selected==0):
            allow_empty = True
            # print('should allow empty...')

        data = self.parent.selected_column(label, self.axis, allow_empty=allow_empty)
        ok = np.isfinite(data)
        data = data[ok].copy()
        if len(data) > 0:
            logging.debug('data range for {} {}-axis: {}, {}'.format(label, self.axis, data.min(), data.max()))
        logging.debug('{}: {} data points selected.'.format(label, len(data)))
        logging.debug(data)

        h, edges = np.histogram(data, bins=self.bins)
        h = h/h.max()
        self._hists[label] = (h, edges)
        return self._hists[label]

    @property
    def histograms(self):
        if self._hists is None:
            bins = self.bins

            hists = {}

            for label in self.labels:
                self._calc_histogram(label)

        return self._hists

    @property
    def labels(self):
        return list(self.parent.labels) + ['all']

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

        renderers['all'].visible = False

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
        logging.debug('updating {} for {} axis....'.format(label, self.axis))
        self._calc_histogram(label)
        hist, edges = self.histograms[label]
        hist = hist / hist.max()
        self.renderers[label].data_source.data = {'top' : hist,
                                                  'left' : edges[:-1],
                                                  'right' : edges[1:]}

    def _set_auto_range(self):
        if self.axis=='x':
            rng = self.parent._figure.x_range
        elif self.axis=='y':
            rng = self.parent._figure.y_range

        minval = rng.start
        maxval = rng.end

        self._figure.x_range.start = minval
        self._figure.x_range.end = maxval

    def _update_figure(self):
        for l in self.labels:
            self.update_histogram(l)
        self._set_auto_range()

class QASkyPlot(ChildQAPlot, QAScatterPlot):
    _figure_kwargs = dict(plot_width=400, plot_height=400)
    _xCol = 'ra'
    _yCol = 'dec'

    def _get_default_range(self):
        xlo = np.rad2deg(self._full_catalog['coord_ra'].min())
        xhi = np.rad2deg(self._full_catalog['coord_ra'].max())
        ylo = np.rad2deg(self._full_catalog['coord_dec'].min())
        yhi = np.rad2deg(self._full_catalog['coord_dec'].max())

        return (xlo, xhi), (ylo, yhi)

    def _get_color_range(self):
        # First check to see if there's a selection
        ylo = np.inf
        yhi = -np.inf

        for l in self.labels:
            if l not in self.include_labels:
                continue
            try:
                data = self.selected_column(l, 'y', allow_empty=True)
                ylo = min(data.min(), ylo)
                yhi = max(data.max(), yhi)
            except ValueError:
                continue

        axis_lo = self.parent.figure.y_range.start
        axis_hi = self.parent.figure.y_range.end

        lo = max(ylo, axis_lo) if np.isfinite(ylo) else axis_lo
        hi = min(yhi, axis_hi) if np.isfinite(yhi) else axis_hi
        return lo, hi
        # return self.df['y'].min(), self.df['y'].max()
        # return self.df['y'].quantile([0.01, 0.99])

    def _make_figure(self, **kwargs):
        super(QASkyPlot, self)._make_figure(title=False, **kwargs)

        lo, hi = self._get_color_range()
        self.color_mapper = LinearColorMapper(palette='Viridis256', low=lo, high=hi)
        for label, r in self.renderers.items():
            r.glyph.fill_color = dict(field='y', transform=self.color_mapper)
            r.selection_glyph.fill_color = dict(field='y', transform=self.color_mapper)

        color_bar = ColorBar(color_mapper=self.color_mapper, #ticker=LinearTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))

        self._figure.add_layout(color_bar, 'right')

    def update_title(self):
        pass

    def update_color(self):
        lo, hi = self._get_color_range()
        self.color_mapper.low = lo
        self.color_mapper.high = hi

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

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

forced = pd.read_hdf('data/forced.h5', 'df')
unforced = pd.read_hdf('data/unforced.h5', 'df')
for c in ['coord_ra', 'coord_dec']:
    print(c, forced[c].describe())
# catalog = catalog.sample(50)

xFuncs = {'base_PsfFlux' : Mag('base_PsfFlux'),
          'modelfit_CModel' : Mag('modelfit_CModel')}
yFuncs = {'modelfit_CModel - base_PsfFlux' : MagDiff('modelfit_CModel', 'base_PsfFlux'),
          'Deconvolved Moments' : DeconvolvedMoments(),
          'test' : TestFunctor()}

xFunc = xFuncs['base_PsfFlux']
yFunc = yFuncs['modelfit_CModel - base_PsfFlux']

s = QAScatterPlot(forced, xFunc=xFunc, yFunc=yFunc, plot_width=600)
hx = QAHistogram(s, 'x')
hy = QAHistogram(s, 'y')
sky_plots = [QASkyPlot(s, unselected_alpha=0., size=2, include_labels=[l])
                for l in s.labels]
sky_tabs = Tabs(tabs=[Panel(child=sky.figure, title=l) for sky, l in zip(sky_plots, s.labels)])
table = QATable(s)

radio_button_group = RadioButtonGroup(labels=['forced', 'unforced'], active=0)

# print(s.figure_kwargs)

size_slider = Slider(start=1, end=10, step=1, value=1, title="Circle Size")
alpha_slider = Slider(start=0, end=1, step=0.01, value=0.8, title='alpha')

x_select = Select(title="X-axis:", value='base_PsfFlux', options=xFuncs.keys())
y_select = Select(title="Y-axis:", value='modelfit_CModel - base_PsfFlux', options=yFuncs.keys())

query_box = TextInput(value='', title="Query")
query_pretext = PreText(text='', width=600, height=20)

def update_catalog(attr, old, new):
    if new==0:
        s.catalog = forced
        s.query_catalog(query_box.value)
    elif new==1:
        s.catalog = unforced
        s.query_catalog(query_box.value)
    elif isinstance(new, basestring): # this means entering a query
        # print('query: "{}"'.format(new))
        # print('length of catalog before query: {}'.format(len(s.catalog)))
        s.query_catalog(new)
        # print('length of catalog after query: {}'.format(len(s.catalog)))
        query_pretext.text = new
        
    for l in s.labels:
        update_histograms(l)
    update_sky_colormap(attr, old, new)
    s.update_title()

def update_radius(attr, old, new):
    for _, src in s.sources.items():
        src.data['size'] = np.ones(len(src.data['size']))*new

def update_alpha(attr, old, new):
    renderers = s.renderers.values()
    for sky in sky_plots:
        renderers += sky.renderers.values()
        
    for r in renderers:
        r.glyph.fill_alpha = new
        r.selection_glyph.fill_alpha = new

def update_sky_colormap(attr, old, new):
    for sky in sky_plots:
        sky.update_color()

def update_yFunc(attr, old, new):
    s.yFunc = yFuncs[new]
    for label in s.labels:
        for h in [hx, hy]:
            h.update_histogram(label)
        table.update_sources(label)
    s.update_title()

def update_xFunc(attr, old, new):
    s.xFunc = xFuncs[new]
    for label in s.labels:
        for h in [hx, hy]:
            h.update_histogram(label)
        table.update_sources(label)
    s.update_title()

radio_button_group.on_change('active', update_catalog)
query_box.on_change('value', update_catalog)
size_slider.on_change('value', update_radius)
alpha_slider.on_change('value', update_alpha)
y_select.on_change('value', update_yFunc)
x_select.on_change('value', update_xFunc)
s.figure.y_range.on_change('start', update_sky_colormap)
s.figure.y_range.on_change('end', update_sky_colormap)

top_widgetbox = widgetbox(children=[radio_button_group, query_box]) 
l = layout([[column(top_widgetbox, query_pretext, s.figure, width=600), sky_tabs], 
            [hx.figure, hy.figure, column([x_select, y_select]), column([size_slider, alpha_slider])], 
            [table.tabs]])

curdoc().add_root(l)


def update_histograms(label):
    def update(attr, old, new):
        for h in [hx, hy]:
            h.update_histogram(label)
            h.update_histogram('all')
        table.update_sources(label)
        update_sky_colormap(attr, old, new)
    return update


for label, r in s.renderers.items():
    r.data_source.on_change('selected', update_histograms(label))
    r.data_source.on_change('selected', lambda attr, old, new: s.update_title())
