from __future__ import print_function, division

import pandas as pd
import numpy as np

from bokeh.models  import ColumnDataSource

class Functor(object):
    def __call__(self, catalog):
        return np.array(self._func(catalog))

class Column(Functor):
    def __init__(self, col):
        self.col = col

    def _func(self, catalog):
        return catalog[self.col]

class CoordColumn(Column):
    def _func(self, catalog):
        return np.rad2deg(catalog[self.col])

class RAColumn(CoordColumn):
    col = 'coord_ra'

class DecColumn(CoordColumn):
    col = 'coord_dec'

def fluxName(col):
    if not col.endswith('_flux'):
        col += '_flux'
    return col

class Mag(Functor):
    def __init__(self, col):
        self.col = fluxName(col)

    def _func(self, catalog):
        return -2.5*np.log10(catalog[self.col])

    @property
    def name(self):
        return 'mag_{0}'.format(self.col)

class MagDiff(Functor):
    """Functor to calculate magnitude difference"""
    def __init__(self, col1, col2):
        self.col1 = fluxName(col1)
        self.col2 = fluxName(col2)

    def _func(self, catalog):
        return -2.5*np.log10(catalog[self.col1]/catalog[self.col2])

    @property
    def name(self):
        return '(mag_{0} - mag_{1})'.format(self.col1, self.col2)


class StarGalaxyLabeller(object):
    _column = "base_ClassificationExtendedness_value"
    def __call__(self, catalog):
        return np.where(catalog[self._column] < 0.5, 'star', 'galaxy')

class Sizer(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, catalog):
        return np.ones(len(catalog)) * self.size

def makeDataSource(catalog, xFunc, yFunc, labeller, sizer=Sizer(1), df=False):
    x = xFunc(catalog)
    y = yFunc(catalog)
    label = labeller(catalog)
    size = sizer(catalog)

    ok = (np.isfinite(x) & np.isfinite(y))
    x = x[ok]
    y = y[ok]
    label = label[ok]
    size = size[ok]
    sourceId = np.array(catalog['id'])[ok]

    d = {'x':x, 'y':y, 'sourceId':sourceId, 'label':label, 'size':size}
    if df:
        return pd.DataFrame(d)
    else:
        return ColumnDataSource(d)


