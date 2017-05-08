''' Present a scatter plot with linked histograms on both axes.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve selection_histogram.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/selection_histogram

in your browser.

'''

import numpy as np

from bokeh.layouts import row, column
from bokeh.models import (BoxSelectTool, LassoSelectTool, 
                        Spacer, ColumnDataSource, Range1d, HoverTool)
from bokeh.plotting import figure, curdoc



from fake_data import simulate_data

df = simulate_data(5000)
x = df.x
y = df.y

NBINS = 20

hover = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]
    )

TOOLS=['pan','wheel_zoom','xbox_select','reset', hover]

# create the scatter plot
p = figure(tools=TOOLS, plot_width=600, plot_height=600, min_border=10, min_border_left=50,
           toolbar_location="above", x_axis_location="below", y_axis_location="right",
           title="Linked Histograms", active_drag='xbox_select', active_scroll='wheel_zoom')
p.background_fill_color = "#fafafa"
# p.select(BoxSelectTool).select_every_mousemove = False
#p.select(LassoSelectTool).select_every_mousemove = False


r = p.scatter(x, y, size=3, color="#3A5785", alpha=0.6)

# create the horizontal histogram
hhist, hedges = np.histogram(x, bins=NBINS, normed=True)
hzeros = np.zeros(len(hedges)-1)
hmax = max(hhist)*1.1

LINE_ARGS = dict(color="#3A5785", line_color=None)

ph = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200, x_range=p.x_range,
            y_range=(0, hmax), min_border=10, min_border_left=50, y_axis_location=None,
            lod_factor=10, x_axis_location=None)
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"

ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="#3A5785")
hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
# hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)

# create the vertical histogram
vhist, vedges = np.histogram(y, bins=NBINS, normed=True)
vzeros = np.zeros(len(vedges)-1)
vmax = max(vhist)*1.1

pv = figure(toolbar_location=None, plot_width=200, plot_height=p.plot_height, x_range=(0, vmax),
            y_range=p.y_range, min_border=10, y_axis_location=None, x_axis_location=None)
pv.ygrid.grid_line_color = None
pv.xaxis.major_label_orientation = np.pi/4
pv.background_fill_color = "#fafafa"

pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="#3A5785")
vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
# vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)

layout = column(row(p, pv), row(ph, Spacer(width=200, height=200)))

curdoc().add_root(layout)
curdoc().title = "Selection Histogram"

def update(attr, old, new):
    inds = np.array(new['1d']['indices'])
    if len(inds) == 0 or len(inds) == len(x):
        hhist1 = hzeros
        vhist1 = vzeros
        vedges1 = vedges 
    else:
        neg_inds = np.ones_like(x, dtype=np.bool)
        neg_inds[inds] = False
        hhist1, _ = np.histogram(x[inds], bins=hedges, normed=True)
        vhist1, vedges1 = np.histogram(y[inds], bins=NBINS, normed=True)
        # hhist2, _ = np.histogram(x[neg_inds], bins=hedges, normed=True)
        # vhist2, vedges2 = np.histogram(y[neg_inds], bins=NBINS, normed=True)

    hh1.data_source.data["top"]   =  hhist1
    # hh2.data_source.data["top"]   = -hhist2
    vh1.data_source.data["right"] =  vhist1
    vh1.data_source.data["bottom"] = vedges1[:-1]
    vh1.data_source.data["top"] = vedges1[1:]
    # vh2.data_source.data["right"] = -vhist2
    # vh2.data_source.data["bottom"] = vedges2[:-1]
    # vh2.data_source.data["top"] = vedges2[1:]
    pv.x_range.start = 0
    pv.x_range.end = max(vhist1.max()*1.1, vmax)
    # cursession().store_objects(pv) 

r.data_source.on_change('selected', update)