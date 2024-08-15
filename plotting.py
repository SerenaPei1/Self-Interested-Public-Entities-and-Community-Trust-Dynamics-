''' code for plotting networks and figs '''

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FixedLocator)
from matplotlib.lines import Line2D

def make_edge(x, y):
    return  go.Scatter(x         = x,
                       y         = y,
                       mode = 'lines',
                       line = dict(width = 10, color = 'gray'),
                       opacity = .5)

def graph_vis(G, colors):
    N = len(G.nodes())
    pos = nx.spring_layout(G, k = 1.2)
    edge_trace = []
    for edge in G.edges(data = True):
        char_1 = edge[0]
        char_2 = edge[1]
        x0, y0 = pos[char_1]
        x1, y1 = pos[char_2]
        trace  = make_edge([x0, x1, None], [y0, y1, None])
        edge_trace.append(trace)

    node_trace = go.Scatter(x         = [],
                            y         = [],
                            textposition = "top center",
                            textfont_size = 10,
                            hoverinfo = 'text',
                            hovertext = [],
                            text      = [],
                            mode      = 'markers+text',
                            marker    = dict(color = colors,
                                             size  = 50,
                                             symbol = 'circle',
                                             line=dict(color='white', width = 2)))
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    layout = go.Layout(xaxis = {'showgrid': False, 'zeroline': False},
                        yaxis = {'showgrid': False, 'zeroline': False},
                        autosize = False,
                        width = 1000,
                        height = 1000,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

    # Create figure
    fig = go.Figure(layout = layout)
    for trace in edge_trace:
        fig.add_trace(trace)
    fig.add_trace(node_trace)
    fig.update_layout(showlegend = False)
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    fig.show()

def vis_G(G):

    N = len(list(G.nodes()))

    types = []
    color = []

    for node in G.nodes():
        types += tuple(([G.nodes()[node]['type']]))

    for i in range(len(types)):
        if types[i] == 'orange':
            color.append('rgb(245, 182, 66)')
        elif types[i] == 'blue':
            color.append('rgb(66, 153, 245)')

    graph_vis(G, color)

def heat_map(arr, dim1, dim2, type, title, save):

    avg_arr = np.mean(arr, axis = 2)

    data = np.rot90(avg_arr)

    x_tick_labels = np.round(dim1,2)
    y_tick_labels = dim2[::-1]
    ax = None

    if type == 'apl' :
        ax = sns.heatmap(data, vmin = 0, vmax = 1,
                 annot=True, cbar=True, square=True,
                 xticklabels = x_tick_labels, yticklabels = y_tick_labels, linewidth=0.5)

    if type == 'triangles':
        ax = sns.heatmap(data, vmin = 0, vmax = .15,
                 annot=True, cbar=True, square=True,
                 xticklabels = x_tick_labels, yticklabels = y_tick_labels, linewidth=0.5)

    if type == 'num_prop':
        ax = sns.heatmap(data, vmin = 0,
                 annot=True, cbar=True, square=True,
                 xticklabels = x_tick_labels, yticklabels = y_tick_labels, linewidth=0.5)

    if type == 'spent':
        ax = sns.heatmap(data, vmin = 0, vmax = 150,
                 annot=True, cbar=True, square=True,
                 xticklabels = x_tick_labels, yticklabels = y_tick_labels, linewidth=0.5, fmt='.2f')

    ax.set_xlabel('Resource Constraint')
    ax.set_ylabel('Agent Trust')
    ax.tick_params(axis='both', which='major', labelsize=10)

    cbar_axes = ax.figure.axes[-1]

    if save:
        title_save = 'figs/' + title +'.pdf'
        plt.savefig(title_save, dpi = 300, bbox_inches = 'tight')
        plt.close('all')
    else:
        plt.show()

def surface_plot(arr, dim1, dim2, type, title, save):

    avg_arr = np.mean(arr, axis = 2)

    data = np.rot90(avg_arr)


    n = len(dim1)
    m = len(dim2)

    tau = np.repeat(dim2, n)[::-1]

    rho = (np.tile(dim1, m))[::-1]

    data = np.fliplr(data)

    df = pd.DataFrame({"rho":rho.reshape(n*m,), "tau":tau.reshape(n*m,), "tri":data.reshape(n*m,)}, index=range(0,n*m))

    #print(df)

    #print(df.head(27))

    X, y = df[["rho", "tau"]], df["tri"]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    print(poly_reg_model.score(poly_features, y))

    i = poly_reg_model.intercept_
    c = poly_reg_model.coef_

    print(i,c)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    tau = np.arange(0, 1.00, 1/100)
    rho = np.arange(0, 2, 1/100)

    R,T = np.meshgrid(rho, tau)
    tri = i + c[0]*R + c[1]*T + c[2]*(R**2) + c[3]*(T**2) + c[4]*T*R

    # 0.012657288401117527 [-0.01259024 -0.06186663 -0.0001596   0.06530567  0.0585887 ]


    surf = ax.plot_surface(R, T, tri, cmap = sns.cm.rocket ,vmin =0, vmax = .2,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink = .4)

    ax.view_init(10, 230)
    #ax.view_init(270, 0)

    ax.set_xlabel('Resource Constraint')
    ax.set_ylabel('Agent Trust')
    ax.tick_params(axis='both', which='major', labelsize=10)
    if save:
        title_save = 'figs/' + title +'.pdf'
        plt.savefig(title_save, dpi = 300, bbox_inches = 'tight')
        plt.close('all')
    else:
        plt.show()

def stacked_lines(arr_big, dim1, dim2, title, type, save):

    fig, ax = plt.subplots()

    for i in range(5):

        arr = arr_big[i]

        avg_arr = np.mean(arr, axis = 2)
        std_arr = np.std(arr, axis = 2)

        avg_data = np.rot90(avg_arr)
        std_data = np.rot90(std_arr)

        n = len(dim1)
        m = len(dim2)

        tau = np.repeat(dim2, n)[::-1]

        rho = (np.tile(dim1, m))[::-1]

        avg_data = np.fliplr(avg_data)
        std_data = np.fliplr(std_data)


        df = pd.DataFrame({"rho":rho.reshape(n*m,), "tau":tau.reshape(n*m,), "avg":avg_data.reshape(n*m,), "std":std_data.reshape(n*m,)}, index=range(0,n*m))

        #print(df)

        rho_df = df.loc[df['rho'] == 1.0]

        ax.plot(rho_df['tau'], rho_df['avg'])
        ax.fill_between(rho_df['tau'], rho_df['avg'] - rho_df['std'], rho_df['avg'] + rho_df['std'], alpha=0.25)

        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['bottom'].set_position(('outward'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=2, color='gray')
    ax.set_xlabel("Average agent trust",  fontsize=13)

    if type == 'triangles':
        ax.plot(rho_df['tau'], [.0053, .0071, .0095, .029, .066, .094, .12][::-1], color = 'black', linestyle = 'dotted')
        ax.set_ylabel("Fairness", fontsize=13)
        custom_lines = [Line2D([0], [0], color='C0', lw=2),
                        Line2D([0], [0], color='C1', lw=2),
                        Line2D([0], [0], color='C2', lw=2),
                        Line2D([0], [0], color='C3', lw=2),
                        Line2D([0], [0], color='C4', lw=2),
                        Line2D([0], [0], color='black', lw=2, ls = 'dotted')]


        ax.legend(custom_lines, ['q = 1/5', 'q = 2/5', 'q = 3/5', 'q = 4/5', 'q = 5/5', 'without intervention'])

    if type == 'new_trust':
        ax.plot(rho_df['tau'], rho_df['tau'], color = 'black', linestyle = 'dotted')
        ax.set_ylabel("New average agent trust", fontsize=13)

    if type == 'spent':
        ax.plot(rho_df['tau'], np.zeros((7)), color = 'black', linestyle = 'dotted')
        ax.set_ylabel("Dollar amount spent",  fontsize=13)


    #plt.show()
        #print(rho_df)

            #ax.plot(m.Time, m.Mean)
            #ax.fill_between(m.Time, m.Mean - m.Std, m.Mean + m.Std, alpha=0.35)
    #
    if save:
        title_save = 'figs/' + title +'.pdf'
        plt.savefig(title_save, dpi = 300, bbox_inches = 'tight')
        plt.close('all')
    else:
        plt.show()
