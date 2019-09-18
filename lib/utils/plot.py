import numpy as np

#plt.show()

def plot_stats(ax, stats, portfolio_values):
    
    ax[0].plot(stats)
    ax[1].plot(portfolio_values)