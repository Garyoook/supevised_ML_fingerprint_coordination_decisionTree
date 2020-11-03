import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import numpy as np
import dt

colors = [mcolors.to_rgba(c)
          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

def visualise_dtree(dtree, depth):
    levels = depth
    width_dist = 200
    depth_dist = 200

    def dtree_level(levels, x, y, width):
        segments = []
        xl = x + depth_dist
        yl = y - width/2
        xr = x + depth_dist
        yr = y + width/2
        segments.append([[x, y], [xl, yl]])
        segments.append([[x, y], [xr, yr]])
        if levels > 1:
            segments+= dtree_level(levels - 1, xl, yl, width/2)
            segments+= dtree_level(levels - 1, xr, yr, width/2)
        return segments

    segs = dtree_level(levels, 0, 0, width_dist)
    line_segments = LineCollection(segs, linewidths=0.1, linestyles='solid')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=600)
    ax.set_xlim(-1, (levels * depth_dist + 1))
    ax.set_ylim(-1.5*width_dist, 1.5*width_dist)
    ax.add_collection(line_segments)
    plt.show()
# COMMENT: decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}




if __name__ == '__main__':
    inputfile = './wifi_db/clean_dataset.txt'
    training_data = np.loadtxt(inputfile)
    dtree, depth = dt.decision_tree_learning(training_data, 0)
    visualise_dtree(dtree, depth)

