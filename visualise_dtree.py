import sys
import matplotlib.pyplot as plt
import numpy as np
import dt

bbox_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# get the number of all leaves in a decision tree
def get_leaf_number(decision_tree):
    cnt_leaf = 0
    if decision_tree['leaf']:
        return 1
    else:
        cnt_leaf += get_leaf_number(decision_tree['left']) + get_leaf_number(decision_tree['right'])
    return cnt_leaf


# plot the box of a single node.
def plot_node(attribute, coordinate_centre, coordinate_parent, arrow_bbox):
    visualise_decision_tree.ax1.annotate(attribute, xytext=coordinate_centre, textcoords="axes fraction",
                                         xy=coordinate_parent, xycoords="axes fraction",
                                         va="bottom", ha="center", bbox=arrow_bbox, arrowprops=arrow_args)


# plot the arrow
def plot_text_on_arrow(coordinate_child, coordinate_parent, text):
    '''
    :param coordinate_child: as name
    :param coordinate_parent: as name
    :param text: the text displayed on the arrow
    :return: void
    '''
    xMid = (coordinate_parent[0] - coordinate_child[0]) / 2 + coordinate_child[0]
    yMid = (coordinate_parent[1] - coordinate_child[1]) / 2 + coordinate_child[1]
    visualise_decision_tree.ax1.text(xMid, yMid, text)


# recursive function to generate the plot of the decision tree.
def plot_tree(decision_tree, coordinate_parent, node_text, depth):
    '''
    :param decision_tree: the decision tree
    :param coordinate_parent: coordinate of parent node
    :param node_text: text on the arrow going out from this node
    :param depth: the depth of the current decision tree
    :return: void
    '''
    cntLeafs = get_leaf_number(decision_tree)
    cntrPt = (plot_tree.x_coord + (1.0 + float(cntLeafs)) / 2.0 / plot_tree.width_total, plot_tree.y_coord)
    feature = decision_tree['attribute'] + str(decision_tree['value'])
    left_branch = decision_tree['left']
    right_branch = decision_tree['right']

    # Plot the current node & arrow out:
    plot_text_on_arrow(cntrPt, coordinate_parent, node_text)
    plot_node(feature, cntrPt, coordinate_parent, bbox_node)

    # adjust the y-coordinate for next node
    plot_tree.y_coord = plot_tree.y_coord - 1.0 / plot_tree.depth_total

    if cntLeafs == 1:  # a leaf
        plot_tree.x_coord = plot_tree.x_coord + 1.0 / plot_tree.width_total  # increment the value of x.
    else:
        plot_tree(left_branch, cntrPt, "yes", depth)
        plot_tree(right_branch, cntrPt, "no", depth)

    # reset the position for next node to plot.
    plot_tree.y_coord = plot_tree.y_coord + 1.0 / plot_tree.depth_total


# main function to plot the dcision tree
def visualise_decision_tree(decision_tree, depth):
    fig = plt.figure(figsize=(40, 40), facecolor="white")
    fig.clf()  # clear the canvas
    axprops = dict(xticks=[], yticks=[])
    visualise_decision_tree.ax1 = plt.subplot(111, frameon=False, **axprops)

    plot_tree.width_total = float(
        get_leaf_number(decision_tree))  # global variable: the total number of leafs in the decision tree
    plot_tree.depth_total = float(depth)  # global variable: the total depth of the decision tree.
    plot_tree.x_coord = -0.5 / plot_tree.width_total
    plot_tree.y_coord = 1.0

    plot_tree(decision_tree, (0.5, 1.0), '', 0)
    plt.show()
    fig.savefig('decision_tree.png')


if __name__ == '__main__':
    inputfile = sys.argv[1]
    training_data = np.loadtxt(inputfile)
    dtree, depth = dt.decision_tree_learning(training_data, 0)

    visualise_decision_tree(dtree, depth)
