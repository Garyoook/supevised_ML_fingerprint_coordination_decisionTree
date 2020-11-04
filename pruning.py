from collections import deque
from evaluate import evaluate


def prune(test_data, d_tree):
    layers = get_layers(d_tree)
    accuracy = evaluate(test_data, d_tree)
    while layers:
        layer = layers.pop()
        for node in layer:
            if node["left"]["leaf"] and node["right"]["leaf"]:
                prev_node = node
                node = prev_node["left"]
                if evaluate(test_data, d_tree) < accuracy:
                    node = prev_node["right"]
                    if evaluate(test_data, d_tree) < accuracy:
                        node = prev_node
                else:
                    prev_node = node
                    node = prev_node["right"]
                    if evaluate(test_data, d_tree) < accuracy:
                        node = prev_node
    return d_tree


def get_layers(d_tree):
    layers = deque()
    queue = deque()
    queue.append(d_tree)
    while queue:
        row = []
        row_size = len(queue)
        while row_size > 0:
            current_node = queue.popleft()
            if not current_node["left"]["leaf"]:
                queue.append(current_node["left"])
            if not current_node["right"]["leaf"]:
                queue.append(current_node["right"])
            row.append(current_node)
            row_size = row_size - 1
        layers.append(row)
    return layers

