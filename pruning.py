from collections import deque


def prune(test_data, d_tree):
    layers = get_layers(d_tree)
    accuracy = get_accuracy(test_data, d_tree)
    while layers:
        layer = layers.pop()
        for node in layer:
            if node["left"]["leaf"] and node["right"]["leaf"]:
                prev_node = node
                node = prev_node["left"]
                if get_accuracy(test_data, d_tree) < accuracy:
                    node = prev_node["right"]
                    if get_accuracy(test_data, d_tree) < accuracy:
                        node = prev_node
                else:
                    prev_node = node
                    node = prev_node["right"]
                    if get_accuracy(test_data, d_tree) < accuracy:
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


def get_accuracy(test_data, d_tree):
    return 0
