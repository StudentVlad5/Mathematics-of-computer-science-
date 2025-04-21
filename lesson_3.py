import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import math


set1 = set([i for i in range(0,31,2)])
set2 = set([i for i in range(0,31,5)])
set3 = set([i for i in range(0,31) if math.sqrt(i) - int(math.sqrt(i)) ==0])
print(set1, set2, set3)

venn3([set1, set2, set3], ('Парні', 'кратні 5', 'Корені чисел'), set_colors=('red', 'green', 'blue'), alpha=0.5)
plt.title('Венн діаграма для множин парних чисел, кратних 5 та квадратів чисел')
plt.show()


import networkx as nx
import matplotlib.pyplot as plt


# Створення пустого графа
G = nx.Graph()


# Додавання вершин
G.add_nodes_from([1, 2, 3, 4, 5])


# Додавання ребер (з'єднань між вершинами)
G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5),(3,4)])


# Візуалізація графа
pos = nx.spring_layout(G)  # Визначення позицій вершин
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=10, edge_color='gray')
plt.show()

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def build_and_visualize_graph(incidence_matrix):
    """
    Функція, яка будує граф із заданої матриці інцидентності та візуалізує його.
    Параметри:
    - incidence_matrix: numpy.ndarray, матриця інцидентності графа.
    Повертає:
    - None
    """
    # Створення графа
    graph = nx.Graph()
    num_nodes, num_edges = incidence_matrix.shape


    # Додавання вершин до графа
    graph.add_nodes_from(range(num_nodes))


    # Додавання ребер та їх ваг
    for edge_index in range(num_edges):
        edge_nodes = np.where(incidence_matrix[:, edge_index] != 0)[0]
        if len(edge_nodes) == 2:
            graph.add_edge(edge_nodes[0], edge_nodes[1], weight=edge_index + 1)


    # Визначення позицій вершин для візуалізації
    pos = nx.spring_layout(graph)


    # Визуалізація графа
    nx.draw(graph, pos, with_labels=True, font_size=10, node_size=700, font_color='black', edge_color='gray')


    # Додавання ваг ребер
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='red')


    plt.title("Graph from Incidence Matrix")
    plt.show()


# Приклад використання функції з попередньо заданою матрицею інцидентності
# example_incidence_matrix = np.array([
#     [1, 0, 0, 0, 1, 0, 0],
#     [1, 1, 0, 0, 0, 1, 0],
#     [0, 1, 1, 0, 0, 0, 0],
#     [0, 0, 1, 1, 0, 0, 1],
#     [0, 0, 0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 1]
# ])

example_incidence_matrix = np.array([
    [-1, 0, 0, 0, 1, 0, 0],
    [1, -1, 0, 0, 0, 1, 0],
    [0, 1, -1, 0, 0, 0, 0],
    [0, 0, 1,- 1, 0, 0, -1],
    [0, 0, 0, 1, -1, -1, 0],
    [0, 0, 0, 0, 0, 0, 1]
])
build_and_visualize_graph(example_incidence_matrix)

# Приклад 15

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def build_and_visualize_graph(adjacency_matrix):
    """
    Функція, яка будує граф із заданої матриці суміжності та візуалізує його.


    Параметри:
    - adjacency_matrix: numpy.ndarray, матриця суміжності графа.


    Повертає:
    - None
    """
    # Створення графа
    graph = nx.Graph()
    num_nodes = adjacency_matrix.shape[0]


    # Додавання вершин до графа
    graph.add_nodes_from(range(num_nodes))


    # Додавання ребер та їх ваг
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] != 0:
                graph.add_edge(i, j, weight=adjacency_matrix[i, j])


    # Визначення позицій вершин для візуалізації
    pos = nx.spring_layout(graph)


    # Визуалізація графа
    nx.draw(graph, pos, with_labels=True, font_size=10, node_size=700, font_color='black', edge_color='gray')


    # Додавання ваг ребер
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='red')


    plt.title("Graph from Adjacency Matrix")
    plt.show()


# Приклад використання функції з попередньо заданою матрицею суміжності
example_adjacency_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
])


build_and_visualize_graph(example_adjacency_matrix)