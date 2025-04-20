import plotly.graph_objects as go
import numpy as np


# Генерація точок
num_points = 10
points_x = np.random.randint(-10, 10, num_points)
points_y = np.zeros(num_points)


# Створення об'єкта "Figure" для графіка
fig = go.Figure()


# Додавання горизонтальної осі X
fig.add_trace(go.Scatter(x=[-10, 10], y=[0, 0], mode='lines', name='X-axis'))


# Додавання точок на координатну пряму
fig.add_trace(go.Scatter(x=points_x, y=points_y, mode='markers', name='Points'))


# Налаштування вигляду графіка
fig.update_layout(
    title='Координатна пряма з точками',
    xaxis=dict(title='Вісь X'),
    yaxis=dict(title='Вісь Y'),
    showlegend=True
)


# Відображення графіка
fig.show()

# Приклад 2

import plotly.graph_objects as go
import numpy as np


# Створення об'єкта "Figure" для графіка
fig = go.Figure()


# Додавання горизонтальної осі X
fig.add_trace(go.Scatter(x=[-10, 10], y=[0, 0], mode='lines', name='X-вісь'))


# Додавання вертикальної осі Y
fig.add_trace(go.Scatter(x=[0, 0], y=[-10, 10], mode='lines', name='Y-вісь'))


# Відмітка для початку координат
fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=8, color='red'), name='Початок координат'))


# Додавання координатної сітки
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')


# Зображення деяких точок
points_x = np.random.randint(-8, 8, 5)
points_y = np.random.randint(-8, 8, 5)


fig.add_trace(go.Scatter(x=points_x, y=points_y, mode='markers', marker=dict(size=10, color='blue'), name='Точка'))


# Налаштування вигляду графіка
fig.update_layout(
    title='Координатна система на площині з точками',
    xaxis=dict(title='Вісь X'),
    yaxis=dict(title='Вісь Y'),
    showlegend=True
)


# Відображення графіка
fig.show()

# Приклад 3

import plotly.graph_objects as go
import numpy as np


# Генерація точок
num_points = 10
points_x = np.random.rand(num_points) * 10
points_y = np.random.rand(num_points) * 10
points_z = np.random.rand(num_points) * 10


# Створення об'єкта "Figure" для 3D графіка
fig = go.Figure()


# Додавання точок у 3D простір
fig.add_trace(go.Scatter3d(
    x=points_x,
    y=points_y,
    z=points_z,
    mode='markers',
    marker=dict(
        size=5,
        color='blue',  # Колір точок
        opacity=0.8
    ),
    name='Random Points'
))


# Налаштування вигляду графіка
fig.update_layout(
    scene=dict(
        xaxis=dict(title='Вісь X'),
        yaxis=dict(title='Вісь Y'),
        zaxis=dict(title='Вісь Z')
    ),
    title='3D простір з точками',
)


# Відображення графіка
fig.show()

# Приклад 4

import matplotlib.pyplot as plt
import numpy as np


# Створення векторів для координат осей
axis_x = np.array([1, 0])
axis_y = np.array([0, 1])


# Побудова системи координат і векторів
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, axis_x[0], axis_x[1], angles='xy', scale_units='xy', scale=1, color='r', label='Вісь X')
plt.quiver(0, 0, axis_y[0], axis_y[1], angles='xy', scale_units='xy', scale=1, color='b', label='Вісь Y')


# Позначення початку координат
plt.scatter(0, 0, color='black', marker='o')


# Додавання підписів
plt.text(axis_x[0], axis_x[1], ' X', fontsize=12, color='r', va='bottom', ha='left')
plt.text(axis_y[0], axis_y[1], ' Y', fontsize=12, color='b', va='bottom', ha='left')


# Налаштування вигляду графіка
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Система координат з одиничними векторами')
plt.xlabel('Вісь X')
plt.ylabel('Вісь Y')
plt.legend()
plt.show()

# Приклад 5

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Створення ортонормованого базису
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])


# Створення 3D-графіка
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


# Додавання векторів до графіка
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='Вектор 1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='g', label='Вектор 2')
ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='b', label='Вектор 3')


# Налаштування вигляду графіка
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel('Вісь X')
ax.set_ylabel('Вісь Y')
ax.set_zlabel('Вісь Z')
ax.set_title('Ортонормований базис у 3D')
ax.legend()


# Відображення графіка
plt.show()

# Приклад 6

import matplotlib.pyplot as plt
import numpy as np


# Створення двох колінеарних векторів
vector_a = np.array([2, 3])
vector_b = 2 * vector_a


# Побудова системи координат і векторів
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='r', label='Вектор A')
plt.quiver(0, 1, vector_b[0], vector_b[1], angles='xy', scale_units='xy', scale=1, color='b', label='Вектор B')


# Позначення початку координат
plt.scatter(0, 0, color='black', marker='o')


# Додавання підписів
plt.text(vector_a[0], vector_a[1], ' A', fontsize=12, color='r', va='bottom', ha='left')
plt.text(vector_b[0], vector_b[1], ' B', fontsize=12, color='b', va='bottom', ha='left')


# Налаштування вигляду графіка
plt.xlim(-1, 7)
plt.ylim(-1, 9)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Колінеарні вектори на площині')
plt.xlabel('Вісь X')
plt.ylabel('Вісь Y')
plt.legend()
plt.show()

# Приклад 7

import plotly.graph_objects as go
import numpy as np


# Створення двох компланарних векторів
vector_a = np.array([2, 3, 1])
vector_b = np.array([1, 4, 2])


# Знаходження нормалі до площини, утвореної векторами
normal_vector = np.cross(vector_a, vector_b)


# Створення масиву точок у площині
point_in_plane = np.array([0, 0, 0])
points_on_plane = np.column_stack((vector_a, vector_b, normal_vector)) + point_in_plane


# Створення об'єкта "Figure" для 3D графіка
fig = go.Figure()


# Додавання векторів до графіка
fig.add_trace(go.Scatter3d(x=[0, vector_a[0]], y=[0, vector_a[1]], z=[0, vector_a[2]],
                           mode='lines+markers', marker=dict(size=5), line=dict(color='red'), name='Вектор A'))
fig.add_trace(go.Scatter3d(x=[0, vector_b[0]], y=[0, vector_b[1]], z=[0, vector_b[2]],
                           mode='lines+markers', marker=dict(size=5), line=dict(color='blue'), name='Вектор B'))
fig.add_trace(go.Scatter3d(x=[0, normal_vector[0]], y=[0, normal_vector[1]], z=[0, normal_vector[2]],
                           mode='lines+markers', marker=dict(size=5), line=dict(color='green'), name='Нормаль'))


# Додавання площини до графіка
xx, yy = np.meshgrid(np.linspace(-2, 4, 5), np.linspace(-2, 4, 5))
zz = (-normal_vector[0] * xx - normal_vector[1] * yy - point_in_plane[2]) / normal_vector[2]
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.3, colorscale='gray', name='Площина'))


# Налаштування вигляду графіка
fig.update_layout(scene=dict(aspectmode='data'))
fig.update_layout(scene=dict(xaxis_title='Вісь X', yaxis_title='Вісь Y', zaxis_title='Вісь Z'))
fig.update_layout(title='Компланарні вектори, що лежать у площині, та нормаль до площини')


# Відображення графіка
fig.show()

# Приклад 8

import matplotlib.pyplot as plt
import numpy as np


# Функція для обчислення модуля вектора
def compute_vector_magnitude(vector):
    return np.sqrt(np.sum(vector ** 2))


# Створення вектора
vector_a = np.array([2, 3])


# Обчислення модуля вектора
magnitude_a = compute_vector_magnitude(vector_a)


# Вивід результату
print(f"Вектор A: {vector_a}")
print(f"Модуль вектора A: {magnitude_a:.2f}")


# Візуалізація вектора на площині
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='b', label='Вектор A')


# Додавання підпису з модулем вектора
plt.text(vector_a[0] / 2, vector_a[1] / 2, f'Modulus: {magnitude_a:.2f}', color='r')


# Налаштування вигляду графіка
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Модуль вектора на площині')
plt.xlabel('Вісь X')
plt.ylabel('Вісь Y')
plt.legend()
plt.show()

# Приклад 9

import plotly.graph_objects as go
import numpy as np


# Функція для обчислення модуля вектора
def compute_vector_magnitude(vector):
    return np.sqrt(np.sum(vector ** 2))


# Створення вектора
vector_a = np.array([2, 5, 4])


# Обчислення модуля вектора
magnitude_a = compute_vector_magnitude(vector_a)


# Вивід результату
print(f"Вектор A: {vector_a}")
print(f"Модуль вектора A: {magnitude_a:.2f}")


# Візуалізація вектора у 3D просторі
fig = go.Figure()


# Додавання вектора до графіка
fig.add_trace(go.Scatter3d(x=[0, vector_a[0]], y=[0, vector_a[1]], z=[0, vector_a[2]],
                           mode='lines+markers', marker=dict(size=5), line=dict(color='blue'), name='Вектор A'))


# Додавання підпису з модулем вектора
fig.add_trace(go.Scatter3d(x=[vector_a[0] / 2], y=[vector_a[1] / 2], z=[vector_a[2] / 2],
                           mode='text', text=[f'Modulus: {magnitude_a:.2f}'], textposition='bottom center', name='Modulus'))


# Налаштування вигляду графіка
fig.update_layout(scene=dict(aspectmode='data'))
fig.update_layout(scene=dict(xaxis_title='Вісь X', yaxis_title='Вісь Y', zaxis_title='Вісь Z'))
fig.update_layout(title='Модуль вектора та візуалізація у 3D просторі')


# Відображення графіка
fig.show()

# Приклад 10
import matplotlib.pyplot as plt
import numpy as np


# Функція для обчислення модуля вектора
def compute_vector_magnitude(vector):
    return np.sqrt(np.sum(vector ** 2))


# Функція для обчислення напрямних косинусів
def compute_direction_cosines(vector):
    magnitude = compute_vector_magnitude(vector)
    cosine_alpha = vector[0] / magnitude
    cosine_beta = vector[1] / magnitude
    cosine_gamma = vector[2] / magnitude
    return cosine_alpha, cosine_beta, cosine_gamma


# Створення вектора
vector_a = np.array([2, 3, 4])


# Обчислення напрямних косинусів
cosine_alpha, cosine_beta, cosine_gamma = compute_direction_cosines(vector_a)


# Вивід результату
print(f"Вектор A: {vector_a}")
print(f"Напрямні косинуси: α={cosine_alpha:.2f}, β={cosine_beta:.2f}, γ={cosine_gamma:.2f}")


# Візуалізація вектора на площині
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, vector_a[0], vector_a[1], vector_a[2], color='b', label='Вектор A')


# Додавання підписів із напрямними косинусами
ax.text(vector_a[0] / 2, vector_a[1] / 2, vector_a[2] / 2, f'α={cosine_alpha:.2f}\nβ={cosine_beta:.2f}\nγ={cosine_gamma:.2f}',
        color='r')


# Налаштування вигляду графіка
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 5])
ax.set_xlabel('Вісь X')
ax.set_ylabel('Вісь Y')
ax.set_zlabel('Вісь Z')
ax.set_title('Напрямні косинуси вектора на площині')


plt.legend()
plt.show()

# Приклад 11

import numpy as np


def calculate_vectors_components(vectors):
    """
    Функція для визначення компонент декількох векторів.


    Параметри:
    - vectors: Список кортежів (start_point, end_point), де start_point та end_point - це координати початкової та кінцевої точок вектора.


    Повертає:
    - components: Список кортежів (x_components, y_components), де x_components та y_components - це компоненти по X та Y для кожного вектора.
    """
    components = []


    for start_point, end_point in vectors:
        x_component = end_point[0] - start_point[0]
        y_component = end_point[1] - start_point[1]
        components.append((x_component, y_component))


    return components


# Приклад використання функції з декількома векторами
vectors = [((1, 2), (4, 6)), ((2, 3), (5, 2)), ((0, 0), (-2, 4))]


vector_components = calculate_vectors_components(vectors)
print(vector_components)

# Приклад 12

import matplotlib.pyplot as plt
import numpy as np


def draw_vectors(vectors, colors=None, title=None):
    """
    Функція для малювання декількох векторів на площині засобами бібліотеки Matplotlib.


    Параметри:
    - vectors: Список кортежів (start_point, end_point), де start_point та end_point - це координати початкової та кінцевої точок вектора.
    - colors: Список кольорів для векторів (за замовчуванням використовується 'blue' для всіх векторів).
    """
    num_vectors = len(vectors)


    if colors is None:
        colors = ['blue'] * num_vectors


    if title is None:
        title = 'Декілька векторів на площині'


    # Створення нового рисунка та осей
    fig, ax = plt.subplots()


    for i in range(num_vectors):
        start_point, end_point = vectors[i]


        # Визначення вектора та його довжини
        vector = np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])
        length = np.linalg.norm(vector)


        # Нормалізація вектора
        # normalized_vector = vector / length
        normalized_vector = vector


        # Додавання стрілки до осей
        ax.arrow(start_point[0], start_point[1], normalized_vector[0], normalized_vector[1],
                 head_width=0.1, head_length=0.1, fc=colors[i], ec=colors[i])


    # Налаштування візуалізації
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)


    # Відображення графіка
    plt.show()


# Приклад використання функції з декількома векторами та кольорами
vectors = [((1, 2), (4, 6)), ((2, 3), (5, 2)), ((0, 0), (-2, 4))]
colors = ['green', 'red', 'purple']


draw_vectors(vectors, colors)


import matplotlib.pyplot as plt
import numpy as np

# Вектори
vectors = [
    (np.array([1, 3]), np.array([3, 1])),
    (np.array([1, -1]), np.array([3, 1])),
    (np.array([1, 1]), np.array([2, -1])),
    (np.array([3, 3]), np.array([-3, 1])),
    (np.array([2, 4]), np.array([-1, 3]))
]

# Побудова графіку
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, (a, b) in enumerate(vectors):
    ax = axes[i]
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color='blue', label='a')
    ax.quiver(a[0], a[1], b[0], b[1], angles='xy', scale_units='xy', scale=1, color='green', label='b (from a)')
    ax.quiver(0, 0, a[0]+b[0], a[1]+b[1], angles='xy', scale_units='xy', scale=1, color='red', label='a + b')
    ax.set_xlim(-5, 7)
    ax.set_ylim(-5, 7)
    ax.grid()
    ax.set_aspect('equal')
    ax.set_title(f'Випадок {i+1}')
    ax.legend()

plt.tight_layout()
plt.show()


# Приклад 16

import matplotlib.pyplot as plt
import numpy as np


# Створення ортогональних векторів
v1 = np.array([2, 1])
v2 = np.array([-1, 2])


# Обчислення скалярного добутку
dot_product = np.dot(v1, v2)


# Створення 2D-графіка
fig, ax = plt.subplots()
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Вектор v1')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='g', label='Вектор v2')


# Додавання підписів та гриду
ax.text(v1[0], v1[1], f'v1 ({v1[0]}, {v1[1]})', fontsize=12, ha='right', va='bottom', color='r')
ax.text(v2[0], v2[1], f'v2 ({v2[0]}, {v2[1]})', fontsize=12, ha='left', va='bottom', color='g')
ax.text(0.5, -0.5, f'Dot Product: {dot_product}', fontsize=12, ha='center', va='top')


ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(color='gray', linestyle='--', linewidth=0.5)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-2, 3])
ax.set_ylim([-1, 3])
ax.set_xlabel('Вісь X')
ax.set_ylabel('Вісь Y')
ax.set_title('Ортогональні вектори та їх скалярний добуток')


# Відображення графіка
plt.legend()
plt.show()

# Приклад 17

import plotly.graph_objects as go
import numpy as np


def dot_product(vector1_start, vector1_end, vector2_start, vector2_end):
    # Обчислюємо вектори
    vector1 = np.array(vector1_end) - np.array(vector1_start)
    vector2 = np.array(vector2_end) - np.array(vector2_start)


    # Обчислюємо скалярний добуток
    scalar_product = np.dot(vector1, vector2)


    return scalar_product



# Задаємо координати векторів
vector1_start = [1, 2]
vector1_end = [4, 6]


vector2_start = [3, 1]
vector2_end = [6, 3]


dot_product(vector1_start, vector1_end, vector2_start, vector2_end)

# Приклад 18

import numpy as np


def calculate_cosine_angle(vector1, vector2):
    """
    Функція для визначення косинусного кута між двома векторами.


    Параметри:
    - vector1: Кортеж (x1, y1) з координатами кінця першого вектора.
    - vector2: Кортеж (x2, y2) з координатами кінця другого вектора.


    Повертає:
    - cosine_angle: Косинус кута між векторами.
    """
    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)


    cosine_angle = dot_product / (magnitude_vector1 * magnitude_vector2)
    return cosine_angle


# Приклад використання функції з двома векторами
vector1 = (3, 4)
vector2 = (1, 2)


cosine_angle = calculate_cosine_angle(vector1, vector2)
angle_degrees = np.degrees(np.arccos(cosine_angle))


print(f"Косинусний кут між векторами: {cosine_angle:.2f}")
print(f"Кут між векторами у градусах: {angle_degrees:.2f}")


# Приклад 19

import numpy as np


def normal_vector(u, v):
    # Обчислення векторного добутку (cross product)
    n = np.cross(u, v)


    return n


# Приклад використання
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])


normal = normal_vector(u, v)
print("Вектор нормалі до площини:", normal)