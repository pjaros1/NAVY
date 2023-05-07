import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#funkce pro aplikaci transformace na bod
def apply_transformation(point, transformation):
    x, y, z = point
    a, b, c, d, e, f, g, h, i, j, k, l = transformation
    x_new = a * x + b * y + c * z + j
    y_new = d * x + e * y + f * z + k
    z_new = g * x + h * y + i * z + l
    return x_new, y_new, z_new

#funkce pro generovani fraktalniho modelu
def generate_fractal_model(transformations, num_points=10000):
    current_point = (0, 0, 0)
    points = [current_point]

    for _ in range(num_points):
        # vyber nahodne transofrmace s pravdepodobnosti 0.25
        chosen_idx = np.random.choice(len(transformations), p=[0.25, 0.25, 0.25, 0.25])
        chosen_transformation = transformations[chosen_idx]
        # aplikace transformace
        current_point = apply_transformation(current_point, chosen_transformation)
        points.append(current_point)

    return points

def plot_points(points):
    xs, ys, _ = zip(*points)
    plt.scatter(xs, ys, s=1, color="green", marker=".")
    plt.axis("equal")
    plt.show()

transformations1 = [
    (0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00),
    (0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00),
    (-0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00),
    (0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00),
]

transformations2 = [
    (0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00),
    (0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45, 0.00, 1.00, 0.00),
    (-0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22, -0.22, 0.45, 0.00, 1.25, 0.00),
    (0.49, -0.08, 0.08, 0.08, 0.49, 0.08, 0.08, -0.08, 0.49, 0.00, 2.00, 0.00)
]

points1 = generate_fractal_model(transformations1)
points2 = generate_fractal_model(transformations2)

# rozdeleni bodu na souradnice
x1, y1, z1 = zip(*points1) 
x2, y2, z2 = zip(*points2)

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x1, y1, z1, s=1, c='g', marker='o', alpha=0.6)
ax1.set_title("model 1")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x2, y2, z2, s=1, c='b', marker='o', alpha=0.6)
ax2.set_title("model 2")

plt.show()