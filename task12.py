import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

EMPTY = 0
TREE = 1
FIRE = 2

# funkce pro aktualizaci stavu bunky
def update_cell(grid, i, j, f, p):
    if grid[i, j] == FIRE:
        return EMPTY
    elif grid[i, j] == TREE:
        # pokud hori alespon jeden strom
        if any(grid[x, y] == FIRE for x, y in get_neighbors(grid, i, j)):
            return FIRE
        elif random.random() < f:
            return FIRE
        else:
            return TREE
    else:
        if random.random() < p:
            return TREE
        else:
            return EMPTY

# vrati sousedy okolni bunky
def get_neighbors(grid, i, j):
    rows, cols = grid.shape
    # vrati seznam souradnic vsech sousednich bunek
    return [(x, y) for x in range(i-1, i+2) for y in range(j-1, j+2)
            if 0 <= x < rows and 0 <= y < cols and (x, y) != (i, j)]

# aktualizuje mrizku
def update_grid(grid, f, p):
    size = grid.shape[0]
    new_grid = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            # aktualizuje stav bunky a ulozi do nove mrizky
            new_grid[i, j] = update_cell(grid, i, j, f, p)
    return new_grid

# funkce pro vykresleni animace
def animate_forest_fire(grid, f, p, steps):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('RdYlGn')
    # nastaveni barev seda - prazdna bunka, zelena - strom, oranzova - horici strom
    colors = [(0.8, 0.8, 0.8, 1), cmap(1.0), cmap(0.3)]
    cmap = ListedColormap(colors)
    
    # funkce pro aktualizaci animace
    def update(i):
        nonlocal grid
        grid = update_grid(grid, f, p)
        mat.set_data(grid)
        return [mat]

    mat = ax.matshow(grid, cmap=cmap)
    ani = animation.FuncAnimation(fig, update, frames=steps, blit=True)
    plt.show()

size = 50
steps = 100
f = 0.001 # pravdepodobnost vzniku pozaru
p = 0.01 # pravdepodobnost vzniku noveho stromu

# inicializace lesa
grid = np.random.choice([EMPTY, TREE], (size, size), p=[1-p, p])

# pridame par horicich stromu do zacatku, aby slo rychleji videt prubeh pozaru
num_burning_trees = 10
for _ in range(num_burning_trees):
    i, j = np.random.randint(0, size, 2)
    grid[i, j] = FIRE

animate_forest_fire(grid, f, p, steps)