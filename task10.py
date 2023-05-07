import numpy as np
import matplotlib.pyplot as plt

# 1. definice logistikce mapy
def logistic_map(x_n, a):
    return a * x_n * (1 - x_n)

#2. vizualizace bifurkacniho diagramu
def bifurcation_diagram(a_min, a_max, num_a, num_iterations, num_last_points):
    # rozdeleni intervalu na rovnomerne casti
    a_values = np.linspace(a_min, a_max, num_a) 
    x = np.zeros(num_last_points)
    
    for a in a_values:
        x_n = 0.5
        for i in range(num_iterations):
            # vypocet dalsi hodnoty pomoci logisticke mapy
            x_n = logistic_map(x_n, a)
            if i >= (num_iterations - num_last_points):
                x[i - (num_iterations - num_last_points)] = x_n
                
        plt.plot([a] * num_last_points, x, 'k.', markersize=1)
    
    plt.xlabel("a")
    plt.ylabel("x")
    plt.title("bifurkacni diagram")
    plt.show()

# vykresleni, v beznem rozsahu
bifurcation_diagram(a_min=2.5, a_max=4, num_a=1000, num_iterations=1000, num_last_points=100)