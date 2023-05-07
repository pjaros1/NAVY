import numpy as np
import matplotlib.pyplot as plt

# vygenerovani 100 nahodnych bodu
points = np.random.randint(-10, 11, size=(100, 2))

# definujeme funkci pro vypocet hodnoty y na primce y = 3x + 2
def line(x):
    return 3 * x + 2

# pro kazdy bod urcime jeho klasifikaci (1 = nad primkou, -1 = pod primkou, 0 = na primce)
labels = np.array([1 if point[1] > line(point[0]) else 
                  -1 if point[1] < line(point[0]) else 
                  0 for point in points])

# inicializujeme vahy nahodnymi hodnotami a nastavime bias
weights = np.random.rand(2)
bias = 0.5
# nastavime rychlost uceni (learning rate) a pocet epoch (opakovani treninku)
learning_rate = 0.1
epochs = 100

# trenink perceptronu
for _ in range(epochs):
    # projdeme vsechny body a jejich klasifikace
    for point, label in zip(points, labels):
        # spocitame predikci pomoci soucasnych vah a biasu
        prediction = np.dot(weights, point) + bias
        # urcime vystup perceptronu pomoci funkce signum
        output = np.sign(prediction)
        
        # pokud vystup neodpovida skutecne klasifikaci, upravime vahy a bias
        if output != label:
            weights += learning_rate * label * point
            bias += learning_rate * label

# definujeme funkci perceptronu pro klasifikaci bodu
def perceptron(point):
    # spocitame predikci pomoci natrenovanych vah a biasu
    prediction = np.dot(weights, point) + bias
    # urcime vystup perceptronu pomoci funkce signum
    return np.sign(prediction)

# klasifikujeme body pomoci natrenovaneho perceptronu
point_classifications = [perceptron(point) for point in points]

# pripravime si vizualizaci pomoci matplotlib
fig, ax = plt.subplots()
# definujeme x-ove hodnoty pro kresleni primky
x = np.linspace(-10, 10, 100)

# nakreslime primku y = 3x + 2 pomoci definovane funkce line
ax.plot(x, line(x), label="y = 3x + 2", color='black', linestyle="-")

# rozdelime body do skupin podle klasifikace (1 = nad primkou, -1 = pod primkou, 0 = na primce)
above = np.array([point for point, classification in zip(points, point_classifications) if classification == 1])
below = np.array([point for point, classification in zip(points, point_classifications) if classification == -1])
on_line = np.array([point for point, classification in zip(points, point_classifications) if classification == 0])

# nakreslime body nad primkou zelenou barvou
ax.scatter(above[:, 0], above[:, 1], c='green', label='nad primkou')

#nakreslime body pod primkou cervenou barvou
ax.scatter(below[:, 0], below[:, 1], c='red', label='pod primkou')

#pokud existuji body na primce, nakreslime je modrou barvou
if len(on_line) > 0:
    ax.scatter(on_line[:, 0], on_line[:, 1], c='blue', label='na primce')

ax.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('klasifikace pomoci perceptronu')
plt.show()