import numpy as np

# definice aktivacni funkce (sigmoid) - pouziva se pro prevod linearniho vystupu neuronu na nelinearni
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# definice derivace aktivacni funkce (sigmoid) - pouziva se pri zpetne propagaci pro aktualizaci vah a biasu
def sigmoid_derivative(x):
    return x * (1 - x)

# vstupy pro XOR problem
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

# ocekavane vystupy pro XOR problem
expected_output = np.array([[0], [1], [1], [0]])

# pocet neuronu ve vstupni, skryte a vystupni vrstve
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# inicializace nahodnych vah a biasu pro skrytou a vystupni vrstvu
np.random.seed(0)
weights_hidden = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
weights_output = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
bias_hidden = np.random.uniform(size=(1, hiddenLayerNeurons))
bias_output = np.random.uniform(size=(1, outputLayerNeurons))

# vypis pocatecnich vah a biasu
print("pocatecni vahy a biasy:")
print(weights_hidden)
print(weights_output)
print(bias_hidden)
print(bias_output)

# bastaveni rychlosti uceni a poctu epoch - ridi proces trenovani
learning_rate = 0.5
epochs = 10000

# trenovani neuronove site
for epoch in range(epochs):

    # vypocet (vstup x vahy + bias) skryte vrstvy
    hidden_layer_calculation = np.dot(inputs, weights_hidden) + bias_hidden
    # vypocet vystupu skryte vrstvy (aktivace prevedena pomoci aktivacni funkce)
    hidden_layer_output = sigmoid(hidden_layer_calculation)
    # vypocet (vystup skryte vrstvy x vahy + bias) vystupni vrstvy
    output_layer_calculation = np.dot(hidden_layer_output, weights_output) + bias_output
    # vypocet testovaneho vystupu (aktivace prevedena pomoci aktivacni funkce)
    tested_output = sigmoid(output_layer_calculation)

    # vypocet chyby mezi ocekavanym a testovanym vystupem
    error = expected_output - tested_output

    # zpetna propagace
    # vypocet delta_output (zmeny pro vahy a biasy mezi skrytou a vystupni vrstvou)
    delta_output = error * sigmoid_derivative(tested_output)
    # vypocet delta_hidden (zmeny pro vahy a biasy mezi vstupni a skrytou vrstvou)
    delta_hidden = np.dot(delta_output, weights_output.T) * sigmoid_derivative(hidden_layer_output)

    # aktualizace vah a biasu mezi skrytou a vystupni vrstvou
    weights_output += np.dot(hidden_layer_output.T, delta_output) * learning_rate
    # aktualizace vah a biasu mezi vstupni a skrytou vrstvou
    weights_hidden += np.dot(inputs.T, delta_hidden) * learning_rate
    # aktualizace biasu ve vystupni vrstve
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    # aktualizace biasu ve skryte vrstve
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

# vypis vah a biasu po uceni
print("\nvahy a biasy po uceni:")
print(weights_hidden)
print(weights_output)
print(bias_hidden)
print(bias_output)

# testovani natrenovane neuronove site
print("\ntestovani funkce XOR:")
for i in range(len(inputs)):
    # vypocet aktivity skryte vrstvy pro testovaci vstup
    hidden_layer_calculation = np.dot(inputs[i], weights_hidden) + bias_hidden
    # vypocet vystupu skryte vrstvy pro testovaci vstup
    hidden_layer_output = sigmoid(hidden_layer_calculation)
    # vypocet aktivity vystupni vrstvy pro testovaci vstup
    output_layer_calculation = np.dot(hidden_layer_output, weights_output) + bias_output
    # vypocet testovaneho vystupu pro testovaci vstup
    tested_output = sigmoid(output_layer_calculation)
    # vypis vstupu, ocekavaneho vystupu a testovaneho vystupu
    print(f"vstup: {inputs[i]} ocekavany vystup: {expected_output[i]} predpovedzeny vystup: {tested_output}")