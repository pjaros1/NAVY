import numpy as np
import matplotlib.pyplot as plt

pattern1 = np.array([[1, -1, 1],
                     [-1, 1, -1],
                     [1, -1, 1]])

pattern2 = np.array([[-1, 1, -1],
                     [1, -1, 1],
                     [-1, 1, -1]])

pattern3 = np.array([[1, 1, 1],
                     [-1, -1, -1],
                     [1, 1, 1]])

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        # inicializace vah 
        self.weights = np.zeros((size, size))  

    # metoda pro trenovani site
    def train(self, patterns):
        n_patterns = patterns.shape[0]  
        for pattern in patterns:
            # aktualizuj vahy  
            self.weights += np.outer(pattern, pattern)
        # normalizace vah  
        self.weights /= n_patterns
        # nastavení diagonaly na nuly  
        np.fill_diagonal(self.weights, 0)  

# vytvoreni a trenovani Hopfieldovy site
patterns = np.array([pattern1.flatten(), pattern2.flatten(), pattern3.flatten()])
hn = HopfieldNetwork(patterns.shape[1])
hn.train(patterns)

# synchronni aktualizace
def synchronous_update(hn, pattern):
    # vypocet aktualizovaneho vzoru
    return np.sign(np.dot(hn.weights, pattern))  

# asynchronni aktualizace
def asynchronous_update(hn, pattern):
    for i in range(pattern.size):  
        # aktualizace prvku
        pattern[i] = np.sign(np.dot(hn.weights[i], pattern))  
    return pattern 

# funkce pro obnoveni vzoru
def renew_pattern(hn, pattern, update_type='synchronous'):
    if update_type == 'synchronous':  
        return synchronous_update(hn, pattern)
    else:
        return asynchronous_update(hn, pattern)

# zmenene vzory
modified_pattern1 = np.array([[1, 1, 1],
                             [-1, 1, -1],
                             [1, -1, 1]])

modified_pattern2 = np.array([[-1, 1, -1],
                             [1, 1, 1],
                             [-1, 1, -1]])

modified_pattern3 = np.array([[1, 1, 1],
                             [-1, -1, 1],
                             [1, 1, 1]])

# prevod zmenenych vzoru na 1D
modified_patterns = np.array([modified_pattern1.flatten(), modified_pattern2.flatten(), modified_pattern3.flatten()])

# vizualizace
def visualize_patterns(title, patterns, shape):
    n_patterns = len(patterns) 
    fig, axes = plt.subplots(1, n_patterns, figsize=(n_patterns * 3, 3))
    for i, ax in enumerate(axes):

        ax.imshow(patterns[i].reshape(shape), cmap='gray_r', vmin=-1, vmax=1)
        ax.set_xticks([])  
        ax.set_yticks([])  
        ax.set_title(f'vzor {i + 1}') 
    
    fig.suptitle(title)  
    plt.show()  


visualize_patterns('puvodní vzory', [pattern1, pattern2, pattern3], (3, 3))
visualize_patterns('zmenene vzory', [modified_pattern1, modified_pattern2, modified_pattern3], (3, 3))


renewed_patterns_sync = []  
renewed_patterns_async = [] 

# obnoveni vzoru sync + async
for modified_pattern in modified_patterns:
    renewed_pattern_sync = renew_pattern(hn, modified_pattern.copy(), update_type='sync')
    renewed_pattern_async = renew_pattern(hn, modified_pattern.copy(), update_type='async')
    
    renewed_patterns_sync.append(renewed_pattern_sync.reshape(3, 3))
    renewed_patterns_async.append(renewed_pattern_async.reshape(3, 3))

visualize_patterns('synchronni obnoveni', renewed_patterns_sync, (3, 3))
visualize_patterns('asynchronni obnoveni', renewed_patterns_async, (3, 3))