from itertools import product

import numpy as np

tuned_parameters = [
    {
        'cell_type': ['lstm', 'lstm_peephole', 'gru'],
        'batch_size': 2 ** np.arange(11)[4:],
        'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.1]
    }
]

grid_params = []

for p in tuned_parameters:
    items = sorted(p.items())
    keys, values = zip(*items)
    for v in product(*values):
        params = dict(zip(keys, v))
        grid_params.append(params)

grid_params = np.array(grid_params)
print(len(grid_params))
