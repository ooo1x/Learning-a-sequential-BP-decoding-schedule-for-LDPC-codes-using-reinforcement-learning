import numpy as np

previous_messages = np.array([-0.3, -0.5])
tanh_product = 2*np.arctanh(np.prod(np.tanh(previous_messages / 2)))

print(tanh_product)