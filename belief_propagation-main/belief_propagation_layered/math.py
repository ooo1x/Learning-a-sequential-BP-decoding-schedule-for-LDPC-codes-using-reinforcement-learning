import numpy as np

previous_messages_1 = np.array([0.7, 0.2])
tanh_product_1 = 2*np.arctanh(np.prod(np.tanh(previous_messages_1 / 2)))

print(tanh_product_1)

# previous_messages_2 = np.array([-0.3, -0.5])
# tanh_product_2 = 2*np.arctanh(np.prod(np.tanh(previous_messages_2 / 2)))
# print(tanh_product_2)

total_messages = -0.4201145069582775+tanh_product_1
print(total_messages)
