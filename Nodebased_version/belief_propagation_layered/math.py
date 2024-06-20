import numpy as np

# 示例中的两个消息，可能来自于两个不同的变量节点
previous_messages_1 = np.array([-0.3, -0.5])

# 计算这些消息的双曲正切值的一半
tanh_values = np.tanh(previous_messages_1 / 2)
print("tanh values:", tanh_values)

# 计算这些双曲正切值的乘积
product_tanh = np.prod(tanh_values)
print("product tanh before clip:", product_tanh)

# 为了防止 arctanh 函数处理数值超出其定义域，使用 clip 函数限制乘积的范围
product_tanh = np.clip(product_tanh, -0.999999, 0.999999)
print("product tanh after clip:", product_tanh)

# 计算最终的消息，该消息是两倍的反双曲正切值
message = 2 * np.arctanh(product_tanh)
print("computed message:", message)


print(message)

# previous_messages_2 = np.array([-0.3, -0.5])
# tanh_product_2 = 2*np.arctanh(np.prod(np.tanh(previous_messages_2 / 2)))
# print(tanh_product_2)
