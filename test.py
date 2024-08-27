import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建一个表示θ的二值图像（0和1组成）
theta_image = np.array([
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1]
], dtype=np.uint8)

# 放大图像以便更好地可视化
theta_image = cv2.resize(theta_image, (100, 100), interpolation=cv2.INTER_NEAREST)

# 定义结构元素
kernel = np.array([
    [1, 0, 1],
    [0, 0, 0],
    [1, 0, 1]
], dtype=np.uint8)

# 形态学开运算
#opened_image = cv2.morphologyEx(theta_image, cv2.MORPH_OPEN, kernel)

# 闭运算以填补中间的空白
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed_image = cv2.morphologyEx(theta_image, cv2.MORPH_CLOSE, kernel2)

# 显示结果
plt.figure(figsize=(10, 4))

plt.subplot(131)
plt.imshow(theta_image, cmap='gray')
plt.title('Original Theta')
plt.axis('off')

plt.subplot(132)
#plt.imshow(opened_image, cmap='gray')
plt.title('Opened Image')
plt.axis('off')

plt.subplot(133)
plt.imshow(closed_image, cmap='gray')
plt.title('Closed Image')
plt.axis('off')

plt.show()
