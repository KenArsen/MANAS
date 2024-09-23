import cv2
import numpy as np

image = cv2.imread('BIL-475/image.jpeg', 0)

_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)

# 1. Эрозия
erosion = cv2.erode(binary_image, kernel, iterations=1)

# 2. Дилатация
dilation = cv2.dilate(binary_image, kernel, iterations=1)

# 3. Открытие (Open)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# 4. Закрытие (Close)
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('BIL-475/task_1_result_images/erosion.jpg', erosion)
cv2.imwrite('BIL-475/task_1_result_images/dilation.jpg', dilation)
cv2.imwrite('BIL-475/task_1_result_images/opening.jpg', opening)
cv2.imwrite('BIL-475/task_1_result_images/closing.jpg', closing)