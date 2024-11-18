"""
GrabCut — это интерактивный алгоритм выделения переднего плана на изображении (например, человека или объекта).
Функции:
cv2.grabCut: Выполняет итеративный процесс разделения переднего и заднего плана.
np.where: Создает итоговую маску, чтобы выделить только передний план.

GrabCut требует начального прямоугольника или маски для выделения объекта и работает на основе вероятностной модели.
"""

import cv2
import numpy as np

# Загружаем изображение
image = cv2.imread('BIL-475/image.jpeg', 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Создаем маску
mask = np.zeros(image.shape[:2], np.uint8)

# Определяем модели фона и переднего плана
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Указываем прямоугольник для выделения объекта (координаты x, y, ширина, высота)
rect = (50, 50, 600, 400)

# Применяем алгоритм GrabCut
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Формируем итоговую маску для выделения объекта
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
result = image * mask2[:, :, np.newaxis]

# Сохраняем результат
cv2.imwrite('BIL-475/task_8_grabcut_result.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
