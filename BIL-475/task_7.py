"""
Преобразует изображение из пространственной области в частотную область. Это полезно для анализа текстур, шумов или фильтрации.
Функции:
np.fft.fft2: Преобразует данные в частотное представление.
np.fft.fftshift: Сдвигает низкие частоты в центр изображения.
np.log: Преобразует амплитудный спектр для лучшей визуализации.

Эта техника больше о преобразовании данных для анализа, а не о выделении объектов.
"""

import cv2
import numpy as np

# Загружаем изображение
image = cv2.imread('BIL-475/image.jpeg', 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Преобразуем изображение в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Применяем преобразование Фурье
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)  # Сдвигаем нулевую частоту в центр
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Сохраняем спектр как изображение
cv2.imwrite('BIL-475/task_7_fourier_transform.jpg', magnitude_spectrum.astype(np.uint8))
