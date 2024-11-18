"""
Сегментация — это процесс разделения изображения на несколько областей или объектов.
"""

import cv2
import numpy as np

# Загружаем изображение
image = cv2.imread('BIL-475/image.jpeg', 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def k_means_clustering():
    """
    K-means кластеризация группирует пиксели изображения в
    k кластеров на основе их цветового сходства.
    Функции:
    cv2.kmeans: Реализует кластеризацию K-means.
    np.reshape: Преобразует изображение в плоский массив для обработки.
    np.float32: Переводит данные в формат с плавающей точкой, необходимый для K-means.
    np.uint8: Конвертирует центры кластеров обратно в формат изображения.

    K-means не учитывает пространственные отношения между пикселями. Это просто группировка по цвету.
    """
    # Преобразуем изображение в массив пикселей
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Настраиваем параметры K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3  # Количество кластеров
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Преобразуем центры кластеров в uint8
    centers = np.uint8(centers)

    # Восстанавливаем изображение с сегментированными кластерами
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Сохраняем результат
    cv2.imwrite('BIL-475/task_5_segmented_image.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))


def watershed_algorithm():
    """
    Watershed используется для разделения объектов, которые находятся рядом друг с другом, и выделения областей на основе их формы.
    Функции:
    cv2.distanceTransform: Вычисляет расстояние от каждого пикселя до ближайшей границы.
    cv2.threshold: Используется для выделения объектов и фона.
    cv2.watershed: Реализует алгоритм, который обрабатывает изображение как топографическую карту, "заливая" области с разных сторон до их пересечения.

    Watershed учитывает пространственные отношения, поэтому он лучше подходит для задач, где важны границы между объектами.
    """
    # Преобразуем изображение в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Применяем бинарный порог
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Определяем фон и передний план
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Выделяем неизвестные области
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Маркируем области
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Применяем алгоритм Watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Обводим границы синим цветом

    # Сохраняем результат
    cv2.imwrite('BIL-475/task_5_watershed_result.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


k_means_clustering()
watershed_algorithm()
