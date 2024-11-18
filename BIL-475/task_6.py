"""
Вычисляет свойства объектов, такие как площадь и периметр, на основе их контуров.

Функции:
cv2.findContours: Находит контуры объектов на бинарном изображении.
cv2.contourArea: Вычисляет площадь объекта.
cv2.arcLength: Вычисляет периметр объекта.
cv2.drawContours: Рисует контуры объектов.

Здесь акцент на измерении объектов, а не на разделении изображения.
"""
import cv2

# Загружаем изображение
image = cv2.imread('BIL-475/image.jpeg', 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Преобразуем изображение в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Применяем бинарный порог
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Ищем контуры
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Создаем изображение для отображения контуров
output_image = image.copy()

# Измеряем свойства каждого контура
for contour in contours:
    area = cv2.contourArea(contour)  # Площадь
    perimeter = cv2.arcLength(contour, True)  # Периметр
    print(f"Площадь: {area}, Периметр: {perimeter}")

    # Рисуем контуры на изображении
    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

# Сохраняем результат
cv2.imwrite('BIL-475/task_6_contours_result.jpg', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
