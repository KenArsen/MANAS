import cv2
import numpy as np

image = cv2.imread('BIL-475/image.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sobel():
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # По оси X
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # По оси Y

    sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

    cv2.imwrite('BIL-475/task_1_result_images/sobel_x.jpg', sobel_x)
    cv2.imwrite('BIL-475/task_1_result_images/sobel_y.jpg', sobel_y)
    cv2.imwrite('BIL-475/task_1_result_images/sobel_combined.jpg', sobel_combined)


def laplacian():
    lap = cv2.Laplacian(gray_image, cv2.CV_64F)

    cv2.imwrite('BIL-475/task_1_result_images/laplacian.jpg', lap)


def canny():
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    edges = cv2.Canny(blurred_image, 50, 150)

    cv2.imwrite('BIL-475/task_1_result_images/edges_detected.jpg', edges)


def prewitt():
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    prewitt_x = cv2.filter2D(gray_image, -1, kernel_x)
    prewitt_y = cv2.filter2D(gray_image, -1, kernel_y)

    prewitt_combined = cv2.bitwise_or(prewitt_x, prewitt_y)

    cv2.imwrite('BIL-475/task_1_result_images/prewitt_x.jpg', prewitt_x)
    cv2.imwrite('BIL-475/task_1_result_images/prewitt_y.jpg', prewitt_y)
    cv2.imwrite('BIL-475/task_1_result_images/prewitt_combined.jpg', prewitt_combined)


print("1. Sobel function")
print("2. Laplacian function")
print("3. Canny function")
print("4. Prewitt function")
choice = int(input("Enter your variant: "))
if choice == 1:
    sobel()
elif choice == 2:
    laplacian()
elif choice == 3:
    canny()
elif choice == 4:
    prewitt()
else:
    print("Invalid choice")
