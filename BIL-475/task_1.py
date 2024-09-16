import cv2

image = cv2.imread('BIL-475/image.jpeg')


def save_image(name, img):
    cv2.imwrite(f'BIL-475/task_1_result_images/{name}', img)


def change_pixel_value_image():
    for i in range(10):
        for j in range(10):
            image[i, j] = [255, 255, 255]

    save_image('image_changed_pixel_value.jpg', image)


def rotate_image():
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    angle = 45
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h))

    save_image('image_rotated.jpeg', rotated)


def resize_image():
    resized = cv2.resize(image, (300, 300))

    save_image('image_resized.jpeg', resized)


print("1. Change pixel value image")
print("2. Rotate image")
print("3. Resize image")
choice = int(input("Enter your variant: "))
if choice == 1:
    change_pixel_value_image()
elif choice == 2:
    rotate_image()
elif choice == 3:
    resize_image()
else:
    print("Invalid choice")
