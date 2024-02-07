import albumentations as A
import cv2 as cv
import os

# transform = A.Compose([
#     A.InvertImg(p=.5),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.Transpose(p=0.5),
#     A.Rotate(limit=70, p=0.5),
#     A.RandomRotate90(p=0.5),
#     A.Sharpen(p=0.2),
#     A.RandomBrightnessContrast(p=0.2),
#     A.GaussNoise(p=0.2),
#     A.PixelDropout(p=0.2),
#     A.GaussianBlur(p=0.2),
# ])

transform = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.InvertImg(p=0.1),
])


def augment_image(image):
    return transform(image=image)['image']


class_name = 'Irregular'

#get all images from class folder
image_files = os.listdir(f'./Oyster Shell/train/{class_name}')
image_files = [file for file in image_files if file.endswith('.tif')]

print(f'There are {len(image_files)} images in class {class_name}')
num_times_thru_data = int(input("How many times trhough the data? "))

for j in range(num_times_thru_data):
    for i, image_file in enumerate(image_files):
        image_path = f'./Oyster Shell/train/{class_name}/{image_file}'
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        augmented_image = augment_image(image)
        cv.imwrite(f'./Oyster Shell/train/{class_name}/{j}_{i}.tif',
                   augmented_image)
        print(
            f"Augmented image {i} of {len(image_files)} for class {class_name} for the {j}th time."
        )
