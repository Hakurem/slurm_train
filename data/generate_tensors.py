
import os
from tqdm import tqdm as tqdm
from torchvision import transforms
import PIL
from PIL import Image
import torch

PIL.Image.MAX_IMAGE_PIXELS = 1809600000
include = ["validate"]

transformer = transforms.Compose([
    transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

control = False

current_path = os.path.dirname(os.path.realpath(__file__))
tensor_path = os.path.join(current_path, "tensors")

if not os.path.exists(tensor_path):
    os.makedirs(tensor_path)


if "train" in include:
    train_image_path = os.path.join(current_path, "train_images")
    train_tensor_path = os.path.join(tensor_path, "train_tensors")

    if not os.path.exists(train_image_path):
        raise ValueError("Train images not found")

    if not os.path.exists(train_tensor_path):
        os.makedirs(train_tensor_path)

    count = 0
    for image in os.listdir(train_image_path):
        if count % 500 == 0:
            print(
                f"Train images: Processed {count} of {len(os.listdir(train_image_path))} images. {round((count / len(os.listdir(train_image_path)))*100)} %"
            )
        count += 1
        try:
            x = Image.open(os.path.join(train_image_path, image)).convert("RGB")
            x = transformer(x)
        except:
            print("Error in image", image)
            continue

        torch.save(x, os.path.join(train_tensor_path, image[:-4] + ".pt"))

    print("Train images: Done")

if "test" in include:
    test_image_path = os.path.join(current_path, "test_images")
    test_tensor_path = os.path.join(tensor_path, "test_tensors")

    if not os.path.exists(test_image_path):
        raise ValueError("Test images not found")

    if not os.path.exists(test_tensor_path):
        os.makedirs(test_tensor_path)

    count = 0
    for image in os.listdir(test_image_path):
        if count % 500 == 0:
            print(
                f"Test images: Processed {count} of {len(os.listdir(test_image_path))} images. {round((count / len(os.listdir(test_image_path)))*100)} %"
            )
        count += 1

        try:
            x = Image.open(os.path.join(test_image_path, image)).convert("RGB")
            x = transformer(x)
        except:
            print("Error in image", image)
            continue

        torch.save(x, os.path.join(test_tensor_path, image + ".pt"))

    print("Test images: Done")

if "validate" in include:
    validate_image_path = os.path.join(current_path, "validate_images")
    validate_tensor_path = os.path.join(tensor_path, "validate_tensors")

    if not os.path.exists(validate_image_path):
        raise ValueError("Validate images not found")

    if not os.path.exists(validate_tensor_path):
        os.makedirs(validate_tensor_path)

    count = 0
    for image in os.listdir(validate_image_path):
        if count % 500 == 0:
            print(
                f"Validate images: Processed {count} of {len(os.listdir(validate_image_path))} images. {round((count / len(os.listdir(validate_image_path)))*100)} %"
            )
        count += 1

        try:
            x = Image.open(os.path.join(validate_image_path, image)).convert("RGB")
            x = transformer(x)
        except:
            print("Error in image", image)
            continue
        
        try:
            torch.save(x, os.path.join(validate_tensor_path, image + ".pt"))
        except RuntimeError:
            print(os.path.join(validate_tensor_path, image + ".pt"))
            print(x.size())
            print(x)
            raise Exception("STOP") 
    print("Validate images: Done")


print("done")
