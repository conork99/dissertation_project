from PIL import Image
import os


def convert_images_in_folder(input_folder, output_folder, target_format):
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png', '.tif')):
                image_path = os.path.join(root, filename)
                image = Image.open(image_path)

                output_filename = os.path.splitext(filename)[0] + '.' + target_format
                output_path = os.path.join(output_folder, output_filename)

                image.save(output_path, format=target_format)
                print(f"Converted '{filename}' to '{output_filename}'")


if __name__ == "__main__":
    input_folder = "C:/Datasets/Segmentation"
    output_folder = "C:/Datasets/Segmentation_png"
    target_format = "png"

    convert_images_in_folder(input_folder, output_folder, target_format)
