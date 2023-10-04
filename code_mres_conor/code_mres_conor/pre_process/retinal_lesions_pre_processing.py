import os
import shutil

source_folder = "C:/Datasets/Raw/Retinal-Lesions/lesion_segs_896x896"
destination_base_folder = "C:/Datasets/retinal_lesions_data"

lesion_types = ["hard_exudate", "cotton_wool_spots", "retinal_hemorrhage", "preretinal_hemorrhage",
                "vitreous_hemorrhage", "fibrous_proliferation", "neovascularization", "microaneurysm"]

destination_folders = {lesion_type: os.path.join(destination_base_folder, lesion_type) for lesion_type in lesion_types}

for folder_name in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder_name)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            for lesion_type in lesion_types:
                if lesion_type in image_name:
                    new_image_name = f"{folder_name}_{image_name}"
                    source_image_path = os.path.join(folder_path, image_name)
                    destination_image_path = os.path.join(destination_folders[lesion_type], new_image_name)
                    shutil.copy(source_image_path, destination_image_path)
