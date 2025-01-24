import os
import shutil
from ultralytics import YOLO
import random

"""
This file starts the training by sorting the data into train and validation groups.
It trains at four resolutions, 640x640, 960x960, 1280x1280 and 1600x1600. It does this so
the model can learn finer details in gradual steps.
"""

def prepare_datasets_with_static_groups(base_dir, train_groups, val_groups, group_files):
    """
    Organizes files into training and validation groups.
    """
    all_files = os.listdir(base_dir)
    image_files = [f for f in all_files if f.endswith(".png")]
    label_files = [f for f in all_files if f.endswith(".txt")]

    #Group files by subject identifier
    for file in image_files + label_files:
        group_id = ''.join(filter(str.isdigit, os.path.splitext(file)[0]))
        if group_id in group_files:
            group_files[group_id].append(file)

def write_data_yaml(train_dir, val_dir, yaml_path, class_names):
    """
    Writes the data.yaml file for YOLO training with correct paths.
    """
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.abspath(train_dir)}\n")
        f.write(f"val: {os.path.abspath(val_dir)}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    print(f"data.yaml written to {yaml_path}")

def train_with_static_groups(output_dir, train_groups, val_groups, model_weights, epochs, configs, patience):
    """
    Trains the model with static training and validation groups across progressive optimization steps.
    """
    for step_idx, (imgsz, batch_size) in enumerate(configs):
        print(f"Starting progression step {step_idx + 1} with imgsz={imgsz}, batch_size={batch_size}...")

        #Create dataset directories for this step
        dataset_dir = os.path.join(output_dir, f"step_{step_idx + 1}")
        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        #Helper to copy files
        def copy_group_files(group_ids, dest_dir):
            for group_id in group_ids:
                for file in group_files[group_id]:
                    shutil.copy(os.path.join(BASE_DIR, file), dest_dir)

        #Copy train and validation data
        copy_group_files(train_groups, train_dir)
        copy_group_files(val_groups, val_dir)

        #Create `data.yaml` for this step
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        write_data_yaml(train_dir, val_dir, yaml_path, CLASS_NAMES)

        #Train the model for this step
        print(f"Training on step {step_idx + 1}...")

        model = YOLO(model_weights, task="obb")
        model.train(
            data=yaml_path,
            task="obb",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            workers=12,
            device=0,
            project=dataset_dir,
            name=f"training_step_{step_idx + 1}",
            save=True,
            patience=patience,
            augment=True,
            mosaic=True,
            mixup=True,
        )

        #Update model weights for the next step
        model_weights = os.path.join(dataset_dir, f"training_step_{step_idx + 1}", "weights", "last.pt")
        print(f"Step {step_idx + 1} completed. Updated weights: {model_weights}")

if __name__ == "__main__":
    #Paths
    BASE_DIR = r"Training Images"
    OUTPUT_DIR = r"ProgressiveValidationResults"
    MODEL_WEIGHTS = "yolo11x-obb.pt"

    #Training parameters
    EPOCHS = 1000 
    PATIENCE = 100

    #Class names for the data.yaml (names of fireworks)
    CLASS_NAMES = [
        "AllBlue", "Apocalypse", "Envy", "GlitteringBrocades", "HardCore",
        "LoudAndClear", "MasterBlaster", "MysticalSky", "Phantom",
        "RainbowCoconut", "ShortCircuit", "Snap", "StrobingCoconut", "ThunderingRainbow"
    ]

    #Static validation groups
    VAL_GROUPS = ["1", "7", "14", "29"]  #Validation groups are fixed, chosen fairly arbitrarily, for diversity
    GROUPS = [str(i) for i in range(1, 35)]
    TRAIN_GROUPS = [g for g in GROUPS if g not in VAL_GROUPS]  #All other groups are used for training

    #Progressive optimization steps
    PROGRESSIVE_CONFIGS = [
        (640, 10),   #imgsz=640, batch_size=10
        (960, 4),    #imgsz=960, batch_size=4
        (1280, 2),   #imgsz=1280, batch_size=2
        (1600, 1)   #imgsz=1600, batch_size=1
    ]

    #Prepare group files
    group_files = {group: [] for group in GROUPS}
    prepare_datasets_with_static_groups(BASE_DIR, TRAIN_GROUPS, VAL_GROUPS, group_files)

    #Train with static groups and progressive optimization
    train_with_static_groups(
        output_dir=OUTPUT_DIR,
        train_groups=TRAIN_GROUPS,
        val_groups=VAL_GROUPS,
        model_weights=MODEL_WEIGHTS,
        epochs=EPOCHS,
        configs=PROGRESSIVE_CONFIGS,
        patience=PATIENCE,
    )