import os
from src.preprocessing_data.pipeline import process_image

INPUT_FOLDER = "src\data\raw"
OUTPUT_FOLDER = "src\data\processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def run():
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(INPUT_FOLDER, file)
            output_path = os.path.join(OUTPUT_FOLDER, file)

            process_image(input_path, output_path)

if __name__ == "__main__":
    run()