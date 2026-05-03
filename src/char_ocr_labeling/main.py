import os
import cv2

from labeling import Labeler
from cli_selection import interactive_best_selection_cli
from saver import save_final_outputs

ROOT = "char_crops"


def load_grouped_segments(root):

    data = {}

    for sub in os.listdir(root):

        sub_path = os.path.join(root, sub)

        if not os.path.isdir(sub_path):
            continue

        data[sub] = {"segments": []}

        for file in os.listdir(sub_path):

            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(sub_path, file)

            img = cv2.imread(path)

            if img is None:
                continue

            data[sub]["segments"].append(img)

    return data


def main():

    data = load_grouped_segments(ROOT)

    labeler = Labeler()

    for sub in data:

        print(f"\nProcessing {sub}")

        labeled = labeler.label_segments(data[sub]["segments"])

        print(f"Labeled: {len(labeled)}")

        data[sub]["labeled"] = labeled

    data = interactive_best_selection_cli(data)

    save_final_outputs(data)

    print("\nDone")


if __name__ == "__main__":
    main()