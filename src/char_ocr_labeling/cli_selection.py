import os
import cv2
import numpy as np

TMP_GRID_DIR = "tmp_grids"

def create_grid(candidates, cols=6, cell_size=64):

    total = len(candidates)
    rows = int(np.ceil(total / cols))

    grid = np.ones((rows * cell_size, cols * cell_size), dtype=np.uint8) * 255

    for idx, item in enumerate(candidates):

        r, c = divmod(idx, cols)

        img = cv2.resize(item["img"], (cell_size, cell_size))

        y1, x1 = r * cell_size, c * cell_size

        grid[y1:y1+cell_size, x1:x1+cell_size] = img

        cv2.putText(
            grid,
            str(idx+1),
            (x1+2, y1+12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (128,),
            1
        )

    return grid  # keep grayscale (no need for BGR)


def interactive_best_selection_cli(data):

    print("\nHeadless grid selection started")

    os.makedirs(TMP_GRID_DIR, exist_ok=True)

    for sub in data:

        print(f"\nProcessing: {sub}")

        labeled = data[sub]["labeled"]

        # ===============================
        # GROUP + SORT
        # ===============================
        groups = {}

        for item in labeled:
            groups.setdefault(item["label"], []).append(item)

        for label in groups:
            groups[label].sort(key=lambda x: x["score"], reverse=True)

        selected = []

        # ===============================
        # PER LABEL SELECTION
        # ===============================
        for label in sorted(groups.keys()):

            candidates = groups[label]

            if not candidates:
                continue

            print(f"\nLabel: {label}")

            grid = create_grid(candidates)

            # SAVE GRID INSTEAD OF SHOW
            grid_path = os.path.join(TMP_GRID_DIR, f"{sub}_{label}.png")
            cv2.imwrite(grid_path, grid)

            print(f"Open this image: {grid_path}")

            total = len(candidates)

            while True:
                choice = input(f"Select (1-{total}) or skip: ").strip().lower()

                if choice == "skip":
                    print(f"Skipped: {label}")
                    break

                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= total:
                        selected.append(candidates[idx-1])
                        print(f"Selected: {idx}/{total}")
                        break

                print("Invalid input")

        data[sub]["best"] = selected

    print("\nSelection complete")

    return data