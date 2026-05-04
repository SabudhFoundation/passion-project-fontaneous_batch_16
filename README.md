[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23178844&assignment_repo_type=AssignmentRepo)

# Fontaneous : Few-Shot Grapheme Extrapolation for Typeface Generation

## Overview

This project implements an end-to-end system that converts handwritten text images into a **custom TTF font file**.

It covers the complete pipeline:

```

Input Images
↓
Preprocessing (denoise, binarization, line removal)
↓
Segmentation (character extraction)
↓
OCR Labeling (character classification)
↓
Scoring + Grouping
↓
User Selection (best character per class)
↓
Dataset Creation
↓
Vectorization (SVG generation)
↓
TTF Font Generation (FontForge)

```



## Python Version

```
Python 3.12.3
````



## Installation

### System dependencies (WSL / Ubuntu required)

```bash
sudo apt-get update
sudo apt-get install -y fontforge python3-fontforge
```

### Python dependencies

```bash
pip install -r requirements.txt
```

### Run application

```bash
streamlit run src/main.py
```



## Project Structure

```
├── README.md
│
├── reports
│   └── figures/                       # pipeline visuals (preprocess → segmentation → OCR)
│
├── src
│   │
│   ├── preprocessing_data
│   │   └── preprocessing.py          # image cleaning + normalization
│   │
│   ├── segmentation
│   │   ├── Segmentation.py           # character segmentation logic
│   │   └── tesseract/                # (moved from root for clarity)
│   │
│   ├── char_ocr_labeling
│   │   ├── cli_selection.py         # manual/interactive selection
│   │   ├── config.py                # OCR + pipeline configs
│   │   ├── labeling.py              # label assignment logic
│   │   ├── ocr_engine.py            # OCR inference layer
│   │   ├── normalization.py         # character normalization
│   │   ├── scoring.py               # confidence/quality scoring
│   │   ├── utils.py                 # shared helpers
│   │   ├── saver.py                 # dataset persistence layer
│   │   └── main.py                  # OCR pipeline entry point
│   │
│   ├── dataset
│   │   ├── final_dataset/           # structured labeled dataset
│   │   ├── best_chars/              # selected clean samples
│   │   └── best_chars_inverted/     # inverted variants
│   │
│   ├── vectorization
│   │   ├── preprocessing.py         # vectorization preprocessing stage
│   │   └── vectorization.py         # raster → vector conversion (SVG)
│   │
│   ├── font_generation
│   │   ├── step2_svg_to_ttf.py      # SVG → font glyph mapping
│   │   └── ttf_final.py             # final TTF builder/export
│   │
│   └── main.py                      # Streamlit / pipeline controller
│
├── require.txt
├── system-requirements.txt
├── .gitignore
└── LICENSE

```



## Pipeline Explanation

### 1. Preprocessing

* Grayscale conversion
* Noise removal
* Adaptive thresholding
* Morphological cleaning
* Line removal using inpainting

Output: clean binary glyph image



### 2. Segmentation

Extracts individual characters from word/image using:

* connected components
* bounding box filtering
* morphological validation

Output: list of character crops



### 3. OCR Labeling

Uses EasyOCR to assign character labels:

* filters invalid predictions
* confidence thresholding
* single-character enforcement

Output:

```
{ img, label, confidence, score }
```



### 4. Scoring System

Each character is ranked using:

* OCR confidence
* fill ratio
* connected components
* aspect ratio
* centroid alignment

Used to select best glyph per class.



### 5. User Selection

A grid-based UI (or CLI fallback):

* groups by label
* sorts by score
* user selects best candidate per class

Output:

```
data[sub]["best"]
```



### 6. Dataset Creation

Generates structured dataset:

```
final_dataset/
   ├── student_1/
   │     ├── capital_a/images/
   │     ├── small_b/images/
```

Also saves:

* best_chars/
* best_chars_inverted/



### 7. Vectorization

Each glyph is converted:

```
PNG → SVG (VTracer)
```

Stored per label for font generation.



### 8. Font Generation

Using FontForge:

* imports SVG glyphs
* normalizes metrics
* applies spacing rules
* handles missing characters
* exports `.ttf`

Output:

```
new_font.ttf
```



## Key Features

* End-to-end handwriting-to-font system
* Robust OCR + scoring pipeline
* Human-in-the-loop selection
* SVG-based vectorization
* Fully automated TTF generation
* Modular and extensible architecture



## Requirements

```
opencv-python
numpy
easyocr
streamlit
fontforge
vtracer
```



## Run Pipeline

### Streamlit UI

```bash
streamlit run src/main.py
```



## Outputs

* `final_dataset/` → full labeled dataset
* `best_chars/` → selected glyphs
* `best_chars_inverted/` → inverted dataset
* `new_font.ttf` → final font

<img width="1897" height="880" alt="image" src="https://github.com/user-attachments/assets/eb42dbb6-d1d4-4e1d-bcd7-1c835d4528ef" />
<br><br>
<img width="1883" height="986" alt="image" src="https://github.com/user-attachments/assets/645e19ec-1def-4f91-8a83-ee60bab9f7f3" />
<br><br>
<img width="1877" height="925" alt="image" src="https://github.com/user-attachments/assets/d4938b46-990c-420f-bdae-f6e6392f5da2" />
<br><br>
<img width="1882" height="960" alt="image" src="https://github.com/user-attachments/assets/787dafdc-f360-4f48-9b4b-8f42100577f1" />
<br><br>
<img width="1882" height="787" alt="image" src="https://github.com/user-attachments/assets/df8a336d-7b40-448e-a065-0d3d110dd7e9" />
<br><br>
<img width="1888" height="955" alt="image" src="https://github.com/user-attachments/assets/b44b5b58-1ef4-4f21-8bbb-c8a91e9204bb" />
<br><br>
<img width="1879" height="991" alt="image" src="https://github.com/user-attachments/assets/1332c588-34f7-4962-90c5-6c6c2c2bcb20" />
<br><br>
<img width="1902" height="658" alt="image" src="https://github.com/user-attachments/assets/99706625-0f03-4591-80e4-d3b9a3d1ac65" />
<br><br>
<img width="1907" height="849" alt="image" src="https://github.com/user-attachments/assets/c045c30a-06ef-4cac-8894-a48343614e92" />
<br><br>

## Generated Font Style Output
<br>
<img width="609" height="452" alt="image" src="https://github.com/user-attachments/assets/f1c18b8a-359f-44d0-8429-6a57cbb418f4" />

<br><br>
```
