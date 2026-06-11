# README

Project Overview

This project trains and evaluates image classification models using EfficientNet-B3 and Vision Transformer (ViT-B/16). The notebook contains code for dataset preparation, model training, validation, testing, and inference.

The final evaluation uses two trained model weight files:

- `final_effnet.pt`
- `final_vit.pt`

These files are not stored in the GitHub repository because they are too big. They must be downloaded separately before running inference.

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision timm pandas numpy matplotlib scikit-learn
```

## Installation

1. Clone or download this repository.

```bash
git clone <your-repository-link>
cd kaggle_project
```

2. Install the required Python packages.

```bash
pip install torch torchvision timm pandas numpy matplotlib scikit-learn
```

3. Download the trained model weights from Google Drive:

```text
https://drive.google.com/drive/folders/1W6oLP4Mufghu0NVnE0vcCgPsLT7sZv-j?usp=drive_link
```

4. Place the downloaded weight files in the main project folder.

Your project folder should look like this:

```text
kaggle_project/
  cse_144_final_75.ipynb
  final_effnet.pt
  final_vit.pt
  README.md
```

Do not place the final model weights inside a `checkpoints/` folder unless you also update the file paths in the notebook.

## Dataset Structure

Place the dataset in the following format:

```text
dataset/
  train/
    0/
    1/
    2/
    ...
    99/
  test/
    image_1.jpg
    image_2.jpg
    ...
```

Note: In this project, the training folders are numeric class labels from `0` to `99`.

## Training

Open and run all cells in:

```text
cse_144_final_75.ipynb
```

The notebook will:

1. Load and preprocess the dataset.
2. Train EfficientNet-B3 and ViT-B/16 models.
3. Evaluate model performance on the validation set.
4. Save final model weights as:

```text
final_effnet.pt
final_vit.pt
```

Training history is saved as:

```text
training_history.csv
```

## Inference

Before running inference, make sure the downloaded weight files are in the main project folder:

```text
final_effnet.pt
final_vit.pt
```

Load a trained model weight file:

```python
model.load_state_dict(torch.load("final_effnet.pt", map_location=device))
model.eval()
```

or

```python
model.load_state_dict(torch.load("final_vit.pt", map_location=device))
model.eval()
```

Run inference on an image:

```python
image = preprocess(image_path)

with torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
    prediction = output.argmax(dim=1)
```

The predicted class corresponds to the numeric class mapping used during training.

## Saved Models

The recommended model files for evaluation are available here:

```text
https://drive.google.com/drive/folders/1W6oLP4Mufghu0NVnE0vcCgPsLT7sZv-j?usp=drive_link
```

Download the following files:

- `final_effnet.pt`
- `final_vit.pt`

Place both files in the main project folder before running inference.