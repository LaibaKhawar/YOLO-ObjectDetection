# Grocery Detection and Billing with YOLO-World

Open-vocabulary detection of vegetables in a photo with YOLO-World (via Roboflow `inference`), turning the detected items into a bill, plus a Flask prototype that does the same through a hosted YOLO-World demo and OCR.

**Case study:** [Read it](https://www.laibakhawar.com/work/grocery-detection)

## How it works

### Notebook: `AI_project (1).ipynb` (Google Colab)

1. Installs `inference-gpu[yolo-world]==0.9.12rc1` and `supervision==0.19.0rc3`.
2. Loads `YOLOWorld(model_id="yolo_world/l")` (YOLO-World Large).
3. Sets the open-vocabulary classes with `model.set_classes(["Tomato", "Potato", "Cucumber", "Green capsicum"])`. No training or fine-tuning is done; the class names are the text prompts.
4. Runs `model.infer(image, confidence=0.2)` and applies non-maximum suppression with `sv.Detections.from_inference(results).with_nms(threshold=0.1)`.
5. Draws boxes and labels with supervision's annotators.
6. Collects the detected class names and adds up a price per detection from a hardcoded table (Tomato 12.50, Potato 10.50, Cucumber 10.00, Green capsicum 13.00).

### Flask prototype: `app.py`

- `/` renders `index.html`; `/upload` accepts an image, saves it as `uploaded_image.jpg` and calls `calculate_bill`.
- `calculate_bill` sends the image and the class list to a hosted YOLO-World Gradio Space through `gradio_client` (`https://stevengrove-yolo-world.hf.space/--replicas/89o68/`, endpoint `/partial`, max 4 boxes, min confidence 0.03, NMS 0.5). The Space returns an annotated image.
- EasyOCR then reads the label text drawn on that image, and any label containing a menu word (tomato 20, potato 50, cucumber 200, capsicum 300) adds that price to the bill.
- The result is rendered with `result.html`.

`PROJECT.ipynb` contains earlier versions of the same Flask code (one variant also includes "pizza" in the menu) and the output of starting the development server inside Jupyter.

## Results

Recorded in `AI_project (1).ipynb` for one test image:

- Detected classes (cell 10): Cucumber, Tomato, Potato, Green capsicum.
- `Total bill: 46.0` (cell 11).

There is no labelled test set, so no precision, recall or mAP is reported.

## Repository contents

- `AI_project (1).ipynb`: YOLO-World detection and billing notebook.
- `app.py`: Flask app using the hosted Space and EasyOCR.
- `PROJECT.ipynb`: earlier Flask code in notebook form.
- `example 1.jpg`, `example 3.jpg`, `example 5.jpg`: sample grocery photos.
- `requirements.txt`: Pillow, jupyter, opencv-python, tqdm.

## Running it

### Detection notebook

Designed for Google Colab (Python 3.10). The `inference-gpu` package is installed in the first cells.

1. Open `AI_project (1).ipynb` in Colab and run the install cells.
2. Upload a photo and set `SOURCE_IMAGE_PATH` in cell 5. It is hardcoded to `/content/sample_data/IMG-20240503-WA0053.jpg`, which is not in the repository; one of the `example *.jpg` files can be used instead.
3. Run the remaining cells.

### Flask app

```bash
git clone https://github.com/LaibaKhawar/YOLO-ObjectDetection.git
cd YOLO-ObjectDetection
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install flask easyocr gradio_client
python app.py
```

This will not work without further changes; see Known limitations.

`requirements.txt` does not cover the imports. The notebook needs `inference` (installed as `inference-gpu[yolo-world]`), `supervision` and `opencv-python`; `app.py` needs `flask`, `easyocr`, `gradio_client` and `Pillow`. Of these, only `opencv-python` and `Pillow` are listed. `tqdm` is listed and imported but not used.

## Known limitations

- **Flask templates are missing.** `app.py` renders `index.html` and `result.html`, but there is no `templates/` folder in the repository, so every route fails.
- **Dependency on a hosted Space replica.** `app.py` calls a specific Hugging Face Space replica URL (`.../--replicas/89o68/`). Replica URLs are temporary, so this endpoint is likely unavailable now.
- **OCR-based billing in the app.** The app bills by reading label text off the annotated image with OCR instead of using the detections directly, so OCR errors change the bill.
- **Incomplete `requirements.txt`** (see above).
- **Hardcoded inputs.** The notebook's image path is a Colab path to a file that is not committed, and the prices and class list are hardcoded.
- **Counting is per detection.** The bill adds one unit price per bounding box; there is no weighing or quantity handling, and with NMS at 0.1 overlapping items of the same kind can be merged.
- The notebook and `app.py` use different price tables.

## Author

[Laiba Khawar](https://github.com/LaibaKhawar) · [LinkedIn](https://www.linkedin.com/in/laibakhawar/)
