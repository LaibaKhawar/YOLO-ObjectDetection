from flask import Flask, render_template, request
from PIL import Image
import easyocr
import os
from gradio_client import Client
import base64

app = Flask(__name__)

# Function to encode image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string

# Menu
MENU = {
    "tomato": 20,
    "potato": 50,
    "cucumber": 200,
    "capsicum": 300,
    
}

# Hardcoded items to detect
items_to_detect = ["Tomato", "Potato","Cucumber","Green capsicum"]

# Function to calculate total bill
def calculate_bill(image_path, items):
    if os.path.exists(image_path):
        # Getting bounding boxes on the image
        client = Client("https://stevengrove-yolo-world.hf.space/--replicas/89o68/")
        image_with_boxes = client.predict(
            image_path,
            items,
            4,   # max boxes
            0.03, # min confidence
            0.5,  # nms value
            api_name="/partial"
        )

        # Performing OCR on bounding boxes
        reader = easyocr.Reader(['en'])
        ocr_results = reader.readtext(image_with_boxes, paragraph=False)

        total_bill = 0
        recognized_items = {}

        for result in ocr_results:
            bbox_text = str(result[1])
            bbox_text = bbox_text.lower()  # Convert to lowercase for case-insensitive matching
            for menu_item in MENU:
                if menu_item in bbox_text:
                    # Increment the count of detected items
                    if menu_item in recognized_items:
                        recognized_items[menu_item] += 1
                    else:
                        recognized_items[menu_item] = 1
                    # Add the price of the item to the total bill
                    total_bill += MENU[menu_item]

        return recognized_items, total_bill, image_with_boxes
    else:
        print("File path doesn't exist")
        exit()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file:
        image_path = "uploaded_image.jpg"
        file.save(image_path)
        recognized_items, total_bill, output_image = calculate_bill(image_path, items_to_detect)
        os.remove(image_path)
        output_image_base64 = image_to_base64(output_image)
        return render_template('result.html', recognized_items=recognized_items, total_bill=total_bill, output_image=output_image_base64)
    return render_template('index.html', error='Error processing file')

if __name__ == '__main__':
    app.run(debug=True)
