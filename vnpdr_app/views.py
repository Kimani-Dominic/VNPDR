# views.py
import cv2
import pytesseract
from django.shortcuts import render
from django.http import HttpResponse
import numpy as np

# License plate detection function using YOLOv3
def detect_license_plate(image):
    # Load pre-trained YOLOv3 model
    model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    # Resize image to match model input size
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    outputs = model.forward(output_layers)

    # Extract bounding boxes for detected cars
    detected_plate_boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class ID 2 represents 'car'
                # Get bounding box coordinates
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                # Calculate top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detected_plate_boxes.append([x, y, w, h])

    return detected_plate_boxes

# Car analysis function
def analyze_car(plate_region):
    # Perform OCR (Optical Character Recognition) on the license plate region using Tesseract
    plate_text = pytesseract.image_to_string(plate_region, config='--psm 8')
    
    # Dummy values for car analysis (excluding country of origin)
    car_model = "Toyota Camry"
    car_type = "Sedan"
    car_color = "Red"
    
    return {
        'plate_text': plate_text.strip(),
        'car_model': car_model,
        'car_type': car_type,
        'car_color': car_color,
    }

# Upload image view
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get uploaded image file
        image_file = request.FILES['image']

        # Read the uploaded image using OpenCV
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform license plate detection
        plate_boxes = detect_license_plate(img)
        
        # Check if license plate detected
        if plate_boxes:
            # Extract license plate region from the image
            for box in plate_boxes:
                x, y, w, h = box
                plate_region = img[y:y+h, x:x+w]
            
                # Perform car analysis
                car_details = analyze_car(plate_region)
                
                # Return the HTML response with car details in a table format
                return render(request, 'analysis_result.html', {
                    'detected_img': cv2.imencode('.jpg', img)[1].tobytes(),  # Convert image to bytes
                    'plate_text': car_details['plate_text'],
                    'car_model': car_details['car_model'],
                    'car_type': car_details['car_type'],
                    'car_color': car_details['car_color'],
                })
        else:
            return render(request, 'upload.html', {'error': 'License plate not detected'})

    else:
        return render(request, 'upload.html')
