from ultralytics import YOLO
import cv2
import os

card_model = YOLO('best.pt')       # your new card detector
objects_model = YOLO('detect_odjects.pt')

input_folder = r'D:\GRAD\Project\id_card_training\datasets\train\images'
output_folder = r'D:\GRAD\Project\OCR_Egyptian_ID-main\OCR_Egyptian_ID-main\raw_crops'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    image = cv2.imread(path)

    card_results = card_model(path)[0]
    if len(card_results.boxes) == 0:
        print(f"No card found: {filename}")
        continue
    x1, y1, x2, y2 = map(int, card_results.boxes[0].xyxy[0])
    card_crop = image[y1:y2, x1:x2]

    field_results = objects_model(card_crop)[0]
    for box in field_results.boxes:
        class_id = int(box.cls[0])
        if field_results.names[class_id] == 'nid':
            fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
            nid_crop = card_crop[fy1:fy2, fx1:fx2]
            out_path = os.path.join(output_folder, f'nid_{filename}')
            cv2.imwrite(out_path, nid_crop)
            print(f"Saved: {out_path}")