"""
import cv2
from deepface import DeepFace

def capture_selfie(save_path='live_selfie.jpg'):
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    print("Press SPACE to capture, ESC to cancel")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Position your face - SPACE to capture', frame)
        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            cv2.imwrite(save_path, frame)
            print(f"Saved: {save_path}")
            break
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return save_path


# Usage — plugs directly into what you already have
selfie_path = capture_selfie()
import os
print("Selfie exists:", os.path.exists(selfie_path))
print("ID photo exists:", os.path.exists('extracted_photo.png'))
if selfie_path:
    try:
        result = DeepFace.verify(
            img1_path=selfie_path,
            img2_path='extracted_photo.png',
            anti_spoofing=False,
            model_name='ArcFace'
        )
        print("Verified:", result["verified"])
        print("Distance:", result["distance"])
    except ValueError as e:
        import traceback
        traceback.print_exc()
        print("Rejected:", e)
"""     
import cv2
import time
from deepface import DeepFace

ID_PHOTO_PATH = 'extracted_photo.png'
CHECK_INTERVAL = 1.5  # seconds between verification attempts — running every frame would be too slow

cap = cv2.VideoCapture(0)
last_check_time = 0
last_result_text = "Waiting..."
last_result_color = (255, 255, 0)

print("Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_check_time > CHECK_INTERVAL:
        last_check_time = now
        temp_path = 'temp_live_frame.jpg'
        cv2.imwrite(temp_path, frame)

        try:
            result = DeepFace.verify(
                img1_path=temp_path,
                img2_path=ID_PHOTO_PATH,
                anti_spoofing=False,
                enforce_detection=False,  # don't crash if no face is visible in a given frame
                model_name='ArcFace'
            )
            if result["verified"]:
                last_result_text = f"MATCH (distance={result['distance']:.3f})"
                last_result_color = (0, 255, 0)
            else:
                last_result_text = f"NO MATCH (distance={result['distance']:.3f})"
                last_result_color = (0, 0, 255)
        except Exception as e:
            last_result_text = "No face detected"
            last_result_color = (0, 165, 255)

    # Draw the latest result on the live feed, every frame (fast, no model call)
    display_frame = frame.copy()
    cv2.putText(display_frame, last_result_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, last_result_color, 2)
    cv2.imshow('Live Verification - ESC to quit', display_frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()   