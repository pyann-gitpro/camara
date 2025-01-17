import cv2
from models.resnet_fb import detect_objects

def run_video_detection():
    """
    Lance la détection vidéo en temps réel avec OpenCV.
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Détection d'objets
        detected = detect_objects(frame)

        # Dessiner les bounding boxes sur le frame
        for box, label, score in detected:
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, f"Label: {label} ({score:.2f})",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Afficher le résultat
        cv2.imshow('Detection', frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
