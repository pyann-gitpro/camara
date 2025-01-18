from ultralytics import YOLO
import cv2

# Charger un modèle YOLO plus performant
model = YOLO('yolov8s.pt')  # Utiliser 'yolov8m.pt' pour un modèle encore plus performant

# Chemin vers l'image
image_path = "../assets/img_test/eteroa-humpback.webp"

# Redimensionner l'image pour de meilleures performances
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (640, 640))  # Taille standard YOLOv8
cv2.imwrite("resized_image.jpg", resized_image)

# Effectuer la détection avec un seuil de confiance ajusté
results = model.predict(source="resized_image.jpg", conf=0.25)

# Vérifier et afficher les résultats
if results and len(results[0].boxes) > 0:
    annotated_image = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_image)
    cv2.waitKey(0)
else:
    print("Aucune détection effectuée.")

cv2.destroyAllWindows()
