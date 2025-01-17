from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import cv2

# Chargement du modèle et du processeur
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def detect_objects(frame, threshold=0.9):
    """
    Détecte les objets dans un frame vidéo à l'aide du modèle DETR.
    Args:
        frame (ndarray): Frame vidéo OpenCV.
        threshold (float): Seuil de confiance pour filtrer les résultats.
    Returns:
        list: Liste des objets détectés (bounding boxes, labels, scores).
    """
    # Conversion en format PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Préparer l'entrée
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Extraire logits et bounding boxes
    logits = outputs.logits[0]
    boxes = outputs.pred_boxes[0]

    # Appliquer le softmax et filtrer
    labels = logits.softmax(-1).argmax(-1)
    scores = logits.softmax(-1).max(-1).values

    detected = []
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box.tolist()
            detected.append(((x_min, y_min, x_max, y_max), label.item(), score.item()))

    return detected


