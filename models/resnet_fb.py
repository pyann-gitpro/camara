
# _________________________________________________________________
# VERSION TRAITEMENT IMAGE detection objet et clasisfication avec model facebook resnet-50 (c'est bof)


# import os
# import sys
# sys.path.insert(0,
#                 os.path.abspath(
#                     os.path.join(
#                         os.path.dirname(__file__),
#                         ".."
#                     )
#                 ))

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch

# Charger le processeur et le modèle pré-entraîné
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Charger une image pour la détection
image_path = "../assets/img_test/eteroa-humpback.webp"  # Remplacez par le chemin de votre image
image = Image.open(image_path).convert("RGB")

# Prétraitement de l'image
inputs = processor(images=image, return_tensors="pt")

# Inférence
with torch.no_grad():
    outputs = model(**inputs)

# Post-traitement : extraction des boîtes de délimitation et des étiquettes
target_sizes = torch.tensor([image.size[::-1]])  # (hauteur, largeur)
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

# Dessiner les résultats sur l'image
draw = ImageDraw.Draw(image)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}", fill="red")

# Afficher ou sauvegarder l'image annotée
image.show()  # Affiche l'image avec les objets détectés
# image.save("output_image.jpg")  # Sauvegarde le résultat





# _________________________________________________________________
# FIRST VERSION

# from transformers import DetrImageProcessor, DetrForObjectDetection
# from PIL import Image
# import torch
# import cv2

# # Chargement du modèle et du processeur
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")



# def detect_objects(frame, threshold=0.9):
#     """
#     Détecte les objets dans un frame vidéo à l'aide du modèle DETR.
#     Args:
#         frame (ndarray): Frame vidéo OpenCV.
#         threshold (float): Seuil de confiance pour filtrer les résultats.
#     Returns:
#         list: Liste des objets détectés (bounding boxes, labels, scores).
#     """
#     # Conversion en format PIL
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Préparer l'entrée
#     inputs = processor(images=image, return_tensors="pt")
#     outputs = model(**inputs)

#     # Extraire logits et bounding boxes
#     logits = outputs.logits[0]
#     boxes = outputs.pred_boxes[0]

#     # Appliquer le softmax et filtrer
#     labels = logits.softmax(-1).argmax(-1)
#     scores = logits.softmax(-1).max(-1).values

#     detected = []
#     for box, label, score in zip(boxes, labels, scores):
#         if score > threshold:
#             x_min, y_min, x_max, y_max = box.tolist()
#             detected.append(((x_min, y_min, x_max, y_max), label.item(), score.item()))

#     return detected


