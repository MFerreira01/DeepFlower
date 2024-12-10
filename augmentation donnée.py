import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import save_img

# Configuration des répertoires
input_dir = 'C:\\Users\\maria\\Documents\\Ipsi3\\Image\\DeepLearning\\Projet\\DeepFlower\\Flowers\\Train'  # Répertoire des images d'origine
output_dir = 'C:\\Users\\maria\\Documents\\Ipsi3\\Image\\DeepLearning\\Projet\\DeepFlower\\Flowers\\Aug'  # Répertoire pour enregistrer les images augmentées

# Créez le répertoire de sortie si nécessaire
os.makedirs(output_dir, exist_ok=True)

# Paramètres d'augmentation des données
datagen = ImageDataGenerator(
    rotation_range=30,  # Rotation aléatoire des images
    width_shift_range=0.2,  # Décalage horizontal
    height_shift_range=0.2,  # Décalage vertical
    shear_range=0.2,  # Transformation en cisaillement
    zoom_range=0.2,  # Zoom aléatoire
    horizontal_flip=True,  # Retourner horizontalement
    fill_mode='nearest'  # Remplissage des pixels vides
)

# Parcours des classes dans le répertoire d'entrée
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)
    
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            img_path = os.path.join(class_path, image_name)
            img = load_img(img_path)  # Charger l'image
            img_array = img_to_array(img)  # Convertir en tableau numpy
            img_array = img_array.reshape((1,) + img_array.shape)  # Redimensionner pour l'augmentation
            
            # Générer des images augmentées
            i = 0
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_class_path,
                                      save_prefix='aug', save_format='jpeg'):
                i += 1
                if i >= 20:  # Générer 20 images augmentées par image d'origine
                    break

print("Augmentation des données terminée.")
