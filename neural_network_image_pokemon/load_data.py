import os
import numpy as np
from PIL import Image

def load_data(data_dir, image_size=(128, 128)):
    images = []
    labels = []
    classes = sorted(os.listdir(data_dir)) # Nazwa katalogów jako etykiety

    # Mapowanie klas na indeksy
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        
        # Próba otworzenia obrazu i przekonwertowania go na wartosci Red Green Blue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).resize(image_size).convert('RGB')
                images.append(np.array(img))
                labels.append(class_to_idx[cls])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    images = np.array(images, dtype=np.float32) / 255.0 # Normalizacja pikseli
    labels = np.array(labels)

    return images, labels, classes

