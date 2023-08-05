import PIL.Image
from glob import glob
import os
from random import shuffle

if __name__ == "__main__":
    
    #for name in glob.glob(, recursive=True):
    path = '/home/yandex/MLFH2023/giladd/hiera/datasets/**/*'
    save_path = '/home/yandex/MLFH2023/giladd/hiera/datasets/processed_images/'
    images_paths = glob(path + '.png', recursive=True) + glob(path + '.jpg', recursive=True) 
    shuffle(images_paths)
    i = 0
    for path in images_paths:
        print(path)          
        name = os.path.basename(path)    
        image = PIL.Image.open(path)
        # Resize image
        image = image.convert('RGB')
        image = image.resize((224, 224))
        # Save image
        image.save(os.path.join(save_path, name))
        i += 1
        if i == 10:
            break