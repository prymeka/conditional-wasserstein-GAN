import numpy as np
from PIL import Image
from pathlib import Path
import os

dir = Path('./imgs')
files = [f for f in os.listdir(dir) if f.endswith('.npy')]

for file in files:
    arr = np.load(dir/file).reshape((28, 28))
    img = Image.fromarray(arr)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    filename = str((dir / Path(file).stem).resolve())
    img.save(filename+'.png')
    os.remove(filename+'.npy')
