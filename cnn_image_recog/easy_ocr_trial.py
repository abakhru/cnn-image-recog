#!/usr/bin/env python

"""
- https://github.com/JaidedAI/EasyOCR
"""

import easyocr
from pathlib import Path

english = 'ML-workflow-diagram.jpg'
hindi = 'Screen Shot 2020-07-17 at 4.30.45 PM.png'
e = 'IMG_0011.JPG'

base_dir = Path('~').expanduser().joinpath('Desktop')
f = base_dir / e
reader = easyocr.Reader(['en'], gpu=False)
t = reader.readtext(image=f.read_bytes(), detail=False)
print(t)
