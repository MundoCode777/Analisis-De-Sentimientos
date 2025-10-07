#analisis_trimodal.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import _warnings
_warnings.filterwarnings('ignore')

#VERIFICAR DEPENDECIAS NUEVAS
TRANSFORMERS_DISPONIBLE = False
PYSENTIMENT_DISPONIBLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_DISPONIBLE = True
    print("✅ Transformers está disponible")
except ImportError:
    print("⚠️ Transformers no está disponible. Algunas funcionalidades estarán limitadas.")
   
try:
    from pysentimientos import create_analyzer
    PYSENTIMENT_DISPONIBLE = True
    print("✅ PySentimiento está disponible")
except ImportError:
    print("⚠️ PySentimiento no está disponible. Algunas funcionalidades estarán limitadas.")

