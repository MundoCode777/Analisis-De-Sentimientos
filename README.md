# 📊 Analizador de Sentimientos Multimétodo

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange?logo=ai&logoColor=white)]()
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-success?logo=tensorflow&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white)](./LICENSE)

Una **suite avanzada de análisis de sentimientos** que compara múltiples algoritmos de Machine Learning y NLP para lograr **máxima precisión y confiabilidad**, con enfoque especial en el idioma **español**.

---

## 🚀 Características

### 🔍 Métodos de Análisis Implementados

| Método          | Tipo                   | Especialidad                  | Precisión    |
|----------------|------------------------|-------------------------------|--------------|
| 🧠 TextBlob     | Análisis léxico         | Análisis general rápido       | ★★★☆☆        |
| 🗣️ Pysentimiento | Transformer (Español)   | Optimizado para español       | ★★★★☆        |
| 🤖 Transformers | Modelos avanzados       | Contextos complejos           | ★★★★★        |
| 📏 VADER        | Léxico + Reglas         | Lenguaje informal / redes     | ★★★☆☆        |
| 📊 Naive Bayes  | Clasificador estadístico| Datos personalizados          | ★★★★☆        |

---

### 📁 Formatos de Entrada Soportados

- 📝 **Texto** (`.txt`)  
- 📊 **CSV** (`.csv`)  
- 📈 **Excel** (`.xlsx`, `.xls`)  
- 📄 **Word** (`.docx`)  
- 📕 **PDF** (`.pdf`)  
- 🧾 **JSON** (`.json`)  
- 📋 **TSV** (`.tsv`)  

---

### ⚙️ Funcionalidades Principales

- 📊 Comparación simultánea entre múltiples métodos de análisis  
- 🔄 Análisis de concordancia entre algoritmos  
- 🎯 Consenso inteligente para máxima confiabilidad  
- 🧹 Limpieza inteligente de texto, preservando jerga y expresiones coloquiales  
- ✍️ Corrección ortográfica automática, sin afectar el tono emocional  
- 📉 Visualización avanzada: gráficos comparativos, radar, etc.  
- 💾 Exportación de resultados en múltiples formatos  

---

## 📦 Instalación

### 🔧 Prerrequisitos

- ✅ Python 3.8 o superior  
- 💾 Al menos 2 GB de espacio libre (para modelos preentrenados)  
- 🌐 Conexión a internet (para descarga de modelos y recursos)

---

### 📄 Contenido del `requirements.txt`

```txt
pandas>=1.4.0
numpy>=1.21.0
textblob>=0.17.1
matplotlib>=3.5.0
seaborn>=0.11.0
openpyxl>=3.0.0
xlrd>=2.0.0
python-docx>=0.8.11
PyPDF2>=2.0.0
pyspellchecker>=0.7.0
pysentimiento>=0.7.0
transformers>=4.25.0
scikit-learn>=1.2.0
nltk>=3.7
torch>=2.0.0
