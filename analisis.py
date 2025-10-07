#analisis.py
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import json
import csv
from io import StringIO
import re
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejorar la visualización
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'DejaVu Sans'  # Fuente que maneja bien el español

# Importaciones opcionales que pueden fallar
try:
    import docx
    DOCX_DISPONIBLE = True
except ImportError:
    DOCX_DISPONIBLE = False

try:
    import PyPDF2
    PDF_DISPONIBLE = True
except ImportError:
    PDF_DISPONIBLE = False

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_DISPONIBLE = True
except ImportError:
    SPELLCHECKER_DISPONIBLE = False

# Nuevas librerías para análisis avanzado
try:
    from pysentimiento import create_analyzer
    PYSENTIMIENTO_DISPONIBLE = True
except ImportError:
    PYSENTIMIENTO_DISPONIBLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_DISPONIBLE = True
except ImportError:
    TRANSFORMERS_DISPONIBLE = False

# NUEVAS IMPORTACIONES PARA VADER Y NAIVE BAYES
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    # Descargar recursos de NLTK si no están disponibles
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    VADER_DISPONIBLE = True
except ImportError:
    VADER_DISPONIBLE = False

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import Pipeline
    SKLEARN_DISPONIBLE = True
except ImportError:
    SKLEARN_DISPONIBLE = False


class AnalizadorSentimientos:
    def __init__(self):
        self.datos = None
        self.datos_originales = None
        self.resultados = None
        self.correcciones_realizadas = {}
        
        # Inicializar corrector ortográfico si está disponible
        if SPELLCHECKER_DISPONIBLE:
            self.spell = SpellChecker(language='es')
            # Agregar palabras especiales al diccionario para preservar insultos/jerga
            self.preservar_palabras = {
                # Insultos comunes que no deben corregirse
                'maldito', 'idiota', 'estúpido', 'tonto', 'bobo', 'pendejo', 'cabron', 
                'jodido', 'pinche', 'chingado', 'madre', 'puta', 'hijo', 'cabrón',
                # Jerga común
                'chido', 'padrísimo', 'genial', 'súper', 'mega', 'ultra',
                # Expresiones emotivas
                'jajaja', 'jeje', 'wow', 'omg', 'wtf', 'lol', 'xd'
            }
            self.spell.word_frequency.load_words(self.preservar_palabras)

        # Inicializar analizadores avanzados
        self.pysentimiento_analyzer = None
        self.transformers_analyzer = None
        self._inicializar_analizadores_avanzados()
        
        # NUEVOS ANALIZADORES: VADER y Naive Bayes
        self.vader_analyzer = None
        self.naive_bayes_model = None
        self._inicializar_vader_naive_bayes()

    def _inicializar_analizadores_avanzados(self):
        """Inicializa los analizadores de pysentimiento y transformers"""
        print("Inicializando analizadores avanzados...")
        
        # Inicializar pysentimiento
        if PYSENTIMIENTO_DISPONIBLE:
            try:
                self.pysentimiento_analyzer = create_analyzer(task="sentiment", lang="es")
                print("Pysentimiento inicializado correctamente")
            except Exception as e:
                print(f"Error al inicializar pysentimiento: {str(e)}")
                self.pysentimiento_analyzer = None
        
        # Inicializar transformers con manejo mejorado
        if TRANSFORMERS_DISPONIBLE:
            try:
                # Usar un modelo más simple y confiable para español
                self.transformers_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                print("Transformers inicializado correctamente")
            except Exception as e:
                print(f"Error al inicializar transformers: {str(e)}")
                # Fallback a un modelo más básico
                try:
                    self.transformers_analyzer = pipeline(
                        "sentiment-analysis",
                        return_all_scores=True
                    )
                    print("Transformers inicializado con modelo básico")
                except Exception as e2:
                    print(f"Error al inicializar transformers con modelo básico: {str(e2)}")
                    self.transformers_analyzer = None

    def _inicializar_vader_naive_bayes(self):
        """NUEVO: Inicializa VADER y prepara modelo Naive Bayes"""
        print("Inicializando VADER y Naive Bayes...")
        
        # Inicializar VADER
        if VADER_DISPONIBLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                print("VADER inicializado correctamente")
            except Exception as e:
                print(f"Error al inicializar VADER: {str(e)}")
                self.vader_analyzer = None
        
        # Preparar modelo Naive Bayes (se entrenará con los datos)
        if SKLEARN_DISPONIBLE:
            try:
                # Crear pipeline para Naive Bayes
                self.naive_bayes_model = Pipeline([
                    ('vectorizer', CountVectorizer(ngram_range=(1, 2), max_features=1000)),
                    ('classifier', BernoulliNB())
                ])
                print("Pipeline de Naive Bayes preparado correctamente")
            except Exception as e:
                print(f"Error al preparar Naive Bayes: {str(e)}")
                self.naive_bayes_model = None

    def cargar_archivo(self, archivo):
        """Carga archivos de múltiples formatos (versión lógica sin GUI)"""
        extension = os.path.splitext(archivo)[1].lower()
        try:
            if extension == '.txt':
                textos = self.leer_txt(archivo)
            elif extension == '.csv':
                textos = self.leer_csv(archivo)
            elif extension in ['.xlsx', '.xls']:
                textos = self.leer_excel(archivo, extension)
            elif extension == '.docx':
                if DOCX_DISPONIBLE:
                    textos = self.leer_docx(archivo)
                else:
                    raise ValueError("Soporte para archivos .docx no disponible. Instale python-docx")
            elif extension == '.pdf':
                if PDF_DISPONIBLE:
                    textos = self.leer_pdf(archivo)
                else:
                    raise ValueError("Soporte para archivos .pdf no disponible. Instale PyPDF2")
            elif extension == '.json':
                textos = self.leer_json(archivo)
            elif extension == '.tsv':
                textos = self.leer_tsv(archivo)
            else:
                # Intentar como archivo de texto
                textos = self.leer_txt(archivo)

            if not textos:
                raise ValueError("No se encontraron textos válidos en el archivo")

            self.datos_originales = pd.DataFrame({'texto': textos})
            self.datos = self.limpiar_datos(self.datos_originales.copy())
            return True, "Archivo cargado exitosamente", len(self.datos)
        except Exception as e:
            return False, f"Error al cargar el archivo: {str(e)}", 0

    def leer_txt(self, archivo):
        """Lee archivos de texto"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(archivo, 'r', encoding=encoding) as f:
                    contenido = f.read()
                return [linea.strip() for linea in contenido.split('\n') if linea.strip()]
            except UnicodeDecodeError:
                continue
        raise ValueError("No se pudo decodificar el archivo de texto")

    def leer_csv(self, archivo):
        """Lee archivos CSV"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separadores = [',', ';', '\t']
        for encoding in encodings:
            for sep in separadores:
                try:
                    df = pd.read_csv(archivo, encoding=encoding, sep=sep)
                    for col in df.columns:
                        if df[col].dtype == 'object' and not df[col].isna().all():
                            return df[col].dropna().astype(str).tolist()
                except:
                    continue
        raise ValueError("No se pudo leer el archivo CSV")

    def leer_excel(self, archivo, extension):
        """Lee archivos Excel"""
        try:
            if extension == '.xlsx':
                df = pd.read_excel(archivo, engine='openpyxl')
            else:
                df = pd.read_excel(archivo, engine='xlrd')
            for col in df.columns:
                if df[col].dtype == 'object' and not df[col].isna().all():
                    return df[col].dropna().astype(str).tolist()
        except Exception as e:
            raise ValueError(f"Error al leer Excel: {str(e)}")

    def leer_docx(self, archivo):
        """Lee archivos Word"""
        if not DOCX_DISPONIBLE:
            raise ValueError("python-docx no está disponible")
        try:
            doc = docx.Document(archivo)
            textos = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    textos.append(paragraph.text.strip())
            return textos
        except Exception as e:
            raise ValueError(f"Error al leer Word: {str(e)}")

    def leer_pdf(self, archivo):
        """Lee archivos PDF"""
        if not PDF_DISPONIBLE:
            raise ValueError("PyPDF2 no está disponible")
        try:
            textos = []
            with open(archivo, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                    textos.extend(paragraphs)
            return textos
        except Exception as e:
            raise ValueError(f"Error al leer PDF: {str(e)}")

    def leer_json(self, archivo):
        """Lee archivos JSON"""
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                data = json.load(f)
            textos = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        textos.append(item)
                    elif isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str) and len(value) > 10:
                                textos.append(value)
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 10:
                        textos.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                textos.append(item)
            return textos
        except Exception as e:
            raise ValueError(f"Error al leer JSON: {str(e)}")

    def leer_tsv(self, archivo):
        """Lee archivos TSV (Tab Separated Values)"""
        try:
            df = pd.read_csv(archivo, sep='\t', encoding='utf-8')
            for col in df.columns:
                if df[col].dtype == 'object' and not df[col].isna().all():
                    return df[col].dropna().astype(str).tolist()
        except Exception as e:
            raise ValueError(f"Error al leer TSV: {str(e)}")

    def limpiar_datos(self, datos):
        """Limpia y prepara los datos"""
        datos = datos[datos['texto'].str.len() >= 5]
        datos['texto'] = datos['texto'].str.strip()
        datos['texto'] = datos['texto'].str.replace(r'\s+', ' ', regex=True)
        datos = datos.drop_duplicates(subset=['texto'])
        datos = datos.reset_index(drop=True)
        return datos

    def limpiar_ortografia(self):
        """Limpia palabras mal escritas preservando insultos y jerga"""
        if self.datos is None:
            return False, "Primero carga un archivo", {}
        
        if not SPELLCHECKER_DISPONIBLE:
            return False, "SpellChecker no está disponible. Instala: pip install pyspellchecker", {}

        try:
            self.correcciones_realizadas = {}
            textos_corregidos = []
            total_correcciones = 0
            
            for idx, texto in enumerate(self.datos['texto']):
                texto_corregido, correcciones_texto = self._corregir_texto(texto)
                textos_corregidos.append(texto_corregido)
                
                if correcciones_texto:
                    self.correcciones_realizadas[idx] = correcciones_texto
                    total_correcciones += len(correcciones_texto)
            
            # Actualizar los datos con las correcciones
            self.datos['texto_original'] = self.datos['texto'].copy()
            self.datos['texto'] = textos_corregidos
            
            return True, f"Limpieza completada. {total_correcciones} correcciones realizadas", self.correcciones_realizadas
        
        except Exception as e:
            return False, f"Error durante la limpieza: {str(e)}", {}

    def _corregir_texto(self, texto):
        """Corrige un texto individual preservando palabras especiales"""
        if pd.isna(texto) or texto == '':
            return texto, {}
        
        # Patrones a preservar (URLs, emails, hashtags, menciones, etc.)
        patrones_preservar = [
            r'https?://\S+',  # URLs
            r'\S+@\S+\.\S+',  # Emails
            r'#\w+',          # Hashtags
            r'@\w+',          # Menciones
            r'\d+',           # Números
        ]
        
        # Extraer patrones a preservar
        preservados = {}
        texto_trabajo = texto
        for i, patron in enumerate(patrones_preservar):
            matches = re.findall(patron, texto_trabajo)
            for j, match in enumerate(matches):
                placeholder = f"__PRESERVE_{i}_{j}__"
                preservados[placeholder] = match
                texto_trabajo = texto_trabajo.replace(match, placeholder, 1)
        
        # Procesar palabra por palabra
        palabras = texto_trabajo.split()
        palabras_corregidas = []
        correcciones = {}
        
        for palabra in palabras:
            # Si es un placeholder preservado, mantenerlo
            if palabra in preservados:
                palabras_corregidas.append(palabra)
                continue
            
            # Limpiar palabra de puntuación para análisis
            palabra_limpia = re.sub(r'[^\w]', '', palabra.lower())
            
            # Si la palabra está vacía después de limpiar, mantener original
            if not palabra_limpia:
                palabras_corregidas.append(palabra)
                continue
            
            # Si la palabra está en nuestro diccionario de preservar, no corregir
            if palabra_limpia in self.preservar_palabras:
                palabras_corregidas.append(palabra)
                continue
            
            # Verificar si la palabra necesita corrección
            if palabra_limpia not in self.spell:
                # Obtener sugerencias
                candidatos = self.spell.candidates(palabra_limpia)
                if candidatos:
                    mejor_candidato = list(candidatos)[0]
                    # Solo corregir si la sugerencia es significativamente diferente
                    # pero no demasiado (evitar cambios drásticos)
                    if (mejor_candidato != palabra_limpia and 
                        len(mejor_candidato) >= len(palabra_limpia) - 2 and
                        len(mejor_candidato) <= len(palabra_limpia) + 2):
                        
                        # Preservar capitalización original
                        if palabra[0].isupper():
                            palabra_corregida = mejor_candidato.capitalize()
                        else:
                            palabra_corregida = mejor_candidato
                        
                        # Preservar puntuación original
                        puntuacion_final = re.findall(r'[^\w]+$', palabra)
                        if puntuacion_final:
                            palabra_corregida += puntuacion_final[0]
                        
                        correcciones[palabra] = palabra_corregida
                        palabras_corregidas.append(palabra_corregida)
                    else:
                        palabras_corregidas.append(palabra)
                else:
                    palabras_corregidas.append(palabra)
            else:
                palabras_corregidas.append(palabra)
        
        # Reconstruir texto
        texto_corregido = ' '.join(palabras_corregidas)
        
        # Restaurar patrones preservados
        for placeholder, original in preservados.items():
            texto_corregido = texto_corregido.replace(placeholder, original)
        
        return texto_corregido, correcciones

    def analizar_sentimientos(self):
        """Realiza análisis de sentimientos con múltiples métodos y comparación"""
        if self.datos is None:
            return False, "Primero carga un archivo"

        try:
            print("Iniciando análisis comparativo de sentimientos...")
            
            # TextBlob Analysis
            print("Analizando con TextBlob...")
            textblob_results = self._analizar_textblob()
            
            # Pysentimiento Analysis
            pysentimiento_results = None
            if self.pysentimiento_analyzer:
                print("Analizando con Pysentimiento...")
                pysentimiento_results = self._analizar_pysentimiento()
            
            # Transformers Analysis
            transformers_results = None
            if self.transformers_analyzer:
                print("Analizando con Transformers...")
                transformers_results = self._analizar_transformers()
            
            # NUEVOS ANÁLISIS: VADER y Naive Bayes
            vader_results = None
            if self.vader_analyzer:
                print("Analizando con VADER...")
                vader_results = self._analizar_vader()
            
            naive_bayes_results = None
            if self.naive_bayes_model and len(self.datos) >= 10:
                print("Analizando con Naive Bayes...")
                naive_bayes_results = self._analizar_naive_bayes()
            
            # Agregar resultados al DataFrame
            self._agregar_resultados_dataframe(textblob_results, pysentimiento_results, 
                                             transformers_results, vader_results, naive_bayes_results)
            
            print("Análisis comparativo completado")
            return True, "Análisis completado con múltiples métodos"
            
        except Exception as e:
            return False, f"Error en el análisis: {str(e)}"

    def _analizar_textblob(self):
        """Análisis con TextBlob"""
        def clasificar_sentimiento_textblob(texto):
            if pd.isna(texto) or texto == '':
                return 'neutro', 0.0, 0.0, 0.0
            
            blob = TextBlob(str(texto))
            polaridad = blob.sentiment.polarity
            subjetividad = blob.sentiment.subjectivity
            intensidad = abs(polaridad)
            
            if polaridad >= 0.5:
                return 'muy_positivo', polaridad, subjetividad, intensidad
            elif polaridad >= 0.1:
                return 'positivo', polaridad, subjetividad, intensidad
            elif polaridad <= -0.5:
                return 'muy_negativo', polaridad, subjetividad, intensidad
            elif polaridad <= -0.1:
                return 'negativo', polaridad, subjetividad, intensidad
            else:
                return 'neutro', polaridad, subjetividad, intensidad

        resultados = []
        for texto in self.datos['texto']:
            resultado = clasificar_sentimiento_textblob(texto)
            resultados.append(resultado)
        
        return resultados

    def _analizar_pysentimiento(self):
        """Análisis con Pysentimiento"""
        if not self.pysentimiento_analyzer:
            return None
        
        resultados = []
        for texto in self.datos['texto']:
            if pd.isna(texto) or texto == '':
                resultados.append(('neutro', 0.0, 0.0))
                continue
            
            try:
                resultado = self.pysentimiento_analyzer.predict(str(texto))
                sentimiento = resultado.output.lower()
                confianza = resultado.probas[resultado.output]
                
                # Mapear sentimientos de pysentimiento a nuestro sistema
                if sentimiento in ['pos', 'positive']:
                    if confianza > 0.8:
                        sent_final = 'muy_positivo'
                    else:
                        sent_final = 'positivo'
                elif sentimiento in ['neg', 'negative']:
                    if confianza > 0.8:
                        sent_final = 'muy_negativo'
                    else:
                        sent_final = 'negativo'
                else:
                    sent_final = 'neutro'
                
                # Convertir confianza a polaridad (-1 a 1)
                if sent_final in ['muy_positivo', 'positivo']:
                    polaridad = confianza
                elif sent_final in ['muy_negativo', 'negativo']:
                    polaridad = -confianza
                else:
                    polaridad = 0.0
                
                resultados.append((sent_final, polaridad, confianza))
            except Exception as e:
                print(f"Error en pysentimiento para texto: {str(e)[:50]}...")
                resultados.append(('neutro', 0.0, 0.0))
        
        return resultados

    def _analizar_transformers(self):
        """Análisis con Transformers - VERSIÓN CORREGIDA"""
        if not self.transformers_analyzer:
            return None
        
        resultados = []
        for texto in self.datos['texto']:
            if pd.isna(texto) or texto == '':
                resultados.append(('neutro', 0.0, 0.0))
                continue
            
            try:
                # Truncar texto si es muy largo (límite del modelo)
                texto_truncado = str(texto)[:512]
                resultado = self.transformers_analyzer(texto_truncado)
                
                # CORRECCIÓN: Verificar si resultado es una lista de diccionarios
                if isinstance(resultado, list) and len(resultado) > 0:
                    # Si return_all_scores=True, obtenemos una lista de listas
                    if isinstance(resultado[0], list):
                        scores = resultado[0]  # Tomar la primera predicción
                    else:
                        scores = resultado
                    
                    # Encontrar el score más alto
                    max_score_item = max(scores, key=lambda x: x['score'])
                    label = max_score_item['label'].lower()
                    score = max_score_item['score']
                    
                else:
                    # Fallback si el formato es diferente
                    print(f"Formato inesperado en transformers: {type(resultado)}")
                    resultados.append(('neutro', 0.0, 0.0))
                    continue
                
                # Mapear etiquetas a nuestro sistema de sentimientos
                # Diferentes modelos pueden usar diferentes etiquetas
                if any(palabra in label for palabra in ['positive', 'pos', 'label_2', '4 star', '5 star']):
                    if score > 0.8:
                        sent_final = 'muy_positivo'
                    else:
                        sent_final = 'positivo'
                    polaridad = score
                elif any(palabra in label for palabra in ['negative', 'neg', 'label_0', '1 star', '2 star']):
                    if score > 0.8:
                        sent_final = 'muy_negativo'
                    else:
                        sent_final = 'negativo'
                    polaridad = -score
                else:  # neutral, label_1, 3 star, etc.
                    sent_final = 'neutro'
                    polaridad = 0.0
                
                resultados.append((sent_final, polaridad, score))
                
            except Exception as e:
                print(f"Error en transformers para texto: {str(e)[:100]}...")
                resultados.append(('neutro', 0.0, 0.0))
        
        return resultados

    def _analizar_vader(self):
        """NUEVO: Análisis con VADER"""
        if not self.vader_analyzer:
            return None
        
        resultados = []
        for texto in self.datos['texto']:
            if pd.isna(texto) or texto == '':
                resultados.append(('neutro', 0.0, 0.0))
                continue
            
            try:
                # VADER funciona mejor con inglés, pero podemos probar con español
                scores = self.vader_analyzer.polarity_scores(str(texto))
                compound = scores['compound']
                
                # Clasificar basado en el score compuesto
                if compound >= 0.5:
                    sentimiento = 'muy_positivo'
                elif compound >= 0.1:
                    sentimiento = 'positivo'
                elif compound <= -0.5:
                    sentimiento = 'muy_negativo'
                elif compound <= -0.1:
                    sentimiento = 'negativo'
                else:
                    sentimiento = 'neutro'
                
                # Usar el score compuesto como polaridad
                polaridad = compound
                # La intensidad es el valor absoluto del compuesto
                intensidad = abs(compound)
                
                resultados.append((sentimiento, polaridad, intensidad))
                
            except Exception as e:
                print(f"Error en VADER para texto: {str(e)[:50]}...")
                resultados.append(('neutro', 0.0, 0.0))
        
        return resultados

    def _analizar_naive_bayes(self):
        """NUEVO: Análisis con Naive Bayes (Bernoulli)"""
        if not self.naive_bayes_model or len(self.datos) < 10:
            return None
        
        try:
            # Primero necesitamos entrenar el modelo con los datos existentes
            # Usaremos TextBlob como etiquetas de referencia para entrenamiento
            textos = self.datos['texto'].tolist()
            
            # Crear etiquetas basadas en TextBlob para entrenamiento
            etiquetas = []
            for texto in textos:
                if pd.isna(texto) or texto == '':
                    etiquetas.append('neutro')
                    continue
                
                blob = TextBlob(str(texto))
                polaridad = blob.sentiment.polarity
                
                if polaridad >= 0.1:
                    etiquetas.append('positivo')
                elif polaridad <= -0.1:
                    etiquetas.append('negativo')
                else:
                    etiquetas.append('neutro')
            
            # Entrenar el modelo
            self.naive_bayes_model.fit(textos, etiquetas)
            
            # Predecir con el modelo entrenado
            predicciones = self.naive_bayes_model.predict(textos)
            probabilidades = self.naive_bayes_model.predict_proba(textos)
            
            resultados = []
            for pred, proba in zip(predicciones, probabilidades):
                # Obtener la probabilidad máxima
                confianza = np.max(proba)
                
                # Mapear a nuestro sistema de sentimientos
                if pred == 'positivo':
                    if confianza > 0.8:
                        sent_final = 'muy_positivo'
                    else:
                        sent_final = 'positivo'
                    polaridad = confianza
                elif pred == 'negativo':
                    if confianza > 0.8:
                        sent_final = 'muy_negativo'
                    else:
                        sent_final = 'negativo'
                    polaridad = -confianza
                else:
                    sent_final = 'neutro'
                    polaridad = 0.0
                
                resultados.append((sent_final, polaridad, confianza))
            
            return resultados
            
        except Exception as e:
            print(f"Error en Naive Bayes: {str(e)}")
            return None

    def _agregar_resultados_dataframe(self, textblob_results, pysentimiento_results, 
                                    transformers_results, vader_results, naive_bayes_results):
        """Agrega todos los resultados al DataFrame"""
        # TextBlob
        self.datos['tb_sentimiento'] = [r[0] for r in textblob_results]
        self.datos['tb_polaridad'] = [r[1] for r in textblob_results]
        self.datos['tb_subjetividad'] = [r[2] for r in textblob_results]
        self.datos['tb_intensidad'] = [r[3] for r in textblob_results]
        
        # Pysentimiento
        if pysentimiento_results:
            self.datos['ps_sentimiento'] = [r[0] for r in pysentimiento_results]
            self.datos['ps_polaridad'] = [r[1] for r in pysentimiento_results]
            self.datos['ps_confianza'] = [r[2] for r in pysentimiento_results]
        
        # Transformers
        if transformers_results:
            self.datos['tf_sentimiento'] = [r[0] for r in transformers_results]
            self.datos['tf_polaridad'] = [r[1] for r in transformers_results]
            self.datos['tf_confianza'] = [r[2] for r in transformers_results]
        
        # NUEVOS: VADER
        if vader_results:
            self.datos['vd_sentimiento'] = [r[0] for r in vader_results]
            self.datos['vd_polaridad'] = [r[1] for r in vader_results]
            self.datos['vd_intensidad'] = [r[2] for r in vader_results]
        
        # NUEVOS: Naive Bayes
        if naive_bayes_results:
            self.datos['nb_sentimiento'] = [r[0] for r in naive_bayes_results]
            self.datos['nb_polaridad'] = [r[1] for r in naive_bayes_results]
            self.datos['nb_confianza'] = [r[2] for r in naive_bayes_results]
        
        # Mantener compatibilidad con versión anterior (usar TextBlob como principal)
        self.datos['sentimiento'] = self.datos['tb_sentimiento']
        self.datos['polaridad'] = self.datos['tb_polaridad']
        self.datos['subjetividad'] = self.datos['tb_subjetividad']
        self.datos['intensidad'] = self.datos['tb_intensidad']
        
        # Agregar información adicional
        self.datos['longitud_texto'] = self.datos['texto'].str.len()
        self.datos['num_palabras'] = self.datos['texto'].str.split().str.len()

    def generar_estadisticas(self):
        """Genera estadísticas detalladas comparativas"""
        if self.datos is None:
            return "", "", ""

        # Estadísticas para TextBlob
        conteo_tb = Counter(self.datos['tb_sentimiento'])
        total = len(self.datos)
        
        stats_text = "COMPARACIÓN DE MÉTODOS\n"
        stats_text += "=" * 40 + "\n\n"
        
        # TextBlob Stats
        positivos_tb = conteo_tb.get('positivo', 0) + conteo_tb.get('muy_positivo', 0)
        negativos_tb = conteo_tb.get('negativo', 0) + conteo_tb.get('muy_negativo', 0)
        neutros_tb = conteo_tb.get('neutro', 0)
        
        stats_text += "🔮TextBlob:\n"
        stats_text += f"  Positivos: {positivos_tb} ({(positivos_tb/total)*100:.1f}%)\n"
        stats_text += f"  Negativos: {negativos_tb} ({(negativos_tb/total)*100:.1f}%)\n"
        stats_text += f"  Neutros: {neutros_tb} ({(neutros_tb/total)*100:.1f}%)\n"
        stats_text += f"  Polaridad promedio: {self.datos['tb_polaridad'].mean():.3f}\n\n"
        
        # Pysentimiento Stats
        if 'ps_sentimiento' in self.datos.columns:
            conteo_ps = Counter(self.datos['ps_sentimiento'])
            positivos_ps = conteo_ps.get('positivo', 0) + conteo_ps.get('muy_positivo', 0)
            negativos_ps = conteo_ps.get('negativo', 0) + conteo_ps.get('muy_negativo', 0)
            neutros_ps = conteo_ps.get('neutro', 0)
            
            stats_text += "🔥Pysentimiento:\n"
            stats_text += f"  Positivos: {positivos_ps} ({(positivos_ps/total)*100:.1f}%)\n"
            stats_text += f"  Negativos: {negativos_ps} ({(negativos_ps/total)*100:.1f}%)\n"
            stats_text += f"  Neutros: {neutros_ps} ({(neutros_ps/total)*100:.1f}%)\n"
            stats_text += f"  Confianza promedio: {self.datos['ps_confianza'].mean():.3f}\n\n"
        
        # Transformers Stats
        if 'tf_sentimiento' in self.datos.columns:
            conteo_tf = Counter(self.datos['tf_sentimiento'])
            positivos_tf = conteo_tf.get('positivo', 0) + conteo_tf.get('muy_positivo', 0)
            negativos_tf = conteo_tf.get('negativo', 0) + conteo_tf.get('muy_negativo', 0)
            neutros_tf = conteo_tf.get('neutro', 0)
            
            stats_text += "🤖Transformers:\n"
            stats_text += f"  Positivos: {positivos_tf} ({(positivos_tf/total)*100:.1f}%)\n"
            stats_text += f"  Negativos: {negativos_tf} ({(negativos_tf/total)*100:.1f}%)\n"
            stats_text += f"  Neutros: {neutros_tf} ({(neutros_tf/total)*100:.1f}%)\n"
            stats_text += f"  Confianza promedio: {self.datos['tf_confianza'].mean():.3f}\n\n"
        
        # NUEVO: VADER Stats
        if 'vd_sentimiento' in self.datos.columns:
            conteo_vd = Counter(self.datos['vd_sentimiento'])
            positivos_vd = conteo_vd.get('positivo', 0) + conteo_vd.get('muy_positivo', 0)
            negativos_vd = conteo_vd.get('negativo', 0) + conteo_vd.get('muy_negativo', 0)
            neutros_vd = conteo_vd.get('neutro', 0)
            
            stats_text += "⚡VADER:\n"
            stats_text += f"  Positivos: {positivos_vd} ({(positivos_vd/total)*100:.1f}%)\n"
            stats_text += f"  Negativos: {negativos_vd} ({(negativos_vd/total)*100:.1f}%)\n"
            stats_text += f"  Neutros: {neutros_vd} ({(neutros_vd/total)*100:.1f}%)\n"
            stats_text += f"  Polaridad promedio: {self.datos['vd_polaridad'].mean():.3f}\n\n"
        
        # NUEVO: Naive Bayes Stats
        if 'nb_sentimiento' in self.datos.columns:
            conteo_nb = Counter(self.datos['nb_sentimiento'])
            positivos_nb = conteo_nb.get('positivo', 0) + conteo_nb.get('muy_positivo', 0)
            negativos_nb = conteo_nb.get('negativo', 0) + conteo_nb.get('muy_negativo', 0)
            neutros_nb = conteo_nb.get('neutro', 0)
            
            stats_text += "📊Naive Bayes:\n"
            stats_text += f"  Positivos: {positivos_nb} ({(positivos_nb/total)*100:.1f}%)\n"
            stats_text += f"  Negativos: {negativos_nb} ({(negativos_nb/total)*100:.1f}%)\n"
            stats_text += f"  Neutros: {neutros_nb} ({(neutros_nb/total)*100:.1f}%)\n"
            stats_text += f"  Confianza promedio: {self.datos['nb_confianza'].mean():.3f}\n"

        resumen = self._generar_resumen_comparativo()
        datos_detallados = self._generar_datos_detallados_comparativo()

        return stats_text, resumen, datos_detallados

    def _generar_resumen_comparativo(self):
        """Genera un resumen comparativo detallado"""
        resumen = "ANÁLISIS COMPARATIVO DE SENTIMIENTOS\n"
        resumen += "=" * 60 + "\n\n"
        
        total = len(self.datos)
        
        # Análisis de concordancia
        resumen += "ANÁLISIS DE CONCORDANCIA ENTRE MÉTODOS\n"
        resumen += "-" * 40 + "\n"
        
        # Calcular concordancia entre todos los métodos disponibles
        metodos = ['tb_sentimiento']
        if 'ps_sentimiento' in self.datos.columns:
            metodos.append('ps_sentimiento')
        if 'tf_sentimiento' in self.datos.columns:
            metodos.append('tf_sentimiento')
        if 'vd_sentimiento' in self.datos.columns:
            metodos.append('vd_sentimiento')
        if 'nb_sentimiento' in self.datos.columns:
            metodos.append('nb_sentimiento')
        
        nombres_metodos = {
            'tb_sentimiento': 'TextBlob',
            'ps_sentimiento': 'Pysentimiento',
            'tf_sentimiento': 'Transformers',
            'vd_sentimiento': 'VADER',
            'nb_sentimiento': 'Naive Bayes'
        }
        
        # Calcular concordancia por pares
        for i in range(len(metodos)):
            for j in range(i + 1, len(metodos)):
                metodo1 = metodos[i]
                metodo2 = metodos[j]
                concordancia = (self.datos[metodo1] == self.datos[metodo2]).sum()
                resumen += f"{nombres_metodos[metodo1]} vs {nombres_metodos[metodo2]}: {concordancia}/{total} ({(concordancia/total)*100:.1f}%)\n"
        
        # Consenso total si hay múltiples métodos
        if len(metodos) >= 2:
            # Calcular consenso (todos los métodos coinciden)
            consenso_condicion = True
            for metodo in metodos:
                consenso_condicion = consenso_condicion & (self.datos[metodo] == self.datos[metodos[0]])
            
            consenso_total = consenso_condicion.sum()
            resumen += f"Consenso total ({len(metodos)} métodos): {consenso_total}/{total} ({(consenso_total/total)*100:.1f}%)\n"
        
        return resumen

    def _generar_datos_detallados_comparativo(self):
        """Genera datos detallados con comparación de métodos"""
        datos_detallados = "COMPARACIÓN DETALLADA POR TEXTO\n"
        datos_detallados += "=" * 120 + "\n\n"
        datos_detallados += "Mostrando los primeros 20 registros con análisis comparativo:\n\n"
        
        # Determinar qué métodos están disponibles
        columnas_disponibles = ['tb_sentimiento']
        nombres_columnas = {'tb_sentimiento': 'TextBlob'}
        
        if 'ps_sentimiento' in self.datos.columns:
            columnas_disponibles.append('ps_sentimiento')
            nombres_columnas['ps_sentimiento'] = 'Pysentimiento'
        
        if 'tf_sentimiento' in self.datos.columns:
            columnas_disponibles.append('tf_sentimiento')
            nombres_columnas['tf_sentimiento'] = 'Transformers'
        
        if 'vd_sentimiento' in self.datos.columns:
            columnas_disponibles.append('vd_sentimiento')
            nombres_columnas['vd_sentimiento'] = 'VADER'
        
        if 'nb_sentimiento' in self.datos.columns:
            columnas_disponibles.append('nb_sentimiento')
            nombres_columnas['nb_sentimiento'] = 'NaiveBayes'
        
        # Header
        header = f"{'#':<3} "
        for col in columnas_disponibles:
            header += f"{nombres_columnas[col]:<12} "
        header += f"{'Texto':<40}\n"
        
        datos_detallados += header
        datos_detallados += "=" * 120 + "\n"
        
        # Mostrar datos comparativos
        for i, row in self.datos.head(20).iterrows():
            linea = f"{i+1:<3} "
            for col in columnas_disponibles:
                sent = str(row[col])[:10] + "..." if len(str(row[col])) > 10 else str(row[col])
                linea += f"{sent:<12} "
            
            texto_corto = row['texto'][:35] + "..." if len(row['texto']) > 35 else row['texto']
            linea += f"{texto_corto}\n"
            datos_detallados += linea
        
        if len(self.datos) > 20:
            datos_detallados += f"\n... y {len(self.datos) - 20} registros más.\n"
        
        return datos_detallados

    def mostrar_graficos(self):
        """Crea y muestra gráficos con mejor espaciado y legibilidad"""
        if self.datos is None:
            print("No hay datos para mostrar gráficos.")
            return

        # Determinar cuántos métodos están disponibles
        metodos_disponibles = ['TextBlob']
        if 'ps_sentimiento' in self.datos.columns:
            metodos_disponibles.append('Pysentimiento')
        if 'tf_sentimiento' in self.datos.columns:
            metodos_disponibles.append('Transformers')
        if 'vd_sentimiento' in self.datos.columns:
            metodos_disponibles.append('VADER')
        if 'nb_sentimiento' in self.datos.columns:
            metodos_disponibles.append('Naive Bayes')
        
        num_metodos = len(metodos_disponibles)
        
        # Crear figura con mejor espaciado
        fig_height = 6 * num_metodos  # Reducir altura por fila
        fig = plt.figure(figsize=(20, fig_height))
        
        # Usar constrained layout para mejor espaciado automático
        fig.suptitle('Análisis Comparativo de Sentimientos', 
                    fontsize=18, fontweight='bold', y=0.98)

        # Colores mejorados
        colores = ['#2E8B57', '#FFD700', '#DC143C']
        
        # Crear subplots con más espacio
        for i, metodo in enumerate(metodos_disponibles):
            # Crear 3 subplots para cada método con más espaciado
            ax1 = plt.subplot(num_metodos, 3, i*3 + 1)
            ax2 = plt.subplot(num_metodos, 3, i*3 + 2)
            ax3 = plt.subplot(num_metodos, 3, i*3 + 3)
            
            axes_row = [ax1, ax2, ax3]
            self._crear_graficos_metodo_mejorado(axes_row, metodo, colores, i)
        
        # Ajustar espaciado manualmente si constrained_layout no es suficiente
        plt.tight_layout(rect=[0, 0.02, 1, 0.96], pad=3.0, h_pad=4.0, w_pad=2.0)
        
        plt.show()
        self._mostrar_estadisticas_comparativas()

    def _crear_graficos_metodo_mejorado(self, axes_row, metodo, colores, row_idx):
        """Crea los 3 gráficos con mejor espaciado para un método específico"""
        # Determinar las columnas según el método
        if metodo == 'TextBlob':
            col_sentimiento = 'tb_sentimiento'
            col_polaridad = 'tb_polaridad'
            col_intensidad = 'tb_intensidad'
        elif metodo == 'Pysentimiento':
            col_sentimiento = 'ps_sentimiento'
            col_polaridad = 'ps_polaridad'
            col_intensidad = 'ps_confianza'
        elif metodo == 'Transformers':
            col_sentimiento = 'tf_sentimiento'
            col_polaridad = 'tf_polaridad'
            col_intensidad = 'tf_confianza'
        elif metodo == 'VADER':
            col_sentimiento = 'vd_sentimiento'
            col_polaridad = 'vd_polaridad'
            col_intensidad = 'vd_intensidad'
        elif metodo == 'Naive Bayes':
            col_sentimiento = 'nb_sentimiento'
            col_polaridad = 'nb_polaridad'
            col_intensidad = 'nb_confianza'
        
        # Preparar datos agrupados
        conteo = Counter(self.datos[col_sentimiento])
        positivos = conteo.get('muy_positivo', 0) + conteo.get('positivo', 0)
        neutros = conteo.get('neutro', 0)
        negativos = conteo.get('muy_negativo', 0) + conteo.get('negativo', 0)
        
        categorias = ['Positivo', 'Neutro', 'Negativo']
        cantidades = [positivos, neutros, negativos]
        total = sum(cantidades)
        
        # GRÁFICO 1: Barras Verticales con mejor espaciado
        ax1 = axes_row[0]
        
        if total > 0:
            bars = ax1.bar(categorias, cantidades, color=colores, alpha=0.8, 
                          edgecolor='black', linewidth=1.5, width=0.6)
            
            # Etiquetas con mejor posicionamiento
            for bar, cantidad in zip(bars, cantidades):
                if cantidad > 0:
                    height = bar.get_height()
                    porcentaje = (cantidad / total) * 100
                    # Posicionar texto con más espacio
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max(cantidades)*0.05,
                            f'{cantidad}\n({porcentaje:.1f}%)', 
                            ha='center', va='bottom', fontweight='bold', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    alpha=0.8, edgecolor='gray'))
            
            # Más espacio arriba para las etiquetas
            ax1.set_ylim(0, max(cantidades) * 1.3)
        else:
            ax1.text(0.5, 0.5, 'Sin datos\ndisponibles', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=11, style='italic')
        
        ax1.set_title(f'{metodo} - Distribución', fontweight='bold', fontsize=12, pad=20)
        ax1.set_ylabel('Cantidad de Textos', fontweight='bold', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Rotar etiquetas si es necesario y ajustar márgenes
        ax1.tick_params(axis='x', rotation=0, labelsize=10, pad=5)
        ax1.tick_params(axis='y', labelsize=9)
        
        # Ajustar márgenes del subplot
        pos1 = ax1.get_position()
        ax1.set_position([pos1.x0, pos1.y0, pos1.width, pos1.height * 0.9])

        # GRÁFICO 2: Gráfico Circular con mejor espaciado
        ax2 = axes_row[1]
        if total > 0:
            wedges, texts, autotexts = ax2.pie(cantidades, labels=None, colors=colores, 
                                              autopct='%1.1f%%', startangle=90,
                                              wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                                              textprops={'fontsize': 10, 'fontweight': 'bold'},
                                              pctdistance=0.85)  # Alejar porcentajes del centro
            
            # Mejorar texto de porcentajes
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            # Leyenda posicionada mejor
            ax2.legend(wedges, [f'{cat}: {cant}' for cat, cant in zip(categorias, cantidades)],
                      title="Cantidades", loc="center left", bbox_to_anchor=(1.1, 0.5),
                      fontsize=9, title_fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Sin datos\ndisponibles', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=11, style='italic')
        
        ax2.set_title(f'{metodo} - Proporción', fontweight='bold', fontsize=12, pad=20)

        # GRÁFICO 3: Distribución con mejor espaciado
        ax3 = axes_row[2]
        valores = self.datos[col_intensidad].dropna()
        
        if len(valores) > 0:
            # Histograma con menos bins para evitar aglomeración
            n, bins, patches = ax3.hist(valores, bins=15, alpha=0.7, 
                                       edgecolor='black', linewidth=1)
            
            # Colorear barras
            bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(patches))]
            max_center = max(bin_centers) if bin_centers else 1
            
            for patch, center in zip(patches, bin_centers):
                if metodo == 'TextBlob':
                    if center < 0.2:
                        color = '#E8E8E8'
                    elif center < 0.5:
                        color = '#FFB74D'
                    else:
                        color = '#FF7043'
                else:
                    normalized = center / max_center if max_center > 0 else 0
                    if normalized < 0.33:
                        color = '#FFCDD2'
                    elif normalized < 0.67:
                        color = '#FFB74D'
                    else:
                        color = '#66BB6A'
                patch.set_facecolor(color)
            
            # Líneas de estadísticas con mejor separación
            promedio = valores.mean()
            mediana = valores.median()
            
            ax3.axvline(x=promedio, color='red', linestyle='--', 
                       linewidth=2, alpha=0.8, label=f'Promedio: {promedio:.3f}')
            ax3.axvline(x=mediana, color='blue', linestyle=':', 
                       linewidth=2, alpha=0.8, label=f'Mediana: {mediana:.3f}')
            
            # Leyenda posicionada para no solapar
            ax3.legend(fontsize=9, framealpha=0.9, loc='upper right')
        else:
            ax3.text(0.5, 0.5, 'Sin datos\ndisponibles', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=11, style='italic')
        
        if metodo == 'TextBlob':
            titulo_metrica = 'Intensidad Emocional'
            xlabel = 'Intensidad (0=Neutral, 1=Intenso)'
        elif metodo == 'VADER':
            titulo_metrica = 'Intensidad del Compuesto'
            xlabel = 'Intensidad (-1=Negativo, 1=Positivo)'
        else:
            titulo_metrica = 'Confianza del Modelo'
            xlabel = 'Confianza (0=Baja, 1=Alta)'
            
        ax3.set_title(f'{metodo} - {titulo_metrica}', fontweight='bold', fontsize=12, pad=20)
        ax3.set_xlabel(xlabel, fontweight='bold', fontsize=10, labelpad=8)
        ax3.set_ylabel('Frecuencia', fontweight='bold', fontsize=10, labelpad=8)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_axisbelow(True)
        ax3.tick_params(axis='both', labelsize=9, pad=3)

    def _mostrar_estadisticas_comparativas(self):
        """Muestra estadísticas comparativas detalladas mejoradas"""
        print("\n" + "="*80)
        print("ANÁLISIS COMPARATIVO DE MÉTODOS")
        print("="*80)
        
        total = len(self.datos)
        
        # Análisis por método
        metodos = []
        if 'tb_sentimiento' in self.datos.columns:
            metodos.append(('TextBlob', 'tb_sentimiento', 'tb_polaridad'))
        if 'ps_sentimiento' in self.datos.columns:
            metodos.append(('Pysentimiento', 'ps_sentimiento', 'ps_confianza'))
        if 'tf_sentimiento' in self.datos.columns:
            metodos.append(('Transformers', 'tf_sentimiento', 'tf_confianza'))
        if 'vd_sentimiento' in self.datos.columns:
            metodos.append(('VADER', 'vd_sentimiento', 'vd_polaridad'))
        if 'nb_sentimiento' in self.datos.columns:
            metodos.append(('Naive Bayes', 'nb_sentimiento', 'nb_confianza'))
        
        for nombre, col_sent, col_metric in metodos:
            conteo = Counter(self.datos[col_sent])
            positivos = conteo.get('muy_positivo', 0) + conteo.get('positivo', 0)
            neutros = conteo.get('neutro', 0)
            negativos = conteo.get('muy_negativo', 0) + conteo.get('negativo', 0)
            
            print(f"\n{nombre}:")
            print(f"  Positivos: {positivos:4d} ({(positivos/total)*100:5.1f}%)")
            print(f"  Neutros:   {neutros:4d} ({(neutros/total)*100:5.1f}%)")
            print(f"  Negativos: {negativos:4d} ({(negativos/total)*100:5.1f}%)")
            
            if nombre == 'TextBlob':
                print(f"  Polaridad promedio: {self.datos[col_metric].mean():6.3f}")
            elif nombre == 'VADER':
                print(f"  Compuesto promedio: {self.datos[col_metric].mean():6.3f}")
            else:
                print(f"  Confianza promedio: {self.datos[col_metric].mean():6.3f}")
            
            # Determinar tendencia dominante
            if positivos > neutros and positivos > negativos:
                tendencia = "PREDOMINANTEMENTE POSITIVA"
            elif negativos > positivos and negativos > neutros:
                tendencia = "PREDOMINANTEMENTE NEGATIVA"
            else:
                tendencia = "EQUILIBRADA/NEUTRA"
            
            print(f"  Tendencia: {tendencia}")
        
        # Análisis de concordancia si hay múltiples métodos
        if len(metodos) > 1:
            print(f"\nANÁLISIS DE CONCORDANCIA:")
            print("-" * 40)
            
            # Calcular concordancia por pares
            nombres_columnas = {
                'tb_sentimiento': 'TextBlob',
                'ps_sentimiento': 'Pysentimiento',
                'tf_sentimiento': 'Transformers',
                'vd_sentimiento': 'VADER',
                'nb_sentimiento': 'Naive Bayes'
            }
            
            columnas = [col_sent for _, col_sent, _ in metodos]
            
            for i in range(len(columnas)):
                for j in range(i + 1, len(columnas)):
                    concordancia = (self.datos[columnas[i]] == self.datos[columnas[j]]).sum()
                    print(f"{nombres_columnas[columnas[i]]} vs {nombres_columnas[columnas[j]]}: {concordancia:3d}/{total} ({(concordancia/total)*100:5.1f}%)")
            
            # Consenso total si hay múltiples métodos
            if len(columnas) >= 2:
                consenso_condicion = True
                for col in columnas:
                    consenso_condicion = consenso_condicion & (self.datos[col] == self.datos[columnas[0]])
                
                consenso_total = consenso_condicion.sum()
                print(f"Consenso total ({len(columnas)} métodos): {consenso_total:3d}/{total} ({(consenso_total/total)*100:5.1f}%)")
        
        # Recomendación de método
        print(f"\nRECOMENDACIONES:")
        print("-" * 40)
        print("• TextBlob: Rápido, bueno para análisis general y textos en inglés")
        print("• Pysentimiento: Específico para español, mejor precisión en idioma español")
        print("• Transformers: Modelos más avanzados, mejor para contextos complejos")
        print("• VADER: Especializado en redes sociales y lenguaje informal (inglés)")
        print("• Naive Bayes: Modelo estadístico, se entrena con los datos disponibles")
        
        if len(metodos) > 1:
            print("• Para mayor confiabilidad, considere textos con consenso entre métodos")
        
        print("="*80)

    def exportar_resultados(self, archivo_salida):
        """Exporta resultados comparativos con información de todos los métodos"""
        if self.datos is None:
            return False, "No hay datos para exportar"

        try:
            extension = os.path.splitext(archivo_salida)[1].lower()
            
            if extension == '.xlsx':
                with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
                    # Hoja principal con todos los resultados
                    self.datos.to_excel(writer, sheet_name='Resultados_Completos', index=False)
                    
                    # Hoja de comparación de métodos
                    metodos_data = []
                    
                    # TextBlob
                    conteo_tb = Counter(self.datos['tb_sentimiento'])
                    metodos_data.append(['TextBlob', 'Positivos', conteo_tb.get('muy_positivo', 0) + conteo_tb.get('positivo', 0)])
                    metodos_data.append(['TextBlob', 'Neutros', conteo_tb.get('neutro', 0)])
                    metodos_data.append(['TextBlob', 'Negativos', conteo_tb.get('muy_negativo', 0) + conteo_tb.get('negativo', 0)])
                    metodos_data.append(['TextBlob', 'Polaridad promedio', f"{self.datos['tb_polaridad'].mean():.3f}"])
                    
                    # Pysentimiento
                    if 'ps_sentimiento' in self.datos.columns:
                        conteo_ps = Counter(self.datos['ps_sentimiento'])
                        metodos_data.append(['Pysentimiento', 'Positivos', conteo_ps.get('muy_positivo', 0) + conteo_ps.get('positivo', 0)])
                        metodos_data.append(['Pysentimiento', 'Neutros', conteo_ps.get('neutro', 0)])
                        metodos_data.append(['Pysentimiento', 'Negativos', conteo_ps.get('muy_negativo', 0) + conteo_ps.get('negativo', 0)])
                        metodos_data.append(['Pysentimiento', 'Confianza promedio', f"{self.datos['ps_confianza'].mean():.3f}"])
                    
                    # Transformers
                    if 'tf_sentimiento' in self.datos.columns:
                        conteo_tf = Counter(self.datos['tf_sentimiento'])
                        metodos_data.append(['Transformers', 'Positivos', conteo_tf.get('muy_positivo', 0) + conteo_tf.get('positivo', 0)])
                        metodos_data.append(['Transformers', 'Neutros', conteo_tf.get('neutro', 0)])
                        metodos_data.append(['Transformers', 'Negativos', conteo_tf.get('muy_negativo', 0) + conteo_tf.get('negativo', 0)])
                        metodos_data.append(['Transformers', 'Confianza promedio', f"{self.datos['tf_confianza'].mean():.3f}"])
                    
                    # NUEVO: VADER
                    if 'vd_sentimiento' in self.datos.columns:
                        conteo_vd = Counter(self.datos['vd_sentimiento'])
                        metodos_data.append(['VADER', 'Positivos', conteo_vd.get('muy_positivo', 0) + conteo_vd.get('positivo', 0)])
                        metodos_data.append(['VADER', 'Neutros', conteo_vd.get('neutro', 0)])
                        metodos_data.append(['VADER', 'Negativos', conteo_vd.get('muy_negativo', 0) + conteo_vd.get('negativo', 0)])
                        metodos_data.append(['VADER', 'Compuesto promedio', f"{self.datos['vd_polaridad'].mean():.3f}"])
                    
                    # NUEVO: Naive Bayes
                    if 'nb_sentimiento' in self.datos.columns:
                        conteo_nb = Counter(self.datos['nb_sentimiento'])
                        metodos_data.append(['Naive Bayes', 'Positivos', conteo_nb.get('muy_positivo', 0) + conteo_nb.get('positivo', 0)])
                        metodos_data.append(['Naive Bayes', 'Neutros', conteo_nb.get('neutro', 0)])
                        metodos_data.append(['Naive Bayes', 'Negativos', conteo_nb.get('muy_negativo', 0) + conteo_nb.get('negativo', 0)])
                        metodos_data.append(['Naive Bayes', 'Confianza promedio', f"{self.datos['nb_confianza'].mean():.3f}"])
                    
                    comparacion_df = pd.DataFrame(metodos_data, columns=['Método', 'Métrica', 'Valor'])
                    comparacion_df.to_excel(writer, sheet_name='Comparacion_Metodos', index=False)
                    
                    # Análisis de concordancia
                    columnas_metodos = []
                    nombres_metodos = {}
                    
                    if 'tb_sentimiento' in self.datos.columns:
                        columnas_metodos.append('tb_sentimiento')
                        nombres_metodos['tb_sentimiento'] = 'TextBlob'
                    if 'ps_sentimiento' in self.datos.columns:
                        columnas_metodos.append('ps_sentimiento')
                        nombres_metodos['ps_sentimiento'] = 'Pysentimiento'
                    if 'tf_sentimiento' in self.datos.columns:
                        columnas_metodos.append('tf_sentimiento')
                        nombres_metodos['tf_sentimiento'] = 'Transformers'
                    if 'vd_sentimiento' in self.datos.columns:
                        columnas_metodos.append('vd_sentimiento')
                        nombres_metodos['vd_sentimiento'] = 'VADER'
                    if 'nb_sentimiento' in self.datos.columns:
                        columnas_metodos.append('nb_sentimiento')
                        nombres_metodos['nb_sentimiento'] = 'Naive Bayes'
                    
                    if len(columnas_metodos) > 1:
                        concordancia_data = []
                        total = len(self.datos)
                        
                        # Calcular concordancia por pares
                        for i in range(len(columnas_metodos)):
                            for j in range(i + 1, len(columnas_metodos)):
                                concordancia = (self.datos[columnas_metodos[i]] == self.datos[columnas_metodos[j]]).sum()
                                concordancia_data.append([
                                    f"{nombres_metodos[columnas_metodos[i]]} vs {nombres_metodos[columnas_metodos[j]]}",
                                    concordancia,
                                    f"{(concordancia/total)*100:.1f}%"
                                ])
                        
                        # Consenso total
                        if len(columnas_metodos) >= 2:
                            consenso_condicion = True
                            for col in columnas_metodos:
                                consenso_condicion = consenso_condicion & (self.datos[col] == self.datos[columnas_metodos[0]])
                            
                            consenso_total = consenso_condicion.sum()
                            concordancia_data.append([
                                f"Consenso total ({len(columnas_metodos)} métodos)",
                                consenso_total,
                                f"{(consenso_total/total)*100:.1f}%"
                            ])
                        
                        concordancia_df = pd.DataFrame(concordancia_data, columns=['Comparación', 'Cantidad', 'Porcentaje'])
                        concordancia_df.to_excel(writer, sheet_name='Concordancia', index=False)
            
            elif extension == '.csv':
                self.datos.to_csv(archivo_salida, index=False, encoding='utf-8')
            
            elif extension == '.json':
                resultado_json = {
                    'metadata': {
                        'total_textos': len(self.datos),
                        'fecha_analisis': pd.Timestamp.now().isoformat(),
                        'metodos_utilizados': []
                    },
                    'resultados': self.datos.to_dict('records')
                }
                
                # Agregar métodos utilizados
                if 'tb_sentimiento' in self.datos.columns:
                    resultado_json['metadata']['metodos_utilizados'].append('TextBlob')
                if 'ps_sentimiento' in self.datos.columns:
                    resultado_json['metadata']['metodos_utilizados'].append('Pysentimiento')
                if 'tf_sentimiento' in self.datos.columns:
                    resultado_json['metadata']['metodos_utilizados'].append('Transformers')
                if 'vd_sentimiento' in self.datos.columns:
                    resultado_json['metadata']['metodos_utilizados'].append('VADER')
                if 'nb_sentimiento' in self.datos.columns:
                    resultado_json['metadata']['metodos_utilizados'].append('Naive Bayes')
                
                with open(archivo_salida, 'w', encoding='utf-8') as f:
                    json.dump(resultado_json, f, indent=2, ensure_ascii=False)
            
            elif extension == '.txt':
                with open(archivo_salida, 'w', encoding='utf-8') as f:
                    f.write("RESULTADOS DEL ANÁLISIS COMPARATIVO DE SENTIMIENTOS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # Escribir resumen comparativo
                    resumen = self._generar_resumen_comparativo()
                    f.write(resumen)
                    
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("RESULTADOS DETALLADOS:\n")
                    f.write("=" * 80 + "\n")
                    
                    for i, row in self.datos.iterrows():
                        f.write(f"Texto {i+1}:\n")
                        f.write(f"Contenido: {row['texto']}\n")
                        f.write(f"TextBlob: {row['tb_sentimiento']} (Polaridad: {row['tb_polaridad']:.3f})\n")
                        if 'ps_sentimiento' in row:
                            f.write(f"Pysentimiento: {row['ps_sentimiento']} (Confianza: {row['ps_confianza']:.3f})\n")
                        if 'tf_sentimiento' in row:
                            f.write(f"Transformers: {row['tf_sentimiento']} (Confianza: {row['tf_confianza']:.3f})\n")
                        if 'vd_sentimiento' in row:
                            f.write(f"VADER: {row['vd_sentimiento']} (Compuesto: {row['vd_polaridad']:.3f})\n")
                        if 'nb_sentimiento' in row:
                            f.write(f"Naive Bayes: {row['nb_sentimiento']} (Confianza: {row['nb_confianza']:.3f})\n")
                        f.write("-" * 60 + "\n")
            
            return True, "Exportación comparativa completada"
        except Exception as e:
            return False, f"Error al exportar: {str(e)}"


def verificar_dependencias():
    """Verifica e instala dependencias necesarias incluyendo las nuevas librerías"""
    dependencias_requeridas = [
        'pandas', 'numpy', 'textblob', 'matplotlib', 'seaborn', 
        'openpyxl', 'xlrd'
    ]
    
    dependencias_opcionales = [
        ('python-docx', 'docx', 'Soporte para archivos Word (.docx)'),
        ('PyPDF2', 'PyPDF2', 'Soporte para archivos PDF (.pdf)'),
        ('wordcloud', 'wordcloud', 'Nubes de palabras'),
        ('nltk', 'nltk', 'Procesamiento de texto avanzado'),
        ('pyspellchecker', 'spellchecker', 'Corrección ortográfica automática'),
        ('pysentimiento', 'pysentimiento', 'Análisis de sentimientos específico para español'),
        ('transformers', 'transformers', 'Modelos de transformers para análisis avanzado'),
        ('scikit-learn', 'sklearn', 'Naive Bayes y otros algoritmos de ML'),  # NUEVO
        ('nltk', 'nltk.sentiment.vader', 'VADER Sentiment Analysis')  # NUEVO
    ]
    
    print("Verificando dependencias...")
    
    # Verificar dependencias requeridas
    faltantes = []
    for dep in dependencias_requeridas:
        try:
            __import__(dep)
        except ImportError:
            faltantes.append(dep)
    
    if faltantes:
        print(f"Dependencias requeridas faltantes: {', '.join(faltantes)}")
        print(f"Instala con: pip install {' '.join(faltantes)}")
        return False
    
    print("Todas las dependencias requeridas están instaladas")
    
    # Verificar dependencias opcionales
    disponibles = []
    no_disponibles = []
    
    for pip_name, import_name, descripcion in dependencias_opcionales:
        try:
            if import_name == 'docx':
                import docx
            elif import_name == 'spellchecker':
                from spellchecker import SpellChecker
            elif import_name == 'pysentimiento':
                from pysentimiento import create_analyzer
            elif import_name == 'transformers':
                from transformers import pipeline
            elif import_name == 'sklearn':
                from sklearn.feature_extraction.text import CountVectorizer
            elif import_name == 'nltk.sentiment.vader':
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
            else:
                __import__(import_name)
            disponibles.append(f"{pip_name}: {descripcion}")
        except ImportError:
            no_disponibles.append(f"{pip_name}: {descripcion} (opcional)")
    
    for disponible in disponibles:
        print(f"✓ {disponible}")
    
    if no_disponibles:
        print("\nDEPENDENCIAS OPCIONALES NO DISPONIBLES:")
        for no_disp in no_disponibles:
            print(f"⚠ {no_disp}")
        print("\nPara instalar todas las opcionales:")
        print("pip install python-docx PyPDF2 wordcloud nltk pyspellchecker pysentimiento transformers scikit-learn torch")
        print("\nNota: pysentimiento y transformers requieren modelos adicionales que se descargarán automáticamente")
    
    return True


# Ejemplo de uso
if __name__ == "__main__":
    # Verificar dependencias
    if not verificar_dependencias():
        exit(1)
    
    # Crear instancia del analizador
    analizador = AnalizadorSentimientos()
    
    # Ejemplo de uso básico
    print("\nEjemplo de análisis de sentimientos comparativo:")
    print("-" * 50)
    
    # Simular datos de ejemplo
    textos_ejemplo = [
        "Me encanta este producto, es fantástico",
        "No me gusta nada, muy malo",
        "Está bien, nada especial",
        "Excelente calidad, lo recomiendo mucho",
        "Terrible experiencia, no lo compren"
    ]
    
    # Cargar datos de ejemplo
    analizador.datos_originales = pd.DataFrame({'texto': textos_ejemplo})
    analizador.datos = analizador.limpiar_datos(analizador.datos_originales.copy())
    
    # Realizar análisis
    exito, mensaje = analizador.analizar_sentimientos()
    
    if exito:
        print(f"Análisis completado: {mensaje}")
        
        # Mostrar estadísticas
        stats, resumen, detalles = analizador.generar_estadisticas()
        print("\nEstadísticas:")
        print(stats)
        
        # Mostrar gráficos
        analizador.mostrar_graficos()
        
    else:
        print(f"Error en análisis: {mensaje}")