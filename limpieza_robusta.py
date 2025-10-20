# limpieza_robusta.py
"""
Módulo de limpieza robusta de datos con corrección ortográfica automática
Proporciona funcionalidades avanzadas para limpiar y corregir textos
"""

import re

# Verificar disponibilidad del corrector ortográfico
SPELLCHECKER_DISPONIBLE = False
try:
    from spellchecker import SpellChecker
    SPELLCHECKER_DISPONIBLE = True
except ImportError:
    pass


class LimpiadorDatosRobusto:
    """Clase especializada para limpieza exhaustiva de datos de texto con corrección ortográfica"""        
    def __init__(self):
        # Diccionario de reemplazos para caracteres especiales y leetspeak
        self.replacements = {
            # Leetspeak básico
            '@': 'a', '4': 'a', '∆': 'a',
            '3': 'e', '€': 'e', 
            '1': 'i', '!': 'i', '|': 'i',
            '0': 'o', '°': 'o',
            '5': 's', '$': 's', '§': 's',
            '7': 't', '+': 't',
            '8': 'b', 'ß': 'b',
            '6': 'g', '9': 'g',
            '2': 'z',
            
            # Caracteres especiales comunes
            'ñ': 'n', 'Ñ': 'N',
            'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ā': 'a', 'ă': 'a', 'ą': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e', 'ē': 'e', 'ĕ': 'e', 'ę': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i', 'ī': 'i', 'ĭ': 'i', 'į': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'ō': 'o', 'ŏ': 'o', 'ő': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'ū': 'u', 'ŭ': 'u', 'ů': 'u',
            
            # Mayúsculas con acentos
            'Á': 'A', 'À': 'A', 'Ä': 'A', 'Â': 'A', 'Ā': 'A', 'Ă': 'A', 'Ą': 'A',
            'É': 'E', 'È': 'E', 'Ë': 'E', 'Ê': 'E', 'Ē': 'E', 'Ĕ': 'E', 'Ę': 'E',
            'Í': 'I', 'Ì': 'I', 'Ï': 'I', 'Î': 'I', 'Ī': 'I', 'Ĭ': 'I', 'Į': 'I',
            'Ó': 'O', 'Ò': 'O', 'Ö': 'O', 'Ô': 'O', 'Ō': 'O', 'Ŏ': 'O', 'Ő': 'O',
            'Ú': 'U', 'Ù': 'U', 'Ü': 'U', 'Û': 'U', 'Ū': 'U', 'Ŭ': 'U', 'Ů': 'U',
            
            # Caracteres raros y símbolos
            '¿': '', '¡': '',
            '«': '"', '»': '"',
            '"': '"', '"': '"',
            ''': "'", ''': "'",
            '…': '...',
            '–': '-', '—': '-',
            '•': '-', '·': '-',
            '°': 'o', '™': '', '®': '', '©': '',
            '€': 'euros', '£': 'libras', '$': 'dolares',
            '&': ' y ', '%': ' por ciento',
        }
        
        # Patrones para diferentes tipos de "basura" en texto
        self.regex_patterns = {
            # URLs y enlaces
            'urls': r'https?://[^\s]+|www\.[^\s]+|[^\s]+\.com[^\s]*|[^\s]+\.org[^\s]*|[^\s]+\.net[^\s]*',
            # Emails
            'emails': r'\S+@\S+\.\S+',
            # Hashtags y menciones
            'social': r'#\w+|@\w+',
            # Números de teléfono
            'phones': r'(\+?1?[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            # Códigos y IDs alfanuméricos
            'codes': r'\b[A-Z0-9]{3,}-[A-Z0-9]{3,}\b|\b[A-Z]{2,}[0-9]{3,}\b',
            # Múltiples espacios, tabs, saltos de línea
            'whitespace': r'\s+',
            # Múltiples signos de puntuación
            'punct_multiple': r'[.]{3,}|[!]{2,}|[?]{2,}|[,]{2,}|[;]{2,}|[:-]{2,}',
            # Caracteres de control y no imprimibles
            'control_chars': r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]',
            # HTML tags básicos
            'html_tags': r'<[^>]+>|&nbsp;|&amp;|&lt;|&gt;|&quot;|&#\d+;',
            # Repeticiones excesivas de caracteres
            'char_repeat': r'([a-zA-Z])\1{3,}',
            # Números largos sin contexto
            'long_numbers': r'\b\d{10,}\b',
        }
        
        # Palabras comunes mal escritas y sus correcciones
        self.common_misspellings = {
            # Errores comunes en español
            'q': 'que', 'x': 'por', 'xq': 'porque', 'pq': 'porque', 'xk': 'porque',
            'tb': 'también', 'tmb': 'también', 'tbn': 'también',
            'd': 'de', 'pa': 'para', 'pr': 'por',
            'sta': 'esta', 'sto': 'esto', 'stoy': 'estoy',
            'bn': 'bien', 'mj': 'mejor', 'mjr': 'mejor',
            'salu2': 'saludos', 'bss': 'besos', 'bso': 'beso',
            'grax': 'gracias', 'grcs': 'gracias', 'grc': 'gracias',
            'wey': 'güey', 'we': 'güey', 'bro': 'hermano',
            'k': 'que', 'ke': 'que', 'kien': 'quien',
            'komo': 'como', 'kuando': 'cuando', 'kual': 'cual',
            'aver': 'a ver', 'ablar': 'hablar', 'acer': 'hacer',
            'dnd': 'donde', 'tmpo': 'tiempo', 'nmbr': 'nombre',
            'msj': 'mensaje', 'msg': 'mensaje', 'txt': 'texto',
            'fb': 'facebook', 'ig': 'instagram', 'tw': 'twitter',
            'lol': 'risa', 'omg': 'dios mío', 'wtf': 'qué diablos',
            
            # Correcciones específicas
            'hpla': 'hola',
            'felis': 'feliz',
            'sentio': 'sentido',
            'senti': 'siento',
            'snt': 'siento',
            'stoy': 'estoy',
            'stas': 'estás',
            'sta': 'está',
            'q': 'que',
            'xq': 'porque',
            'x': 'por',
            'k': 'que',
            'kiero': 'quiero',
            'keria': 'quería',
            'ay': 'hay',
            'ai': 'hay',
            'all': 'ahí',
            'ahi': 'ahí',
            'ps': 'pues',
            'pos': 'pues',
            'pq': 'porque',
            'Quiero': 'Quiero',
            'queiro': 'quiero',
            'sento': 'siento',
            'quijero': 'quiero',
                   
            # Errores de escritura con números
            'm8': 'mate', 'l8r': 'later', 'u2': 'you too',
            '2morrow': 'mañana', '4ever': 'para siempre',
            'b4': 'antes', '2day': 'hoy', '2night': 'esta noche',
            
            # Errores ortográficos comunes
            'seto': 'siento', 'felis': 'feliz', 'estoi': 'estoy',
            'aser': 'hacer', 'ablar': 'hablar', 'tener': 'tener',
            'saver': 'saber', 'bamos': 'vamos', 'benir': 'venir',
            'aora': 'ahora', 'lla': 'ya', 'tambien': 'también',
            'despues': 'después', 'facil': 'fácil', 'dificil': 'difícil'
        }
        
        # Emojis y emoticonos a texto
        self.emoji_to_text = {
            '😀': ' feliz ', '😃': ' feliz ', '😄': ' feliz ', '😁': ' feliz ',
            '😅': ' risa nerviosa ', '😂': ' risa ', '🤣': ' risa ',
            '😊': ' sonrisa ', '😇': ' angelical ', '🙂': ' sonrisa leve ',
            '😉': ' guiño ', '😌': ' aliviado ', '😍': ' enamorado ',
            '😘': ' beso ', '😗': ' beso ', '😙': ' beso ', '😚': ' beso ',
            '😋': ' delicioso ', '😛': ' lengua fuera ', '😜': ' guiño lengua ',
            '🤪': ' loco ', '😝': ' lengua fuera ', '🤗': ' abrazo ',
            '😏': ' picaro ', '😒': ' aburrido ', '🙄': ' ojos en blanco ',
            '😬': ' nervioso ', '🤐': ' callado ', '😷': ' enfermo ',
            '🤒': ' enfermo ', '🤕': ' herido ', '🤢': ' nauseas ',
            '🤮': ' vomito ', '🤧': ' estornudo ', '😵': ' mareado ',
            '😴': ' dormido ', '😪': ' somnoliento ', '😔': ' triste ',
            '😟': ' preocupado ', '😕': ' confundido ', '🙁': ' triste ',
            '😖': ' confundido ', '😣': ' perseverante ', '😞': ' decepcionado ',
            '😓': ' sudor frio ', '😩': ' cansado ', '😫': ' cansado ',
            '😤': ' enojado ', '😠': ' enojado ', '😡': ' furioso ',
            '🤬': ' palabrotas ', '😈': ' diablillo ', '👿': ' demonio ',
            '💀': ' muerte ', '☠️': ' calavera ', '👻': ' fantasma ',
            '👽': ' alien ', '👾': ' monstruo ', '🤖': ' robot ',
            '💩': ' caca ', '🤡': ' payaso ', '👹': ' ogro ',
            '👺': ' duende ', '🔥': ' fuego ', '💯': ' cien por cien ',
            '💢': ' enojado ', '💥': ' explosion ', '💫': ' mareado ',
            '💦': ' gotas ', '💨': ' viento ', '🕳️': ' hoyo ',
            '💣': ' bomba ', '💤': ' dormido ','👏👏': ' aplausos ', '👋': ' saludo ',
            '❤️': ' amor ', '💔': ' corazon roto ','🇪🇨': 'bandera de Ecuador ','<3': ' amor ',

            # Emoticonos texto
            ':)': ' feliz ', ':-)': ' feliz ', '(:': ' feliz ',
            ':D': ' muy feliz ', ':-D': ' muy feliz ', 'XD': ' risa ',
            ':P': ' lengua fuera ', ':-P': ' lengua fuera ', ':p': ' lengua fuera ',
            ';)': ' guiño ', ';-)': ' guiño ', ';P': ' guiño lengua ',
            ':(': ' triste ', ':-(': ' triste ', ')=': ' triste ',
            ":'(": ' llorando ', ':,(': ' llorando ', 'T_T': ' llorando ',
            ':S': ' confundido ', ':-S': ' confundido ', ':s': ' confundido ',
            ':O': ' sorprendido ', ':-O': ' sorprendido ', ':o': ' sorprendido ',
            ':|': ' serio ', ':-|': ' serio ', '-_-': ' serio ',
            ':@': ' enojado ', '>:(': ' enojado ', '>:-(': ' enojado ',
            '<3': ' amor ', '</3': ' corazon roto ', '<\\3': ' corazon roto ',
        }
        
        # Variables para estadísticas
        self.estadisticas = {}
        
        # Inicializar corrector ortográfico si está disponible
        self.corrector_disponible = False
        if SPELLCHECKER_DISPONIBLE:
            try:
                self.spell = SpellChecker(language='es')  # Español
                self.corrector_disponible = True
                print("✅ Corrector ortográfico español inicializado")
            except Exception as e:
                print(f"⚠️ Error al inicializar corrector: {e}")
                self.corrector_disponible = False

    def limpiar_texto_completo(self, texto):
        """
        Aplica todas las limpiezas a un texto individual
        """
        if not isinstance(texto, str):
            texto = str(texto)
        
        texto_limpio = texto
        correcciones_realizadas = 0
        
        # 1. CORRECCIÓN DE PALABRAS COMUNES MAL ESCRITAS (DICCIONARIO)
        palabras = texto_limpio.split()
        palabras_corregidas = []
        
        for palabra in palabras:
            palabra_limpia = re.sub(r'[^\w]', '', palabra.lower())  # Quitar puntuación para comparar
            
            # Buscar en el diccionario de errores comunes
            if palabra_limpia in self.common_misspellings:
                correccion = self.common_misspellings[palabra_limpia]
                # Preservar mayúsculas originales
                if palabra[0].isupper():
                    correccion = correccion.capitalize()
                # Reemplazar manteniendo puntuación
                palabra_corregida = re.sub(r'\w+', correccion, palabra, count=1)
                palabras_corregidas.append(palabra_corregida)
                correcciones_realizadas += 1
            else:
                palabras_corregidas.append(palabra)
        
        texto_limpio = ' '.join(palabras_corregidas)
        
        # 2. CORRECCIÓN ORTOGRÁFICA CON SPELLCHECKER (si está disponible)
        if self.corrector_disponible:
            palabras_para_revisar = re.findall(r'\b\w+\b', texto_limpio.lower())
            for palabra_original in palabras_para_revisar:
                if len(palabra_original) > 2 and palabra_original not in self.spell:
                    candidatos = self.spell.candidates(palabra_original)
                    if candidatos:
                        correccion = list(candidatos)[0]
                        if palabra_original != correccion:
                            # Reemplazar en el texto manteniendo mayúsculas
                            patron = r'\b' + re.escape(palabra_original) + r'\b'
                            def reemplazar_palabra(match):
                                palabra_match = match.group()
                                if palabra_match.isupper():
                                    return correccion.upper()
                                elif palabra_match.istitle():
                                    return correccion.capitalize()
                                else:
                                    return correccion
                            
                            nuevo_texto = re.sub(patron, reemplazar_palabra, texto_limpio, flags=re.IGNORECASE)
                            if nuevo_texto != texto_limpio:
                                correcciones_realizadas += 1
                                texto_limpio = nuevo_texto
        
        # 3. Reemplazar emojis y emoticonos
        for emoji, texto_emoji in self.emoji_to_text.items():
            if emoji in texto_limpio:
                texto_limpio = texto_limpio.replace(emoji, texto_emoji)
        
        # 4. Aplicar reemplazos de caracteres especiales
        for char, replacement in self.replacements.items():
            texto_limpio = texto_limpio.replace(char, replacement)
        
        # 5. Limpiar URLs, emails, etc.
        for pattern_name, pattern in self.regex_patterns.items():
            if pattern_name == 'urls':
                texto_limpio = re.sub(pattern, '[URL_REMOVIDA]', texto_limpio)
            elif pattern_name == 'emails':
                texto_limpio = re.sub(pattern, '[EMAIL_REMOVIDO]', texto_limpio)
            elif pattern_name == 'social':
                texto_limpio = re.sub(pattern, '[MENCION_REMOVIDA]', texto_limpio)
            elif pattern_name == 'phones':
                texto_limpio = re.sub(pattern, '[TELEFONO_REMOVIDO]', texto_limpio)
            elif pattern_name == 'whitespace':
                texto_limpio = re.sub(pattern, ' ', texto_limpio)
            elif pattern_name == 'punct_multiple':
                texto_limpio = re.sub(pattern, '.', texto_limpio)
            elif pattern_name == 'control_chars':
                texto_limpio = re.sub(pattern, '', texto_limpio)
            elif pattern_name == 'html_tags':
                texto_limpio = re.sub(pattern, '', texto_limpio)
            elif pattern_name == 'char_repeat':
                texto_limpio = re.sub(pattern, r'\1\1', texto_limpio)
            elif pattern_name == 'long_numbers':
                texto_limpio = re.sub(pattern, '[NUMERO_REMOVIDO]', texto_limpio)
        
        # Limpiar espacios extra
        texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
        
        return texto_limpio, correcciones_realizadas

    def limpiar_datos_post_analisis(self, datos):
        """
        Realiza una limpieza robusta de los datos después del análisis CON CORRECCIÓN ORTOGRÁFICA
        """
        import pandas as pd
        
        datos_limpios = datos.copy()
        self.estadisticas = {
            'textos_procesados': len(datos),
            'urls_removidas': 0,
            'emails_removidos': 0,
            'menciones_removidas': 0,
            'hashtags_removidos': 0,
            'telefonos_removidos': 0,
            'caracteres_especiales_removidos': 0,
            'puntuacion_normalizada': 0,
            'mayusculas_normalizadas': 0,
            'espacios_normalizados': 0,
            'palabras_corregidas': 0,
            'textos_con_correcciones': 0,
            'textos_modificados': 0,
            'emojis_convertidos': 0
        }
        
        textos_originales = datos['texto'].copy()
        
        print("🧹 Iniciando limpieza robusta con corrección ortográfica...")
        print(f"📊 Procesando {len(datos)} textos...")
        
        ejemplos_correcciones = []  # Para mostrar ejemplos
        
        for idx, texto in enumerate(datos_limpios['texto']):
            if (idx + 1) % 50 == 0:  # Progreso cada 50 textos
                print(f"📊 Procesando texto {idx+1}/{len(datos_limpios)}...")
            
            texto_original = str(texto)
            texto_limpio, correcciones = self.limpiar_texto_completo(texto_original)
            
            # Contar estadísticas
            if correcciones > 0:
                self.estadisticas['palabras_corregidas'] += correcciones
                self.estadisticas['textos_con_correcciones'] += 1
                
                # Guardar ejemplo para el reporte
                if len(ejemplos_correcciones) < 5 and texto_original != texto_limpio:
                    ejemplos_correcciones.append({
                        'original': texto_original[:100] + ('...' if len(texto_original) > 100 else ''),
                        'corregido': texto_limpio[:100] + ('...' if len(texto_limpio) > 100 else '')
                    })
            
            # Contar otros elementos procesados
            self.estadisticas['urls_removidas'] += len(re.findall(self.regex_patterns['urls'], texto_original))
            self.estadisticas['emails_removidos'] += len(re.findall(self.regex_patterns['emails'], texto_original))
            
            # Contar emojis convertidos
            for emoji in self.emoji_to_text.keys():
                if emoji in texto_original:
                    self.estadisticas['emojis_convertidos'] += texto_original.count(emoji)
            
            # Actualizar el texto si fue modificado
            if texto_limpio != texto_original:
                datos_limpios.loc[idx, 'texto'] = texto_limpio
                self.estadisticas['textos_modificados'] += 1
        
        # Conservar textos originales para comparación
        if 'texto_pre_limpieza' not in datos_limpios.columns:
            datos_limpios['texto_pre_limpieza'] = textos_originales
        
        # Guardar ejemplos en estadísticas
        self.estadisticas['ejemplos_correcciones'] = ejemplos_correcciones
        
        print("✅ Limpieza robusta completada!")
        print(f"✏️ Se corrigieron {self.estadisticas['palabras_corregidas']} palabras en {self.estadisticas['textos_con_correcciones']} textos")
        print(f"😊 Se convirtieron {self.estadisticas['emojis_convertidos']} emojis a texto")
        
        return True, "Limpieza robusta con corrección ortográfica completada exitosamente", datos_limpios, self.estadisticas
    
    def generar_reporte_limpieza(self):
        """Genera un reporte detallado de la limpieza realizada CON EJEMPLOS DE CORRECCIÓN"""
        if not self.estadisticas:
            return "No se ha ejecutado ninguna limpieza aún."
        
        total_elementos_procesados = (
            self.estadisticas['palabras_corregidas'] +
            self.estadisticas['urls_removidas'] +
            self.estadisticas['emails_removidos'] +
            self.estadisticas['emojis_convertidos']
        )
        
        reporte = "🧹 REPORTE DE LIMPIEZA ROBUSTA CON CORRECCIÓN ORTOGRÁFICA\n"
        reporte += "=" * 80 + "\n"
        reporte += f"📊 ESTADÍSTICAS GENERALES:\n"
        reporte += f"   • Textos procesados: {self.estadisticas['textos_procesados']:,}\n"
        reporte += f"   • Textos modificados: {self.estadisticas['textos_modificados']:,}\n"
        reporte += f"   • Porcentaje modificado: {(self.estadisticas['textos_modificados']/self.estadisticas['textos_procesados']*100):.1f}%\n"
        reporte += f"   • Total de elementos procesados: {total_elementos_procesados:,}\n\n"
        
        # SECCIÓN DE CORRECCIÓN ORTOGRÁFICA
        reporte += f"✏️ CORRECCIÓN ORTOGRÁFICA:\n"
        reporte += f"   📝 Palabras corregidas: {self.estadisticas['palabras_corregidas']:,}\n"
        reporte += f"   📄 Textos con correcciones: {self.estadisticas['textos_con_correcciones']:,}\n"
        if self.estadisticas['textos_con_correcciones'] > 0:
            promedio = self.estadisticas['palabras_corregidas'] / self.estadisticas['textos_con_correcciones']
            reporte += f"   📊 Promedio correcciones por texto: {promedio:.1f}\n"
        
        # EJEMPLOS DE CORRECCIONES
        if 'ejemplos_correcciones' in self.estadisticas and self.estadisticas['ejemplos_correcciones']:
            reporte += f"\n💡 EJEMPLOS DE CORRECCIONES REALIZADAS:\n"
            for i, ejemplo in enumerate(self.estadisticas['ejemplos_correcciones'][:3], 1):
                reporte += f"   {i}. Original: {ejemplo['original']}\n"
                reporte += f"      Corregido: {ejemplo['corregido']}\n"
        
        reporte += f"\n🔧 OTROS ELEMENTOS PROCESADOS:\n"
        reporte += f"   🔍 URLs removidas: {self.estadisticas['urls_removidas']:,}\n"
        reporte += f"   📧 Emails removidos: {self.estadisticas['emails_removidos']:,}\n"
        reporte += f"   😊 Emojis convertidos: {self.estadisticas['emojis_convertidos']:,}\n"
        
        reporte += f"\n✅ BENEFICIOS DE LA LIMPIEZA:\n"
        reporte += f"   • Ortografía corregida automáticamente\n"
        reporte += f"   • Texto más legible y profesional\n"
        reporte += f"   • Emojis convertidos a texto descriptivo\n"
        reporte += f"   • Información personal protegida\n"
        reporte += f"   • Datos optimizados para análisis\n"
        reporte += f"   • Mejor calidad para exportación\n"
        
        return reporte