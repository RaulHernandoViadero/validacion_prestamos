"""
text_postprocessor.py — Corrección semántica y normalización del texto OCR.

El OCR comete errores predecibles en documentos de identidad:
  - Confusión de caracteres similares: 0↔O, 1↔I, 5↔S, 8↔B
  - Espacios extra, caracteres especiales no esperados
  - Fechas en formatos distintos (DD-MM-YYYY, DD/MM/YYYY, DD MM YYYY…)
  - NIF con letra minúscula o caracteres OCR incorrectos

Este módulo normaliza todos los campos extraídos a formatos canónicos
y valida su coherencia semántica.

Uso típico:
    from src.ocr.text_postprocessor import TextPostprocessor
    proc = TextPostprocessor()
    campos = proc.procesar_lote(textos_extraidos)
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.ocr.easyocr_engine import TextoExtraido
from src.data_generation.fake_data_factory import FakeDataFactory

logger = logging.getLogger(__name__)

# ─── Constantes de normalización ─────────────────────────────────────────────

# Correcciones OCR comunes en campos numéricos/alfanuméricos
_OCR_DIGIT_FIXES = str.maketrans("OIlBSZGQ", "01185260")

# Correcciones en campos de texto (nombres)
_OCR_LETTER_FIXES = str.maketrans("01", "OI")

# Meses en español para parseo de fechas
_MESES_ES = {
    "ene": "01", "feb": "02", "mar": "03", "abr": "04",
    "may": "05", "jun": "06", "jul": "07", "ago": "08",
    "sep": "09", "oct": "10", "nov": "11", "dic": "12",
    "enero":"01","febrero":"02","marzo":"03","abril":"04",
    "mayo":"05","junio":"06","julio":"07","agosto":"08",
    "septiembre":"09","octubre":"10","noviembre":"11","diciembre":"12",
}


@dataclass
class CampoNormalizado:
    """Campo de texto normalizado con metadatos de validación."""
    clase:         str
    texto_raw:     str      # texto original del OCR
    texto_norm:    str      # texto normalizado
    confianza_ocr: float    # confianza del motor OCR
    valido:        bool     # pasa la validación semántica del campo
    mensaje_error: str = "" # descripción del error si no es válido

    def to_dict(self) -> dict:
        return {
            "clase":         self.clase,
            "texto_norm":    self.texto_norm,
            "confianza_ocr": round(self.confianza_ocr, 4),
            "valido":        self.valido,
            "mensaje_error": self.mensaje_error,
        }


class TextPostprocessor:
    """
    Normaliza y valida el texto extraído por EasyOCR.

    Aplica reglas específicas por tipo de campo:
      - Nombres y apellidos: limpieza, mayúsculas, sin números
      - NIF/DNI: 8 dígitos + letra de control válida
      - Fechas: normalización a DD/MM/YYYY
      - Importes: eliminación de símbolos, separadores de miles
      - MRZ: solo caracteres permitidos [A-Z0-9<]
    """

    def procesar_lote(
        self,
        textos: list[TextoExtraido],
    ) -> dict[str, CampoNormalizado]:
        """
        Procesa una lista de textos extraídos y devuelve un diccionario
        {clase → CampoNormalizado}.

        Si una clase aparece varias veces (detecciones múltiples),
        se queda con la de mayor confianza OCR.
        """
        # Agrupar por clase, quedarse con la de mayor confianza
        mejores: dict[str, TextoExtraido] = {}
        for t in textos:
            if t.clase not in mejores or t.confianza > mejores[t.clase].confianza:
                mejores[t.clase] = t

        resultado: dict[str, CampoNormalizado] = {}
        for clase, texto_ext in mejores.items():
            resultado[clase] = self.procesar_campo(clase, texto_ext)

        return resultado

    def procesar_campo(
        self,
        clase: str,
        texto_ext: TextoExtraido,
    ) -> CampoNormalizado:
        """
        Normaliza y valida un campo individual según su tipo.

        Args:
            clase: Nombre de la clase YOLO (ej. "numero_dni", "nombre"…).
            texto_ext: Objeto TextoExtraido del motor OCR.

        Returns:
            CampoNormalizado con el texto normalizado y el resultado de validación.
        """
        raw = texto_ext.texto_raw
        conf = texto_ext.confianza
        bbox = texto_ext.bbox_roi

        # Despachar al normalizador específico del campo
        dispatch = {
            # DNI
            "nombre":            self._norm_nombre,
            "apellidos":         self._norm_nombre,
            "numero_dni":        self._norm_nif,
            "fecha_nacimiento":  self._norm_fecha,
            "fecha_caducidad":   self._norm_fecha,
            "nacionalidad":      self._norm_texto_simple,
            "mrz_line":          self._norm_mrz,
            # Formulario
            "sol_nombre":           self._norm_nombre,
            "sol_apellidos":        self._norm_nombre,
            "sol_nif":              self._norm_nif,
            "sol_fecha_nacimiento": self._norm_fecha,
            "sol_domicilio":        self._norm_texto_simple,
            "sol_telefono":         self._norm_telefono,
            "sol_email":            self._norm_email,
            "sol_situacion_laboral": self._norm_texto_simple,
            "sol_empresa":          self._norm_texto_simple,
            "sol_ingresos_netos":   self._norm_importe,
            "prestamo_importe":     self._norm_importe,
            "prestamo_plazo":       self._norm_entero,
            "prestamo_finalidad":   self._norm_texto_simple,
            "prestamo_cuota":       self._norm_importe,
        }

        normalizador = dispatch.get(clase, self._norm_texto_simple)
        try:
            texto_norm, valido, mensaje = normalizador(raw)
        except Exception as exc:
            logger.warning("Error normalizando '%s': %s", clase, exc)
            texto_norm, valido, mensaje = raw.strip(), False, str(exc)

        return CampoNormalizado(
            clase=clase,
            texto_raw=raw,
            texto_norm=texto_norm,
            confianza_ocr=conf,
            valido=valido,
            mensaje_error=mensaje,
        )

    # ── Normalizadores específicos ────────────────────────────────────────────

    def _norm_nombre(self, texto: str) -> tuple[str, bool, str]:
        """Normaliza nombres y apellidos: mayúsculas, sin números ni símbolos."""
        # Aplicar correcciones OCR inversas para nombres (0→O, 1→I)
        texto = texto.translate(_OCR_LETTER_FIXES)
        # Eliminar caracteres no alfabéticos ni espacios ni guiones
        texto = re.sub(r"[^A-ZÁÉÍÓÚÑa-záéíóúñ\s\-]", "", texto)
        texto = " ".join(texto.upper().split())  # normalizar espacios + mayúsculas

        if len(texto) < 2:
            return texto, False, "Nombre demasiado corto"
        if any(c.isdigit() for c in texto):
            return texto, False, "El nombre contiene dígitos"
        return texto, True, ""

    def _norm_nif(self, texto: str) -> tuple[str, bool, str]:
        """Normaliza y valida el NIF español."""
        # Corregir errores OCR en la parte numérica
        texto = texto.strip().upper().replace(" ", "").replace("-", "")

        # Aplicar correcciones de dígitos a los primeros 8 caracteres
        if len(texto) >= 8:
            parte_num = texto[:8].translate(_OCR_DIGIT_FIXES)
            parte_let = texto[8:9] if len(texto) >= 9 else ""
            texto = parte_num + parte_let

        # Limpiar caracteres extra
        texto = re.sub(r"[^0-9A-Z]", "", texto)

        if len(texto) != 9:
            return texto, False, f"NIF con longitud incorrecta: {len(texto)}"

        if not FakeDataFactory.validar_nif(texto):
            return texto, False, f"Dígito de control NIF incorrecto: {texto}"

        return texto, True, ""

    def _norm_fecha(self, texto: str) -> tuple[str, bool, str]:
        """Normaliza fechas al formato DD/MM/YYYY."""
        texto = texto.strip()
        texto = texto.translate(_OCR_DIGIT_FIXES)  # corregir O→0, I→1…

        # Intentar patrones en orden de mayor a menor especificidad
        patrones = [
            # YYYY/MM/DD o YYYY-MM-DD (primero para no confundir con DD/MM/YY)
            (r"(\d{4})[/\-\s.](\d{1,2})[/\-\s.](\d{1,2})", "ymd"),
            # DD/MM/YYYY
            (r"(\d{1,2})[/\-\s.](\d{1,2})[/\-\s.](\d{4})", "dmy4"),
            # DD/MM/YY
            (r"(\d{1,2})[/\-\s.](\d{1,2})[/\-\s.](\d{2})\b", "dmy2"),
            # DD mes YYYY
            (r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", "dmy_str"),
        ]

        for patron, fmt in patrones:
            m = re.search(patron, texto, re.IGNORECASE)
            if not m:
                continue
            g = m.groups()
            try:
                if fmt == "ymd":
                    anyo, mes, dia = g[0], g[1].zfill(2), g[2].zfill(2)
                elif fmt == "dmy4":
                    dia, mes, anyo = g[0].zfill(2), g[1].zfill(2), g[2]
                elif fmt == "dmy2":
                    dia, mes = g[0].zfill(2), g[1].zfill(2)
                    anyo = "19" + g[2] if int(g[2]) > 30 else "20" + g[2]
                else:  # dmy_str
                    dia  = g[0].zfill(2)
                    mes  = _MESES_ES.get(g[1].lower()[:3], g[1].zfill(2))
                    anyo = g[2]

                d, m_n, a = int(dia), int(mes), int(anyo)
                if not (1 <= d <= 31 and 1 <= m_n <= 12 and 1900 <= a <= 2100):
                    continue
                return f"{dia}/{mes}/{anyo}", True, ""
            except (ValueError, KeyError):
                continue

        return texto, False, f"No se pudo parsear la fecha: '{texto}'"

    def _norm_importe(self, texto: str) -> tuple[str, bool, str]:
        """Normaliza importes monetarios a formato float string.

        Soporta formatos:
          Europeo:  1.234,56 €    → 1234.56
          US:       1,234.56 €    → 1234.56  (OCR a veces usa este)
          Sin dec:  27.182 €      → 27182.00 (punto = miles si 3 dígitos tras él)
          Sufijos:  334.63 €/mes  → 334.63   (también €lmes por OCR)
        """
        texto = texto.strip()
        # 1. Eliminar símbolos de moneda, espacios y sufijos temporales
        texto = re.sub(r"[€$£\s]",              "", texto)
        texto = re.sub(r"[Ee][Uu][Rr][Oo][Ss]?","", texto, flags=re.I)
        texto = re.sub(r"EUR",                   "", texto, flags=re.I)
        # "/mes", "lmes" (OCR lee '/' como 'l'), "mes" suelto
        texto = re.sub(r"[/l][Mm][Ee][Ss]",     "", texto, flags=re.I)
        texto = re.sub(r"[Mm][Ee][Ss][Ee][Ss]?","", texto, flags=re.I)
        # Eliminar cualquier letra suelta que quede pegada (artefacto OCR)
        texto = re.sub(r"[A-Za-z]",             "", texto)

        # 2. Normalizar separadores de miles/decimal
        if "," in texto and "." in texto:
            # Determinar qué es miles y qué es decimal según cuál aparece último
            if texto.rfind(".") > texto.rfind(","):
                # Formato US: 1,234.56 → quitar comas, mantener punto
                texto = texto.replace(",", "")
            else:
                # Formato europeo: 1.234,56 → quitar puntos, coma→punto
                texto = texto.replace(".", "").replace(",", ".")
        elif "," in texto:
            partes = texto.split(",")
            if len(partes[-1]) <= 2:
                texto = texto.replace(",", ".")   # decimal europeo: 1,5 → 1.5
            else:
                texto = texto.replace(",", "")    # miles: 1,500 → 1500
        elif "." in texto:
            partes = texto.split(".")
            # Punto como separador de miles: exactamente 3 dígitos tras él
            if len(partes) == 2 and len(partes[1]) == 3 and partes[1].isdigit():
                texto = texto.replace(".", "")    # 27.182 → 27182

        # 3. Correcciones OCR en dígitos
        texto = texto.translate(_OCR_DIGIT_FIXES)
        try:
            valor = float(texto)
            if valor < 0:
                return texto, False, "Importe negativo"
            return f"{valor:.2f}", True, ""
        except ValueError:
            return texto, False, f"No se pudo parsear el importe: '{texto}'"

    def _norm_entero(self, texto: str) -> tuple[str, bool, str]:
        """Normaliza un número entero (ej. plazo en meses)."""
        texto = re.sub(r"[^0-9]", "", texto.translate(_OCR_DIGIT_FIXES))
        if not texto:
            return texto, False, "No se encontró número entero"
        return texto, True, ""

    def _norm_telefono(self, texto: str) -> tuple[str, bool, str]:
        """Normaliza número de teléfono español."""
        texto = re.sub(r"[^0-9\+]", "", texto.translate(_OCR_DIGIT_FIXES))
        if len(texto) < 9:
            return texto, False, f"Teléfono demasiado corto: {len(texto)} dígitos"
        return texto[-9:], True, ""  # quedarse con los últimos 9 dígitos

    def _norm_email(self, texto: str) -> tuple[str, bool, str]:
        """Normaliza y valida un email."""
        texto = texto.strip().lower().replace(" ", "")
        if "@" not in texto or "." not in texto.split("@")[-1]:
            return texto, False, f"Email inválido: '{texto}'"
        return texto, True, ""

    def _norm_mrz(self, texto: str) -> tuple[str, bool, str]:
        """Normaliza la zona MRZ: solo caracteres [A-Z0-9<]."""
        texto = texto.upper().replace(" ", "<").replace("-", "<")
        texto = re.sub(r"[^A-Z0-9<]", "<", texto)
        return texto, len(texto) >= 20, "" if len(texto) >= 20 else "MRZ demasiado corta"

    def _norm_texto_simple(self, texto: str) -> tuple[str, bool, str]:
        """Normalización básica: strip y mayúsculas."""
        texto = " ".join(texto.strip().upper().split())
        return texto, len(texto) > 0, "" if texto else "Campo vacío"
