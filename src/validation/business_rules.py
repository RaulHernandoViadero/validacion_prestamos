"""
business_rules.py — Reglas de negocio auxiliares para validación de préstamos.

Complementa al CrossValidator con funciones de dominio financiero y
documental que pueden reutilizarse en otras partes del sistema:

  - Verificación del formato MRZ ICAO 9303 TD1
  - Cálculo y verificación de cuota de amortización francesa
  - Clasificación de finalidad del préstamo
  - Evaluación del perfil de riesgo crediticio
  - Comprobación de coherencia de fechas en el expediente

Todas las funciones son puras (sin efectos secundarios) y están
tipadas para facilitar los tests unitarios.

Uso típico:
    from src.validation.business_rules import (
        verificar_mrz_checkdigit,
        verificar_cuota_francesa,
        evaluar_perfil_riesgo,
    )
"""

import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from src.config import (
    MAX_DEBT_RATIO,
    MIN_LOAN_AMOUNT,
    MAX_LOAN_AMOUNT,
)

logger = logging.getLogger(__name__)

# ── Constantes de negocio ─────────────────────────────────────────────────────

# Tabla ICAO 9303 para cálculo de dígito de control MRZ
_MRZ_WEIGHTS = [7, 3, 1]
_MRZ_CHARS   = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Tolerancia relativa en la verificación de cuota (errores de redondeo)
_CUOTA_TOLERANCIA_REL = 0.02   # ±2% para cubrir redondeo a 2 decimales

# Finalidades de préstamo aceptadas
_FINALIDADES_VALIDAS = {
    "VIVIENDA", "VEHÍCULO", "VEHICULO", "REFORMAS", "EDUCACIÓN", "EDUCACION",
    "SALUD", "NEGOCIOS", "DEUDAS", "CONSUMO", "OTROS",
}

# Clasificación de riesgo crediticio (ratio de endeudamiento)
_RIESGO_BAJO    = 0.20   # ≤ 20%
_RIESGO_MEDIO   = 0.30   # ≤ 30%
# > 35% = alto (supera el umbral legal MAX_DEBT_RATIO)


# ── Dataclasses de resultados ─────────────────────────────────────────────────

@dataclass
class ResultadoMRZ:
    """Resultado de la verificación del dígito de control MRZ."""
    linea:       str    # "1" o "2"
    valido:      bool
    detalle:     str


@dataclass
class ResultadoCuota:
    """Resultado de la verificación de la cuota de amortización francesa."""
    cuota_declarada:  float
    cuota_calculada:  float
    diferencia_abs:   float
    diferencia_rel:   float
    valido:           bool
    detalle:          str


@dataclass
class PerfilRiesgo:
    """Perfil de riesgo crediticio del solicitante."""
    nivel:          str    # BAJO | MEDIO | ALTO
    ratio_deuda:    float  # cuota / ingresos
    edad:           int
    ingresos_netos: float
    cuota_mensual:  float
    importe:        float
    plazo_meses:    int
    detalle:        str


# ── MRZ — Dígito de control ICAO 9303 ────────────────────────────────────────

def calcular_digito_control_mrz(cadena: str) -> int:
    """
    Calcula el dígito de control ICAO 9303 de una cadena MRZ.

    Algoritmo:
      1. Cada carácter se convierte a valor numérico:
         '0'–'9' → 0–9, 'A'–'Z' → 10–35, '<' → 0
      2. Se multiplica por el peso correspondiente (ciclo 7, 3, 1)
      3. Se suma y se hace módulo 10

    Args:
        cadena: Subcadena MRZ (sin el propio dígito de control).

    Returns:
        Dígito de control (0–9).
    """
    total = 0
    for i, ch in enumerate(cadena.upper()):
        if ch.isdigit():
            val = int(ch)
        elif ch.isalpha():
            val = ord(ch) - ord("A") + 10
        else:
            val = 0  # '<' y cualquier otro carácter de relleno
        total += val * _MRZ_WEIGHTS[i % 3]
    return total % 10


def verificar_mrz_checkdigit(
    linea_1: str,
    linea_2: str,
) -> list[ResultadoMRZ]:
    """
    Verifica los dígitos de control de las dos líneas MRZ TD1.

    En el formato TD1 (DNI español) se comprueban:
      - Línea 1, pos 28: dígito de control del número de documento (pos 5-14)
      - Línea 2, pos 6:  dígito de control de la fecha de nacimiento (pos 1-6)
      - Línea 2, pos 15: dígito de control de la fecha de caducidad (pos 9-14)
      - Línea 2, pos 30: dígito de control compuesto (toda la info variable)

    Args:
        linea_1: Primera línea MRZ (30 caracteres).
        linea_2: Segunda línea MRZ (30 caracteres).

    Returns:
        Lista de ResultadoMRZ, uno por cada dígito verificado.
    """
    resultados: list[ResultadoMRZ] = []

    l1 = linea_1.upper().ljust(30, "<")[:30]
    l2 = linea_2.upper().ljust(30, "<")[:30]

    # ── Línea 1: dígito del número de documento ───────────────────────────────
    if len(l1) >= 29:
        num_doc    = l1[5:14]          # posiciones 6-14 (0-indexed: 5-13)
        dig_esp    = int(l1[14]) if l1[14].isdigit() else -1
        dig_calc   = calcular_digito_control_mrz(num_doc)
        ok_doc     = dig_esp == dig_calc
        resultados.append(ResultadoMRZ(
            linea="1",
            valido=ok_doc,
            detalle=(
                f"Num.doc '{num_doc}': dígito esperado={dig_calc}, "
                f"encontrado={dig_esp} → {'OK' if ok_doc else 'FALLO'}"
            ),
        ))

    # ── Línea 2: dígito de fecha de nacimiento ────────────────────────────────
    if len(l2) >= 7:
        fecha_nac  = l2[0:6]
        dig_esp    = int(l2[6]) if l2[6].isdigit() else -1
        dig_calc   = calcular_digito_control_mrz(fecha_nac)
        ok_nac     = dig_esp == dig_calc
        resultados.append(ResultadoMRZ(
            linea="2",
            valido=ok_nac,
            detalle=(
                f"Fecha nac '{fecha_nac}': dígito esperado={dig_calc}, "
                f"encontrado={dig_esp} → {'OK' if ok_nac else 'FALLO'}"
            ),
        ))

    # ── Línea 2: dígito de fecha de caducidad ────────────────────────────────
    if len(l2) >= 15:
        fecha_cad  = l2[8:14]
        dig_esp    = int(l2[14]) if l2[14].isdigit() else -1
        dig_calc   = calcular_digito_control_mrz(fecha_cad)
        ok_cad     = dig_esp == dig_calc
        resultados.append(ResultadoMRZ(
            linea="2",
            valido=ok_cad,
            detalle=(
                f"Fecha cad '{fecha_cad}': dígito esperado={dig_calc}, "
                f"encontrado={dig_esp} → {'OK' if ok_cad else 'FALLO'}"
            ),
        ))

    # ── Dígito de control compuesto (posición 30 de línea 2) ─────────────────
    if len(l1) >= 30 and len(l2) >= 30:
        # Compuesto: num_doc(9) + dig_doc(1) + opcional1(15) + fecha_nac(6) +
        #            dig_nac(1) + sexo(1) + fecha_cad(6) + dig_cad(1) +
        #            opcional2(11)
        compuesto = (
            l1[5:15]      # num_doc + dig_doc
            + l1[15:30]   # opcional1
            + l2[0:7]     # fecha_nac + dig_nac
            + l2[7:15]    # sexo + fecha_cad + dig_cad
            + l2[15:29]   # opcional2
        )
        dig_esp  = int(l2[29]) if l2[29].isdigit() else -1
        dig_calc = calcular_digito_control_mrz(compuesto)
        ok_comp  = dig_esp == dig_calc
        resultados.append(ResultadoMRZ(
            linea="2",
            valido=ok_comp,
            detalle=(
                f"Compuesto: dígito esperado={dig_calc}, "
                f"encontrado={dig_esp} → {'OK' if ok_comp else 'FALLO'}"
            ),
        ))

    return resultados


# ── Cuota de amortización francesa ────────────────────────────────────────────

def calcular_cuota_francesa(
    importe: float,
    tae: float,
    plazo_meses: int,
) -> float:
    """
    Calcula la cuota mensual constante por el método de amortización francesa.

    Fórmula:
        C = P * (r * (1+r)^n) / ((1+r)^n - 1)

    donde:
        P = importe del préstamo (€)
        r = tipo de interés mensual = TAE / 12
        n = número de cuotas (meses)

    Args:
        importe:     Capital prestado en euros.
        tae:         Tasa Anual Equivalente (porcentaje, ej. 5.5 para 5,5%).
        plazo_meses: Número de mensualidades.

    Returns:
        Cuota mensual en euros, redondeada a 2 decimales.
        Devuelve 0.0 si los parámetros son inválidos.
    """
    if importe <= 0 or plazo_meses <= 0:
        return 0.0

    r = tae / 100 / 12   # tipo mensual
    if r == 0:
        # Sin intereses: cuota plana
        return round(importe / plazo_meses, 2)

    factor = (1 + r) ** plazo_meses
    cuota  = importe * (r * factor) / (factor - 1)
    return round(cuota, 2)


def verificar_cuota_francesa(
    cuota_declarada: float,
    importe: float,
    tae: float,
    plazo_meses: int,
) -> ResultadoCuota:
    """
    Verifica que la cuota declarada coincide con la fórmula francesa.

    La verificación tolera un ±2% de diferencia relativa para absorber
    el redondeo a 2 decimales y las variaciones de TAE mensualizada.

    Args:
        cuota_declarada: Cuota mensual extraída del formulario.
        importe:         Capital prestado en euros.
        tae:             TAE del préstamo (porcentaje).
        plazo_meses:     Número de mensualidades.

    Returns:
        ResultadoCuota con la comparación y el veredicto.
    """
    cuota_calc = calcular_cuota_francesa(importe, tae, plazo_meses)

    if cuota_calc == 0.0:
        return ResultadoCuota(
            cuota_declarada=cuota_declarada,
            cuota_calculada=0.0,
            diferencia_abs=0.0,
            diferencia_rel=0.0,
            valido=False,
            detalle="Parámetros inválidos para calcular cuota",
        )

    dif_abs = abs(cuota_declarada - cuota_calc)
    dif_rel = dif_abs / cuota_calc if cuota_calc > 0 else 1.0
    valido  = dif_rel <= _CUOTA_TOLERANCIA_REL

    detalle = (
        f"Declarada={cuota_declarada:.2f}€ / "
        f"Calculada={cuota_calc:.2f}€ / "
        f"Diferencia={dif_abs:.2f}€ ({dif_rel:.1%}) → "
        + ("OK" if valido else f"FALLO (máx ±{_CUOTA_TOLERANCIA_REL:.0%})")
    )
    logger.debug("Verificación cuota francesa: %s", detalle)

    return ResultadoCuota(
        cuota_declarada=cuota_declarada,
        cuota_calculada=cuota_calc,
        diferencia_abs=round(dif_abs, 2),
        diferencia_rel=round(dif_rel, 4),
        valido=valido,
        detalle=detalle,
    )


# ── Perfil de riesgo crediticio ───────────────────────────────────────────────

def evaluar_perfil_riesgo(
    ingresos_netos: float,
    cuota_mensual:  float,
    importe:        float,
    plazo_meses:    int,
    fecha_nacimiento: Optional[str] = None,
) -> PerfilRiesgo:
    """
    Evalúa el perfil de riesgo crediticio del solicitante.

    Clasifica el riesgo en BAJO / MEDIO / ALTO según el ratio de
    endeudamiento (cuota / ingresos) y otros factores:
      - BAJO:  ratio ≤ 20%
      - MEDIO: ratio > 20% y ≤ 30%
      - ALTO:  ratio > 30% (incluye los casos que superan el 35% legal)

    Args:
        ingresos_netos: Ingresos mensuales netos en euros.
        cuota_mensual:  Cuota mensual del préstamo en euros.
        importe:        Importe total del préstamo en euros.
        plazo_meses:    Plazo en meses.
        fecha_nacimiento: Fecha en formato DD/MM/YYYY (opcional, para calcular edad).

    Returns:
        PerfilRiesgo con el nivel de riesgo y los indicadores clave.
    """
    # Calcular edad
    edad = 0
    if fecha_nacimiento:
        try:
            partes = fecha_nacimiento.replace("-", "/").split("/")
            if len(partes) == 3:
                d, m, a = int(partes[0]), int(partes[1]), int(partes[2])
                fecha_nac = date(a, m, d)
                edad = (date.today() - fecha_nac).days // 365
        except (ValueError, IndexError):
            pass

    # Ratio de endeudamiento
    ratio = cuota_mensual / ingresos_netos if ingresos_netos > 0 else 1.0

    # Nivel de riesgo
    if ratio <= _RIESGO_BAJO:
        nivel = "BAJO"
    elif ratio <= _RIESGO_MEDIO:
        nivel = "MEDIO"
    else:
        nivel = "ALTO"

    # Detalle del perfil
    detalle_partes = [
        f"ratio_deuda={ratio:.1%}",
        f"ingresos={ingresos_netos:.0f}€/mes",
        f"cuota={cuota_mensual:.0f}€/mes",
        f"importe={importe:,.0f}€ a {plazo_meses} meses",
    ]
    if edad > 0:
        detalle_partes.append(f"edad={edad} años")
    if ratio > MAX_DEBT_RATIO:
        detalle_partes.append(f"⚠ supera umbral legal {MAX_DEBT_RATIO:.0%}")

    detalle = f"Riesgo {nivel}: " + ", ".join(detalle_partes)
    logger.debug(detalle)

    return PerfilRiesgo(
        nivel=nivel,
        ratio_deuda=round(ratio, 4),
        edad=edad,
        ingresos_netos=ingresos_netos,
        cuota_mensual=cuota_mensual,
        importe=importe,
        plazo_meses=plazo_meses,
        detalle=detalle,
    )


# ── Finalidad del préstamo ────────────────────────────────────────────────────

def validar_finalidad(finalidad: str) -> tuple[bool, str]:
    """
    Verifica que la finalidad declarada del préstamo sea una categoría válida.

    Args:
        finalidad: Texto de la finalidad extraído del formulario.

    Returns:
        Tupla (valido, mensaje).
    """
    fin_norm = finalidad.upper().strip()
    # Buscar coincidencia parcial por si el OCR captura texto extra
    for f in _FINALIDADES_VALIDAS:
        if f in fin_norm or fin_norm in f:
            return True, f"Finalidad reconocida: '{f}'"
    return False, f"Finalidad no reconocida: '{finalidad}'"


# ── Coherencia de fechas ──────────────────────────────────────────────────────

def verificar_coherencia_fechas(
    fecha_nacimiento:  str,
    fecha_emision:     Optional[str] = None,
    fecha_caducidad:   Optional[str] = None,
) -> list[tuple[str, bool, str]]:
    """
    Comprueba la coherencia lógica entre las fechas del DNI.

    Verificaciones:
      1. La fecha de emisión debe ser posterior a la de nacimiento.
      2. La fecha de caducidad debe ser posterior a la de emisión.
      3. El periodo de validez (emision→caducidad) debe ser 5 o 10 años
         (plazos estándar del DNI español).
      4. La fecha de caducidad debe ser futura.

    Args:
        fecha_nacimiento: DD/MM/YYYY — fecha de nacimiento del titular.
        fecha_emision:    DD/MM/YYYY — fecha de expedición del DNI (opcional).
        fecha_caducidad:  DD/MM/YYYY — fecha de caducidad del DNI (opcional).

    Returns:
        Lista de tuplas (descripcion, valido, detalle) para cada verificación.
    """
    def _parsear(texto: str) -> Optional[date]:
        try:
            partes = texto.replace("-", "/").split("/")
            if len(partes) == 3:
                d, m, a = int(partes[0]), int(partes[1]), int(partes[2])
                return date(a, m, d)
        except (ValueError, IndexError):
            pass
        return None

    resultados: list[tuple[str, bool, str]] = []
    hoy = date.today()

    f_nac = _parsear(fecha_nacimiento)
    f_emi = _parsear(fecha_emision)  if fecha_emision  else None
    f_cad = _parsear(fecha_caducidad) if fecha_caducidad else None

    # 1. Emisión posterior al nacimiento
    if f_nac and f_emi:
        ok  = f_emi > f_nac
        resultados.append((
            "Emisión > Nacimiento",
            ok,
            f"Nac={fecha_nacimiento}, Emi={fecha_emision} → {'OK' if ok else 'FALLO (emisión anterior al nacimiento)'}",
        ))

    # 2. Caducidad posterior a la emisión
    if f_emi and f_cad:
        ok  = f_cad > f_emi
        resultados.append((
            "Caducidad > Emisión",
            ok,
            f"Emi={fecha_emision}, Cad={fecha_caducidad} → {'OK' if ok else 'FALLO (caducidad anterior a emisión)'}",
        ))

    # 3. Periodo de validez: 5 o 10 años (±30 días de tolerancia)
    if f_emi and f_cad:
        dias_validez = (f_cad - f_emi).days
        anios_validez_5  = abs(dias_validez - 365 * 5) <= 30
        anios_validez_10 = abs(dias_validez - 365 * 10) <= 30
        ok  = anios_validez_5 or anios_validez_10
        resultados.append((
            "Plazo validez DNI (5 o 10 años)",
            ok,
            f"Validez calculada: {dias_validez // 365} años ({dias_validez} días) → {'OK' if ok else 'FALLO (no es 5 ni 10 años)'}",
        ))

    # 4. Caducidad futura
    if f_cad:
        ok  = f_cad >= hoy
        dias_restantes = (f_cad - hoy).days
        resultados.append((
            "Documento vigente",
            ok,
            f"Cad={fecha_caducidad}: {'vigente (' + str(dias_restantes) + ' días)' if ok else 'CADUCADO hace ' + str(-dias_restantes) + ' días'}",
        ))

    return resultados


# ── Validación de límites del préstamo ────────────────────────────────────────

def validar_limites_prestamo(
    importe: float,
    plazo_meses: int,
    tae: float,
) -> tuple[bool, list[str]]:
    """
    Verifica que los parámetros del préstamo estén dentro de los límites
    de negocio configurados en config.py.

    Args:
        importe:     Capital solicitado en euros.
        plazo_meses: Duración en meses (6–360).
        tae:         Tasa Anual Equivalente (0.1–30%).

    Returns:
        Tupla (todos_validos, lista_de_errores).
    """
    errores: list[str] = []

    if not (MIN_LOAN_AMOUNT <= importe <= MAX_LOAN_AMOUNT):
        errores.append(
            f"Importe {importe:,.2f}€ fuera de rango "
            f"[{MIN_LOAN_AMOUNT:,.0f}€ – {MAX_LOAN_AMOUNT:,.0f}€]"
        )

    if not (6 <= plazo_meses <= 360):
        errores.append(
            f"Plazo {plazo_meses} meses fuera de rango [6 – 360]"
        )

    if not (0.1 <= tae <= 30.0):
        errores.append(
            f"TAE {tae:.2f}% fuera de rango [0.1% – 30%]"
        )

    return len(errores) == 0, errores
