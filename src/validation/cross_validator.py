"""
cross_validator.py — Validación cruzada entre datos del DNI y del formulario.

Implementa las 9 reglas de negocio (R01–R09) definidas en el TFM para
comparar los campos extraídos del DNI con los del formulario de préstamo
y detectar inconsistencias.

Reglas:
  R01: nombre DNI == nombre formulario
  R02: apellidos DNI == apellidos formulario
  R03: NIF DNI == NIF formulario (+ dígito de control)
  R04: fecha_nacimiento DNI == fecha_nacimiento formulario
  R05: fecha_caducidad > hoy (documento no caducado)
  R06: edad del titular >= 18 años
  R07: cuota mensual <= 35% de ingresos netos
  R08: importe del préstamo > 0 y dentro de límites
  R09: autenticidad DNI y formulario == LEGÍTIMO

Uso típico:
    from src.validation.cross_validator import CrossValidator
    validator = CrossValidator()
    resultado = validator.validar(campos_dni, campos_formulario, autenticidades)
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from src.config import MAX_DEBT_RATIO, MIN_LOAN_AMOUNT, MAX_LOAN_AMOUNT
from src.data_generation.fake_data_factory import FakeDataFactory
from src.ocr.text_postprocessor import CampoNormalizado

logger = logging.getLogger(__name__)


@dataclass
class ResultadoRegla:
    """Resultado de la evaluación de una regla de validación."""
    codigo:    str    # R01, R02… R09
    nombre:    str    # descripción de la regla
    pasada:    bool
    detalle:   str    # descripción del resultado (valores comparados)
    severidad: str = "ERROR"  # ERROR | WARNING


@dataclass
class ResultadoValidacion:
    """Resultado completo de la validación cruzada."""
    reglas:           list[ResultadoRegla] = field(default_factory=list)
    es_consistente:   bool = True
    n_reglas_ok:      int = 0
    n_reglas_fallo:   int = 0
    reglas_fallidas:  list[str] = field(default_factory=list)

    def agregar_regla(self, regla: ResultadoRegla) -> None:
        self.reglas.append(regla)
        if regla.pasada:
            self.n_reglas_ok += 1
        else:
            self.n_reglas_fallo += 1
            self.reglas_fallidas.append(regla.codigo)
            self.es_consistente = False

    def to_dict(self) -> dict:
        return {
            "es_consistente":  self.es_consistente,
            "n_reglas_ok":     self.n_reglas_ok,
            "n_reglas_fallo":  self.n_reglas_fallo,
            "reglas_fallidas": self.reglas_fallidas,
            "reglas": [
                {
                    "codigo":   r.codigo,
                    "nombre":   r.nombre,
                    "pasada":   r.pasada,
                    "detalle":  r.detalle,
                    "severidad": r.severidad,
                }
                for r in self.reglas
            ],
        }


class CrossValidator:
    """
    Valida la coherencia entre los datos del DNI y del formulario.

    Aplica las 9 reglas de negocio definidas en el TFM y produce
    un ResultadoValidacion con el detalle de cada regla.

    Args:
        tolerancia_nombres: Fracción mínima de similitud para considerar
                           dos nombres iguales (útil para errores OCR leves).
    """

    def __init__(self, tolerancia_nombres: float = 0.85) -> None:
        self._tolerancia = tolerancia_nombres

    def validar(
        self,
        campos_dni:   dict[str, CampoNormalizado],
        campos_form:  dict[str, CampoNormalizado],
        autenticidad_dni:  Optional[dict] = None,
        autenticidad_form: Optional[dict] = None,
    ) -> ResultadoValidacion:
        """
        Ejecuta las 9 reglas de validación cruzada.

        Args:
            campos_dni:  Campos normalizados del DNI {clase: CampoNormalizado}.
            campos_form: Campos normalizados del formulario.
            autenticidad_dni:  Resultado del clasificador para el DNI.
            autenticidad_form: Resultado del clasificador para el formulario.

        Returns:
            ResultadoValidacion con el detalle de cada regla.
        """
        resultado = ResultadoValidacion()

        resultado.agregar_regla(self._r01_nombre(campos_dni, campos_form))
        resultado.agregar_regla(self._r02_apellidos(campos_dni, campos_form))
        resultado.agregar_regla(self._r03_nif(campos_dni, campos_form))
        resultado.agregar_regla(self._r04_fecha_nacimiento(campos_dni, campos_form))
        resultado.agregar_regla(self._r05_caducidad(campos_dni))
        resultado.agregar_regla(self._r06_mayoria_edad(campos_dni))
        resultado.agregar_regla(self._r07_ratio_endeudamiento(campos_form))
        resultado.agregar_regla(self._r08_importe_valido(campos_form))
        resultado.agregar_regla(self._r09_autenticidad(autenticidad_dni, autenticidad_form))

        logger.info(
            "Validación cruzada: %d/9 reglas OK — consistente=%s",
            resultado.n_reglas_ok, resultado.es_consistente,
        )
        return resultado

    # ── Reglas R01–R09 ────────────────────────────────────────────────────────

    def _r01_nombre(self, dni: dict, form: dict) -> ResultadoRegla:
        """R01: El nombre del DNI debe coincidir con el del formulario."""
        v_dni  = self._get_valor(dni,  "nombre")
        v_form = self._get_valor(form, "sol_nombre")
        sim    = self._similitud(v_dni, v_form)
        pasada = sim >= self._tolerancia
        return ResultadoRegla(
            codigo="R01", nombre="Coincidencia nombre",
            pasada=pasada,
            detalle=f"DNI='{v_dni}' / Form='{v_form}' (sim={sim:.2%})",
        )

    def _r02_apellidos(self, dni: dict, form: dict) -> ResultadoRegla:
        """R02: Los apellidos del DNI deben coincidir con los del formulario."""
        v_dni  = self._get_valor(dni,  "apellidos")
        v_form = self._get_valor(form, "sol_apellidos")
        sim    = self._similitud(v_dni, v_form)
        pasada = sim >= self._tolerancia
        return ResultadoRegla(
            codigo="R02", nombre="Coincidencia apellidos",
            pasada=pasada,
            detalle=f"DNI='{v_dni}' / Form='{v_form}' (sim={sim:.2%})",
        )

    def _r03_nif(self, dni: dict, form: dict) -> ResultadoRegla:
        """R03: El NIF del DNI debe coincidir con el del formulario y ser válido."""
        v_dni  = self._get_valor(dni,  "numero_dni")
        v_form = self._get_valor(form, "sol_nif")

        # Coincidencia exacta
        coincide = v_dni.upper() == v_form.upper()
        # Validez del dígito de control
        valido_dni  = FakeDataFactory.validar_nif(v_dni)
        valido_form = FakeDataFactory.validar_nif(v_form)

        pasada = coincide and valido_dni and valido_form
        detalle = (
            f"DNI='{v_dni}' (válido={valido_dni}) / "
            f"Form='{v_form}' (válido={valido_form}) / "
            f"coinciden={coincide}"
        )
        return ResultadoRegla(codigo="R03", nombre="NIF coincidente y válido",
                              pasada=pasada, detalle=detalle)

    def _r04_fecha_nacimiento(self, dni: dict, form: dict) -> ResultadoRegla:
        """R04: La fecha de nacimiento debe coincidir entre DNI y formulario."""
        v_dni  = self._get_valor(dni,  "fecha_nacimiento")
        v_form = self._get_valor(form, "sol_fecha_nacimiento")
        # Normalizar separadores antes de comparar
        norm = lambda s: s.replace("-", "/").replace(" ", "/").replace(".", "/")
        pasada = norm(v_dni) == norm(v_form)
        return ResultadoRegla(
            codigo="R04", nombre="Coincidencia fecha nacimiento",
            pasada=pasada,
            detalle=f"DNI='{v_dni}' / Form='{v_form}'",
        )

    def _r05_caducidad(self, dni: dict) -> ResultadoRegla:
        """R05: El DNI no debe estar caducado."""
        v_cad = self._get_valor(dni, "fecha_caducidad")
        try:
            partes = v_cad.replace("-", "/").replace(" ", "/").split("/")
            if len(partes) == 3:
                d, m, a = int(partes[0]), int(partes[1]), int(partes[2])
                fecha_cad = date(a, m, d)
                hoy = date.today()
                pasada = fecha_cad >= hoy
                dias   = (fecha_cad - hoy).days
                detalle = (
                    f"Caducidad: {v_cad} — "
                    + (f"válido ({dias} días restantes)" if pasada else f"CADUCADO hace {-dias} días")
                )
            else:
                pasada, detalle = False, f"Formato de fecha no reconocido: '{v_cad}'"
        except (ValueError, IndexError):
            pasada, detalle = False, f"No se pudo parsear la fecha de caducidad: '{v_cad}'"

        return ResultadoRegla(codigo="R05", nombre="Documento no caducado",
                              pasada=pasada, detalle=detalle)

    def _r06_mayoria_edad(self, dni: dict) -> ResultadoRegla:
        """R06: El titular debe ser mayor de 18 años."""
        v_nac = self._get_valor(dni, "fecha_nacimiento")
        try:
            partes = v_nac.replace("-", "/").replace(" ", "/").split("/")
            if len(partes) == 3:
                d, m, a = int(partes[0]), int(partes[1]), int(partes[2])
                fecha_nac = date(a, m, d)
                hoy = date.today()
                edad = (hoy - fecha_nac).days // 365
                pasada = edad >= 18
                detalle = f"Fecha nac: {v_nac} → edad calculada: {edad} años"
            else:
                pasada, detalle = False, f"Formato de fecha no reconocido: '{v_nac}'"
        except (ValueError, IndexError):
            pasada, detalle = False, f"No se pudo parsear la fecha: '{v_nac}'"

        return ResultadoRegla(codigo="R06", nombre="Titular mayor de edad",
                              pasada=pasada, detalle=detalle)

    def _r07_ratio_endeudamiento(self, form: dict) -> ResultadoRegla:
        """R07: La cuota mensual no debe superar el 35% de los ingresos."""
        v_cuota    = self._get_valor(form, "prestamo_cuota")
        v_ingresos = self._get_valor(form, "sol_ingresos_netos")
        try:
            cuota    = float(v_cuota.replace(",", "."))
            ingresos = float(v_ingresos.replace(",", "."))
            if ingresos <= 0:
                return ResultadoRegla(
                    codigo="R07", nombre="Ratio endeudamiento ≤35%",
                    pasada=False, detalle="Ingresos netos = 0 o negativos",
                )
            ratio  = cuota / ingresos
            pasada = ratio <= MAX_DEBT_RATIO
            detalle = (
                f"Cuota={cuota:.2f}€ / Ingresos={ingresos:.2f}€ → "
                f"ratio={ratio:.1%} (máx {MAX_DEBT_RATIO:.0%})"
            )
        except (ValueError, ZeroDivisionError):
            pasada  = False
            detalle = f"No se pudo calcular ratio: cuota='{v_cuota}', ingresos='{v_ingresos}'"

        return ResultadoRegla(codigo="R07", nombre="Ratio endeudamiento ≤35%",
                              pasada=pasada, detalle=detalle)

    def _r08_importe_valido(self, form: dict) -> ResultadoRegla:
        """R08: El importe del préstamo debe estar dentro de los límites."""
        v_importe = self._get_valor(form, "prestamo_importe")
        try:
            importe = float(v_importe.replace(",", "."))
            pasada  = MIN_LOAN_AMOUNT <= importe <= MAX_LOAN_AMOUNT
            detalle = (
                f"Importe={importe:,.2f}€ "
                f"(rango válido: {MIN_LOAN_AMOUNT:,.0f}€ – {MAX_LOAN_AMOUNT:,.0f}€)"
            )
        except ValueError:
            pasada  = False
            detalle = f"No se pudo parsear el importe: '{v_importe}'"

        return ResultadoRegla(codigo="R08", nombre="Importe dentro de límites",
                              pasada=pasada, detalle=detalle)

    def _r09_autenticidad(
        self,
        aut_dni:  Optional[dict],
        aut_form: Optional[dict],
    ) -> ResultadoRegla:
        """R09: Ambos documentos deben clasificarse como LEGÍTIMOS."""
        if aut_dni is None and aut_form is None:
            return ResultadoRegla(
                codigo="R09", nombre="Autenticidad documentos",
                pasada=True,
                detalle="Clasificador no disponible — regla omitida",
                severidad="WARNING",
            )

        etq_dni  = (aut_dni  or {}).get("etiqueta", "LEGÍTIMO")
        etq_form = (aut_form or {}).get("etiqueta", "LEGÍTIMO")
        conf_dni  = (aut_dni  or {}).get("confianza", 1.0)
        conf_form = (aut_form or {}).get("confianza", 1.0)

        pasada = etq_dni == "LEGÍTIMO" and etq_form == "LEGÍTIMO"
        detalle = (
            f"DNI={etq_dni}({conf_dni:.2%}) / "
            f"Formulario={etq_form}({conf_form:.2%})"
        )
        return ResultadoRegla(codigo="R09", nombre="Autenticidad documentos",
                              pasada=pasada, detalle=detalle)

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_valor(campos: dict[str, CampoNormalizado], clave: str) -> str:
        """Extrae el texto normalizado de un campo, o cadena vacía si no existe."""
        campo = campos.get(clave)
        if campo is None:
            return ""
        return campo.texto_norm if campo.texto_norm else campo.texto_raw

    @staticmethod
    def _similitud(a: str, b: str) -> float:
        """
        Calcula la similitud entre dos cadenas usando la distancia de Levenshtein
        normalizada. Robusta frente a errores OCR menores (un carácter distinto).
        """
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        a, b = a.upper().strip(), b.upper().strip()
        if a == b:
            return 1.0

        # Distancia de Levenshtein (implementación O(n*m))
        n, m = len(a), len(b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, m + 1):
                costo = 0 if a[i-1] == b[j-1] else 1
                dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev[j-1] + costo)

        distancia = dp[m]
        return 1.0 - distancia / max(n, m)
