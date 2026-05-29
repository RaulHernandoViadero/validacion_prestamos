"""
fake_data_factory.py — Factoría de datos falsos coherentes para el TFM.

Genera todos los datos personales, financieros y documentales necesarios
para crear expedientes sintéticos de verificación de préstamos. Todos los
datos son 100% ficticios y cumplen con las validaciones formales españolas
(NIF con dígito de control, MRZ ICAO 9303, cuota de amortización francesa).

Uso típico:
    from src.data_generation.fake_data_factory import FakeDataFactory
    factory = FakeDataFactory(seed=42)
    expediente = factory.generar_expediente()
"""

import random
import string
import logging
from datetime import date, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

from faker import Faker

from src.config import (
    RANDOM_SEED,
    INCONSISTENCY_RATE,
    MIN_AGE_YEARS,
    MAX_DEBT_RATIO,
    MIN_LOAN_AMOUNT,
    MAX_LOAN_AMOUNT,
    MIN_LOAN_TERM_MONTHS,
    MAX_LOAN_TERM_MONTHS,
)

logger = logging.getLogger(__name__)

# ─── Constantes del NIF español ───────────────────────────────────────────────

_NIF_LETRAS: str = "TRWAGMYFPDXBNJZSQVHLCKE"
"""Tabla de dígitos de control del NIF según el algoritmo oficial español."""

_LETRAS_NO_NIF: str = "AEIOU"
"""Letras que no pueden aparecer como primer carácter de un NIF (reservadas para NIE)."""

# ─── Datos de dominio español ─────────────────────────────────────────────────

_SITUACIONES_LABORALES: list[str] = [
    "Empleado por cuenta ajena - contrato indefinido",
    "Empleado por cuenta ajena - contrato temporal",
    "Autónomo",
    "Funcionario",
    "Pensionista",
]

_FINALIDADES_PRESTAMO: list[str] = [
    "Adquisición de vehículo",
    "Reforma del hogar",
    "Consumo personal",
    "Estudios y formación",
    "Viaje y ocio",
    "Equipamiento del hogar",
]

_PROVINCIAS_ESPANA: list[str] = [
    "Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza",
    "Málaga", "Murcia", "Palma", "Las Palmas", "Bilbao",
    "Alicante", "Córdoba", "Valladolid", "Vigo", "Gijón",
    "Granada", "Elche", "Oviedo", "Badalona", "Cartagena",
]

_TIPOS_VIA: list[str] = [
    "Calle", "Avenida", "Paseo", "Plaza", "Ronda",
    "Carretera", "Travesía", "Camino",
]


# ─── Dataclasses de expediente ────────────────────────────────────────────────

@dataclass
class DatosDNI:
    """Datos completos del DNI sintético."""
    numero_dni: str
    nombre: str
    apellidos: str
    fecha_nacimiento: str        # DD/MM/YYYY
    fecha_caducidad: str         # DD/MM/YYYY
    fecha_emision: str           # DD/MM/YYYY
    nacionalidad: str
    sexo: str                    # M / F
    mrz_linea_1: str
    mrz_linea_2: str


@dataclass
class DatosFormulario:
    """Datos completos del formulario de solicitud de préstamo sintético."""
    sol_nombre: str
    sol_apellidos: str
    sol_nif: str
    sol_fecha_nacimiento: str    # DD/MM/YYYY
    sol_domicilio: str
    sol_telefono: str
    sol_email: str
    sol_situacion_laboral: str
    sol_empresa: str
    sol_ingresos_netos: float    # €/mes
    prestamo_importe: float      # €
    prestamo_plazo: int          # meses
    prestamo_finalidad: str
    prestamo_cuota: float        # €/mes (calculada)
    tae: float                   # Tasa Anual Equivalente (%)


@dataclass
class Expediente:
    """Expediente completo: DNI + formulario + metadatos de validación."""
    expediente_id: str
    dni: DatosDNI
    formulario: DatosFormulario
    es_consistente: bool
    inconsistencias: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serializa el expediente a diccionario para guardar como JSON."""
        return {
            "expediente_id": self.expediente_id,
            "dni": asdict(self.dni),
            "formulario": asdict(self.formulario),
            "es_consistente": self.es_consistente,
            "inconsistencias": self.inconsistencias,
        }


# ─── Clase principal ──────────────────────────────────────────────────────────

class FakeDataFactory:
    """
    Factoría de datos sintéticos coherentes para expedientes bancarios españoles.

    Genera pares (DNI, formulario de préstamo) con datos que respetan las
    validaciones formales españolas. Un porcentaje configurable de expedientes
    introduce inconsistencias deliberadas (ground truth negativo).

    Args:
        seed: Semilla aleatoria para reproducibilidad.
        inconsistency_rate: Fracción [0, 1] de expedientes inconsistentes.
    """

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        inconsistency_rate: float = INCONSISTENCY_RATE,
    ) -> None:
        self._rng = random.Random(seed)
        self._fake = Faker("es_ES")
        self._fake.seed_instance(seed)
        self._inconsistency_rate = inconsistency_rate
        self._contador = 0
        logger.info(
            "FakeDataFactory inicializada — seed=%d, inconsistency_rate=%.0f%%",
            seed, inconsistency_rate * 100,
        )

    # ── NIF ───────────────────────────────────────────────────────────────────

    def generar_nif(self) -> str:
        """
        Genera un NIF español aleatorio con dígito de control válido.

        El algoritmo oficial calcula la letra como:
            letra = _NIF_LETRAS[numero % 23]

        Returns:
            NIF en formato "12345678X".
        """
        numero = self._rng.randint(10_000_000, 99_999_999)
        letra = _NIF_LETRAS[numero % 23]
        return f"{numero}{letra}"

    def _letra_nif(self, numero: int) -> str:
        """Devuelve la letra de control para un número de NIF dado."""
        return _NIF_LETRAS[numero % 23]

    def _corromper_nif(self, nif: str) -> str:
        """
        Devuelve un NIF con dígito de control incorrecto.
        Se usa para generar inconsistencias deliberadas.
        """
        numero = int(nif[:-1])
        letra_correcta = _NIF_LETRAS[numero % 23]
        letras_incorrectas = [c for c in _NIF_LETRAS if c != letra_correcta]
        letra_falsa = self._rng.choice(letras_incorrectas)
        return f"{numero}{letra_falsa}"

    # ── Fechas ────────────────────────────────────────────────────────────────

    def generar_fecha_nacimiento(self) -> date:
        """
        Genera una fecha de nacimiento entre 1950 y 2000, garantizando
        que el titular sea mayor de edad (>= 18 años).
        """
        hoy = date.today()
        inicio = date(1950, 1, 1)
        fin = hoy - timedelta(days=MIN_AGE_YEARS * 365)
        delta = (fin - inicio).days
        return inicio + timedelta(days=self._rng.randint(0, delta))

    def generar_fecha_emision_y_caducidad(
        self, fecha_nacimiento: date
    ) -> tuple[date, date]:
        """
        Genera fechas de emisión y caducidad del DNI coherentes.

        La emisión se sitúa entre los 18 años del titular y la actualidad.
        La caducidad es 10 años posterior a la emisión (norma española).

        Args:
            fecha_nacimiento: Fecha de nacimiento del titular.

        Returns:
            Tupla (fecha_emision, fecha_caducidad).
        """
        hoy = date.today()
        primer_dia_valido = fecha_nacimiento + timedelta(days=MIN_AGE_YEARS * 365)
        delta = max(1, (hoy - primer_dia_valido).days)
        emision = primer_dia_valido + timedelta(days=self._rng.randint(0, delta))
        caducidad = emision.replace(year=emision.year + 10)
        return emision, caducidad

    # ── Persona ───────────────────────────────────────────────────────────────

    def generar_persona(self) -> dict:
        """
        Genera nombre, apellidos, sexo y NIF coherentes para una persona.

        Returns:
            Diccionario con claves: nombre, apellidos, sexo, nif,
            fecha_nacimiento, fecha_emision, fecha_caducidad.
        """
        sexo = self._rng.choice(["M", "F"])
        if sexo == "M":
            nombre = self._fake.first_name_male().upper()
        else:
            nombre = self._fake.first_name_female().upper()

        apellido1 = self._fake.last_name().upper()
        apellido2 = self._fake.last_name().upper()
        apellidos = f"{apellido1} {apellido2}"

        nif = self.generar_nif()
        fecha_nac = self.generar_fecha_nacimiento()
        emision, caducidad = self.generar_fecha_emision_y_caducidad(fecha_nac)

        return {
            "nombre": nombre,
            "apellidos": apellidos,
            "sexo": sexo,
            "nif": nif,
            "fecha_nacimiento": fecha_nac,
            "fecha_emision": emision,
            "fecha_caducidad": caducidad,
        }

    # ── MRZ ICAO 9303 ─────────────────────────────────────────────────────────

    def _mrz_check_digit(self, cadena: str) -> str:
        """
        Calcula el dígito de control MRZ según el estándar ICAO 9303.

        Pesos: 7, 3, 1 (ciclando). Caracteres válidos: 0-9, A-Z, '<'.
        """
        pesos = [7, 3, 1]
        tabla = {str(i): i for i in range(10)}
        tabla.update({chr(ord("A") + i): i + 10 for i in range(26)})
        tabla["<"] = 0

        total = sum(
            tabla.get(c, 0) * pesos[i % 3]
            for i, c in enumerate(cadena)
        )
        return str(total % 10)

    def _formatear_fecha_mrz(self, fecha: date) -> str:
        """Formatea una fecha al formato MRZ: AAMMDD."""
        return fecha.strftime("%y%m%d")

    def _normalizar_mrz(self, texto: str, longitud: int) -> str:
        """
        Normaliza texto para MRZ: mayúsculas, sustituye espacios/ñ por '<',
        elimina caracteres no permitidos y rellena con '<' hasta la longitud.
        """
        replacements = {
            "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
            "Ñ": "N", " ": "<", "-": "<",
        }
        texto = texto.upper()
        for orig, dest in replacements.items():
            texto = texto.replace(orig, dest)
        texto = "".join(c for c in texto if c.isalpha() or c == "<")
        return texto[:longitud].ljust(longitud, "<")

    def generar_mrz(
        self,
        nombre: str,
        apellidos: str,
        nif: str,
        fecha_nacimiento: date,
        fecha_caducidad: date,
        sexo: str,
    ) -> tuple[str, str]:
        """
        Genera las dos líneas MRZ del DNI español (formato TD1, 30 caracteres c/u).

        Línea 1 (30 chars): IDESPXXXXXXXXXXXXXXXXXXXXXXXX
        Línea 2 (30 chars): NNNNNNNN<CFAAAMMDDCSEXCADUCI
        Línea 3 (30 chars): APELLIDOS<<NOMBRE

        Para el DNI español se usan las líneas 1 y 2 combinadas en 2 filas de 30.

        Returns:
            Tupla (linea_1, linea_2) de 30 caracteres cada una.
        """
        # Número de documento (9 chars) + dígito control
        num_doc = nif[:8].ljust(9, "<")
        dc_doc = self._mrz_check_digit(num_doc)

        # Fecha nacimiento AAMMDD + dígito control
        fn = self._formatear_fecha_mrz(fecha_nacimiento)
        dc_fn = self._mrz_check_digit(fn)

        # Fecha caducidad AAMMDD + dígito control
        fc = self._formatear_fecha_mrz(fecha_caducidad)
        dc_fc = self._mrz_check_digit(fc)

        sexo_mrz = "M" if sexo == "M" else "F"

        # Línea 1: ID + ESP + número doc + dc + opcionales (relleno)
        opcional1 = "<" * 15
        dc_opcional = self._mrz_check_digit(opcional1)
        linea_1 = f"IDESP{num_doc}{dc_doc}{opcional1}{dc_opcional}"[:30]

        # Línea 2 (TD1, 30 chars): fn(6)+dc(1)+sexo(1)+fc(6)+dc(1)+nat(3)+opc(11)+dc_compuesto(1)
        opcional2 = "<" * 11
        cuerpo_l2 = f"{fn}{dc_fn}{sexo_mrz}{fc}{dc_fc}ESP{opcional2}"
        dc_compuesto = self._mrz_check_digit(linea_1[5:] + cuerpo_l2[:7] + cuerpo_l2[8:15])
        linea_2 = f"{cuerpo_l2}{dc_compuesto}"[:30]

        # Línea 3: apellidos << nombre (no se incluye aquí pero se genera)
        apellidos_mrz = self._normalizar_mrz(apellidos, 13)
        nombre_mrz = self._normalizar_mrz(nombre, 10)
        linea_3 = f"{apellidos_mrz}<<{nombre_mrz}"[:30].ljust(30, "<")

        logger.debug("MRZ generada: L1=%s | L2=%s | L3=%s", linea_1, linea_2, linea_3)
        return linea_1, linea_2

    # ── Préstamo ──────────────────────────────────────────────────────────────

    def calcular_cuota_francesa(
        self, importe: float, tae: float, plazo_meses: int
    ) -> float:
        """
        Calcula la cuota mensual de un préstamo por el método de amortización
        francés (cuota constante).

        Fórmula:
            C = P * (r * (1+r)^n) / ((1+r)^n - 1)
        donde:
            P = principal, r = tipo mensual, n = número de cuotas.

        Args:
            importe: Capital prestado en euros.
            tae: Tasa Anual Equivalente en porcentaje (ej. 5.5 para 5,5%).
            plazo_meses: Número de cuotas mensuales.

        Returns:
            Cuota mensual en euros, redondeada a 2 decimales.
        """
        r = (tae / 100) / 12  # tipo mensual
        if r == 0:
            return round(importe / plazo_meses, 2)
        factor = (1 + r) ** plazo_meses
        cuota = importe * (r * factor) / (factor - 1)
        return round(cuota, 2)

    def generar_prestamo(
        self, ingresos_netos: float
    ) -> tuple[float, int, str, float, float]:
        """
        Genera parámetros de un préstamo coherente con los ingresos del solicitante.

        La cuota mensual no superará el MAX_DEBT_RATIO de los ingresos netos
        (regla de oro bancaria del 35%).

        Args:
            ingresos_netos: Ingresos netos mensuales del solicitante en euros.

        Returns:
            Tupla (importe, plazo_meses, finalidad, cuota, tae).
        """
        cuota_maxima = ingresos_netos * MAX_DEBT_RATIO
        tae = round(self._rng.uniform(3.5, 12.0), 2)
        finalidad = self._rng.choice(_FINALIDADES_PRESTAMO)

        # Buscar importe y plazo que generen una cuota aceptable
        for _ in range(100):
            importe = round(self._rng.uniform(MIN_LOAN_AMOUNT, MAX_LOAN_AMOUNT), 2)
            plazo = self._rng.randint(MIN_LOAN_TERM_MONTHS, MAX_LOAN_TERM_MONTHS)
            cuota = self.calcular_cuota_francesa(importe, tae, plazo)
            if cuota <= cuota_maxima:
                return importe, plazo, finalidad, cuota, tae

        # Fallback: ajustar importe para cumplir la regla
        plazo = MAX_LOAN_TERM_MONTHS
        importe = min(MAX_LOAN_AMOUNT, cuota_maxima * plazo * 0.8)
        importe = max(MIN_LOAN_AMOUNT, round(importe, 2))
        cuota = self.calcular_cuota_francesa(importe, tae, plazo)
        return importe, plazo, finalidad, cuota, tae

    def generar_ingresos(self, situacion: str) -> float:
        """
        Genera ingresos netos mensuales coherentes con la situación laboral.

        Args:
            situacion: Tipo de contrato/situación laboral.

        Returns:
            Ingresos netos mensuales en euros.
        """
        rangos = {
            "indefinido": (1_400, 4_500),
            "temporal":   (1_000, 2_500),
            "autónomo":   (1_200, 5_000),
            "funcionario": (1_600, 4_000),
            "pensionista": (800, 2_200),
        }
        key = next(
            (k for k in rangos if k in situacion.lower()),
            "indefinido",
        )
        lo, hi = rangos[key]
        return round(self._rng.uniform(lo, hi), 2)

    # ── Domicilio y contacto ──────────────────────────────────────────────────

    def generar_domicilio(self) -> str:
        """Genera una dirección postal española sintética."""
        tipo_via = self._rng.choice(_TIPOS_VIA)
        nombre_via = self._fake.street_name()
        numero = self._rng.randint(1, 150)
        piso = self._rng.choice(["", f", {self._rng.randint(1,9)}º {self._rng.randint(1,6)}ª"])
        cp = str(self._rng.randint(1000, 52999)).zfill(5)
        ciudad = self._rng.choice(_PROVINCIAS_ESPANA)
        return f"{tipo_via} {nombre_via}, {numero}{piso}, {cp} {ciudad}"

    def generar_telefono(self) -> str:
        """Genera un número de teléfono móvil español (6xx/7xx)."""
        prefijo = self._rng.choice(["6", "7"])
        resto = "".join([str(self._rng.randint(0, 9)) for _ in range(8)])
        return f"{prefijo}{resto}"

    def generar_email(self, nombre: str, apellidos: str) -> str:
        """Genera un email sintético basado en el nombre del titular."""
        n = nombre.lower().replace(" ", "").replace("á","a").replace("é","e") \
            .replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
        a = apellidos.split()[0].lower() \
            .replace("á","a").replace("é","e").replace("í","i") \
            .replace("ó","o").replace("ú","u").replace("ñ","n")
        sufijo = self._rng.randint(1, 999)
        dominio = self._rng.choice(["gmail.com", "hotmail.com", "yahoo.es", "outlook.com"])
        return f"{n}.{a}{sufijo}@{dominio}"

    # ── Expediente completo ───────────────────────────────────────────────────

    def generar_expediente(
        self, forzar_inconsistente: Optional[bool] = None
    ) -> Expediente:
        """
        Genera un expediente completo (DNI + formulario) con datos coherentes.

        Con probabilidad `inconsistency_rate`, introduce discrepancias deliberadas
        entre los datos del DNI y los del formulario (ground truth negativo).

        Args:
            forzar_inconsistente: Si se especifica, fuerza el tipo del expediente
                                  independientemente de la tasa configurada.

        Returns:
            Objeto Expediente con todos los datos y metadatos de consistencia.
        """
        self._contador += 1
        exp_id = f"EXP{self._contador:05d}"

        # Decidir si el expediente será inconsistente
        if forzar_inconsistente is not None:
            es_inconsistente = forzar_inconsistente
        else:
            es_inconsistente = self._rng.random() < self._inconsistency_rate

        # ── Generar persona base ──────────────────────────────────────────────
        persona = self.generar_persona()
        nombre = persona["nombre"]
        apellidos = persona["apellidos"]
        sexo = persona["sexo"]
        nif = persona["nif"]
        fecha_nac: date = persona["fecha_nacimiento"]
        fecha_emision: date = persona["fecha_emision"]
        fecha_caducidad: date = persona["fecha_caducidad"]

        # ── Generar MRZ ───────────────────────────────────────────────────────
        mrz_l1, mrz_l2 = self.generar_mrz(
            nombre, apellidos, nif, fecha_nac, fecha_caducidad, sexo
        )

        # ── Construir DNI ─────────────────────────────────────────────────────
        dni = DatosDNI(
            numero_dni=nif,
            nombre=nombre,
            apellidos=apellidos,
            fecha_nacimiento=fecha_nac.strftime("%d/%m/%Y"),
            fecha_caducidad=fecha_caducidad.strftime("%d/%m/%Y"),
            fecha_emision=fecha_emision.strftime("%d/%m/%Y"),
            nacionalidad="ESPAÑOLA",
            sexo=sexo,
            mrz_linea_1=mrz_l1,
            mrz_linea_2=mrz_l2,
        )

        # ── Generar datos del formulario (inicialmente coherentes) ────────────
        situacion = self._rng.choice(_SITUACIONES_LABORALES)
        empresa = self._fake.company().upper()
        ingresos = self.generar_ingresos(situacion)
        importe, plazo, finalidad, cuota, tae = self.generar_prestamo(ingresos)
        domicilio = self.generar_domicilio()
        telefono = self.generar_telefono()
        email = self.generar_email(nombre, apellidos)

        # Datos del formulario inicialmente iguales al DNI
        form_nombre = nombre
        form_apellidos = apellidos
        form_nif = nif
        form_fecha_nac = fecha_nac.strftime("%d/%m/%Y")

        # ── Introducir inconsistencias si corresponde ─────────────────────────
        inconsistencias: list[str] = []

        if es_inconsistente:
            inconsistencias = self._introducir_inconsistencias(
                form_nombre=form_nombre,
                form_apellidos=form_apellidos,
                form_nif=form_nif,
                form_fecha_nac=form_fecha_nac,
            )
            # Aplicar los cambios según las inconsistencias detectadas
            for inc in inconsistencias:
                if "nombre" in inc:
                    form_nombre = self.generar_persona()["nombre"]
                elif "apellidos" in inc:
                    form_apellidos = self.generar_persona()["apellidos"]
                elif "NIF" in inc:
                    form_nif = self._corromper_nif(nif)
                elif "fecha" in inc:
                    otra_persona = self.generar_persona()
                    form_fecha_nac = otra_persona["fecha_nacimiento"].strftime("%d/%m/%Y")

        formulario = DatosFormulario(
            sol_nombre=form_nombre,
            sol_apellidos=form_apellidos,
            sol_nif=form_nif,
            sol_fecha_nacimiento=form_fecha_nac,
            sol_domicilio=domicilio,
            sol_telefono=telefono,
            sol_email=email,
            sol_situacion_laboral=situacion,
            sol_empresa=empresa,
            sol_ingresos_netos=ingresos,
            prestamo_importe=importe,
            prestamo_plazo=plazo,
            prestamo_finalidad=finalidad,
            prestamo_cuota=cuota,
            tae=tae,
        )

        expediente = Expediente(
            expediente_id=exp_id,
            dni=dni,
            formulario=formulario,
            es_consistente=not es_inconsistente,
            inconsistencias=inconsistencias,
        )

        logger.debug(
            "Expediente %s generado — consistente=%s, inconsistencias=%s",
            exp_id, not es_inconsistente, inconsistencias,
        )
        return expediente

    def _introducir_inconsistencias(
        self,
        form_nombre: str,
        form_apellidos: str,
        form_nif: str,
        form_fecha_nac: str,
    ) -> list[str]:
        """
        Selecciona aleatoriamente qué campos del formulario diferirán del DNI.

        Siempre se introduce al menos una inconsistencia; puede haber varias.

        Returns:
            Lista de descripciones de las inconsistencias introducidas.
        """
        campos_posibles = ["nombre", "apellidos", "NIF", "fecha_nacimiento"]
        # Mínimo 1, máximo 3 campos inconsistentes
        n = self._rng.randint(1, min(3, len(campos_posibles)))
        campos_elegidos = self._rng.sample(campos_posibles, n)
        return [f"Discrepancia en campo: {campo}" for campo in campos_elegidos]

    # ── Utilidades ────────────────────────────────────────────────────────────

    def generar_lote(self, n: int) -> list[Expediente]:
        """
        Genera un lote de N expedientes respetando la tasa de inconsistencias.

        Args:
            n: Número de expedientes a generar.

        Returns:
            Lista de Expediente ordenada aleatoriamente.
        """
        n_inconsistentes = round(n * self._inconsistency_rate)
        n_consistentes = n - n_inconsistentes

        expedientes = (
            [self.generar_expediente(forzar_inconsistente=False) for _ in range(n_consistentes)]
            + [self.generar_expediente(forzar_inconsistente=True) for _ in range(n_inconsistentes)]
        )
        self._rng.shuffle(expedientes)
        logger.info(
            "Lote generado: %d expedientes (%d consistentes, %d inconsistentes)",
            n, n_consistentes, n_inconsistentes,
        )
        return expedientes

    @staticmethod
    def validar_nif(nif: str) -> bool:
        """
        Valida que un NIF español tenga el dígito de control correcto.

        Args:
            nif: Cadena en formato "12345678X".

        Returns:
            True si el NIF es formalmente válido.
        """
        if len(nif) != 9:
            return False
        try:
            numero = int(nif[:8])
        except ValueError:
            return False
        letra_esperada = _NIF_LETRAS[numero % 23]
        return nif[8].upper() == letra_esperada
