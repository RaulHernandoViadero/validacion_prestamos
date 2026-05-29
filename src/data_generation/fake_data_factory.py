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
    "Liquidación de deudas",
    "Negocio o actividad profesional",
]

# Municipios españoles con su código postal base (primeros 2 dígitos) y provincia
_MUNICIPIOS: list[tuple[str, str, str]] = [
    ("Madrid",         "Madrid",         "28"),
    ("Barcelona",      "Barcelona",      "08"),
    ("Valencia",       "Valencia",       "46"),
    ("Sevilla",        "Sevilla",        "41"),
    ("Zaragoza",       "Zaragoza",       "50"),
    ("Málaga",         "Málaga",         "29"),
    ("Murcia",         "Murcia",         "30"),
    ("Palma",          "Baleares",       "07"),
    ("Las Palmas de Gran Canaria", "Las Palmas", "35"),
    ("Bilbao",         "Vizcaya",        "48"),
    ("Alicante",       "Alicante",       "03"),
    ("Córdoba",        "Córdoba",        "14"),
    ("Valladolid",     "Valladolid",     "47"),
    ("Vigo",           "Pontevedra",     "36"),
    ("Gijón",          "Asturias",       "33"),
    ("Granada",        "Granada",        "18"),
    ("Elche",          "Alicante",       "03"),
    ("Oviedo",         "Asturias",       "33"),
    ("Badalona",       "Barcelona",      "08"),
    ("Cartagena",      "Murcia",         "30"),
    ("Terrassa",       "Barcelona",      "08"),
    ("Jerez de la Frontera", "Cádiz",   "11"),
    ("Sabadell",       "Barcelona",      "08"),
    ("Santa Cruz de Tenerife", "Santa Cruz de Tenerife", "38"),
    ("Almería",        "Almería",        "04"),
    ("Alcalá de Henares", "Madrid",     "28"),
    ("Pamplona",       "Navarra",        "31"),
    ("Fuenlabrada",    "Madrid",         "28"),
    ("Leganés",        "Madrid",         "28"),
    ("San Sebastián",  "Guipúzcoa",      "20"),
    ("Burgos",         "Burgos",         "09"),
    ("Santander",      "Cantabria",      "39"),
    ("Albacete",       "Albacete",       "02"),
    ("Castellón de la Plana", "Castellón", "12"),
    ("Getafe",         "Madrid",         "28"),
    ("Salamanca",      "Salamanca",      "37"),
    ("Huelva",         "Huelva",         "21"),
    ("Logroño",        "La Rioja",       "26"),
    ("Badajoz",        "Badajoz",        "06"),
    ("León",           "León",           "24"),
]

_TIPOS_VIA: list[str] = [
    "Calle", "Avenida", "Paseo", "Plaza", "Ronda",
    "Carretera", "Travesía", "Camino", "Bulevar", "Vía",
]

_NOMBRES_VIA: list[str] = [
    "de la Constitución", "Mayor", "del Carmen", "de España",
    "de Cervantes", "Real", "de la Paz", "del Sol",
    "de Goya", "de la República", "de las Flores", "del Prado",
    "de Colón", "de la Libertad", "del Pilar", "de Alfonso X",
    "de los Reyes Católicos", "de la Independencia", "de San Juan",
    "del Generalísimo", "de la Cruz", "de Ponferrada", "del Norte",
    "de Castilla", "de Aragón", "de Cataluña", "de Andalucía",
]

# Bancos y empresas españolas realistas para el campo "empresa"
_SECTORES_EMPRESA: dict[str, list[str]] = {
    "indefinido": [
        "Telefónica S.A.", "Iberdrola S.A.", "Inditex S.A.",
        "Banco Santander S.A.", "BBVA S.A.", "CaixaBank S.A.",
        "Endesa S.A.", "Repsol S.A.", "Mapfre S.A.",
        "El Corte Inglés S.A.", "Mercadona S.A.", "Lidl España S.A.",
        "Acciona S.A.", "ACS Grupo S.A.", "Ferrovial S.A.",
        "Seat S.A.", "Airbus España S.L.", "Amazon España S.L.",
        "Google España S.L.", "Microsoft Ibérica S.R.L.",
    ],
    "temporal": [
        "Adecco España S.A.", "Randstad España S.L.",
        "Manpower Group S.L.", "Eulen S.A.",
        "Securitas España S.A.", "ISS Facility Services S.A.",
    ],
    "autónomo": [
        "Autónomo", "Trabajador por cuenta propia",
        "Freelance", "Profesional liberal",
    ],
    "funcionario": [
        "Ministerio de Hacienda", "Ministerio de Educación",
        "Comunidad de Madrid", "Generalitat de Catalunya",
        "Junta de Andalucía", "Ayuntamiento de Madrid",
        "Ayuntamiento de Barcelona", "Diputación Provincial",
        "Universidad Complutense de Madrid", "Hospital La Paz",
        "Guardia Civil", "Policía Nacional", "Ejército de Tierra",
    ],
    "pensionista": [
        "Seguridad Social - Pensión de jubilación",
        "Seguridad Social - Pensión de incapacidad",
        "MUFACE - Pensión de jubilación",
    ],
}


# ─── Dataclasses de expediente ────────────────────────────────────────────────

@dataclass
class DatosDNI:
    """Datos completos del DNI sintético."""
    numero_dni: str
    nombre: str
    apellidos: str
    fecha_nacimiento: str          # DD/MM/YYYY
    fecha_caducidad: str           # DD/MM/YYYY
    fecha_emision: str             # DD/MM/YYYY
    nacionalidad: str
    sexo: str                      # M / F
    lugar_nacimiento: str          # Ciudad, Provincia
    numero_soporte: str            # Número de soporte (CAN) - 9 dígitos
    mrz_linea_1: str               # 30 caracteres
    mrz_linea_2: str               # 30 caracteres


@dataclass
class DatosFormulario:
    """Datos completos del formulario de solicitud de préstamo sintético."""
    sol_nombre: str
    sol_apellidos: str
    sol_nif: str
    sol_fecha_nacimiento: str      # DD/MM/YYYY
    sol_domicilio: str
    sol_municipio: str
    sol_provincia: str
    sol_codigo_postal: str
    sol_telefono: str
    sol_email: str
    sol_situacion_laboral: str
    sol_empresa: str
    sol_ingresos_netos: float      # €/mes
    prestamo_importe: float        # €
    prestamo_plazo: int            # meses
    prestamo_finalidad: str
    prestamo_cuota: float          # €/mes (calculada)
    tae: float                     # Tasa Anual Equivalente (%)
    ratio_endeudamiento: float     # cuota / ingresos (porcentaje)


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

    def generar_numero_soporte(self) -> str:
        """
        Genera un número de soporte (CAN) del DNI: 3 letras + 6 dígitos.
        Ejemplo: AAA123456
        """
        letras = "".join(self._rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3))
        digitos = "".join(str(self._rng.randint(0, 9)) for _ in range(6))
        return f"{letras}{digitos}"

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
        self, fecha_nacimiento: date, forzar_vigente: bool = True
    ) -> tuple[date, date]:
        """
        Genera fechas de emisión y caducidad del DNI coherentes.

        La caducidad es 10 años posterior a la emisión (norma española).
        Por defecto garantiza que el documento esté vigente (emisión reciente).

        Args:
            fecha_nacimiento: Fecha de nacimiento del titular.
            forzar_vigente: Si True, la emisión será dentro de los últimos
                            9 años para garantizar que la caducidad sea futura.

        Returns:
            Tupla (fecha_emision, fecha_caducidad).
        """
        hoy = date.today()
        primer_dia_valido = fecha_nacimiento + timedelta(days=MIN_AGE_YEARS * 365)

        if forzar_vigente:
            # Emisión entre hace 9 años y hace 1 mes → caducidad siempre en el futuro
            inicio_emision = max(primer_dia_valido, hoy - timedelta(days=9 * 365))
            fin_emision = hoy - timedelta(days=30)
            if inicio_emision >= fin_emision:
                inicio_emision = fin_emision - timedelta(days=365)
        else:
            # Puede ser un documento expirado (para test de regla R05)
            inicio_emision = primer_dia_valido
            fin_emision = hoy

        delta = max(1, (fin_emision - inicio_emision).days)
        emision = inicio_emision + timedelta(days=self._rng.randint(0, delta))
        caducidad = emision.replace(year=emision.year + 10)
        return emision, caducidad

    # ── Persona ───────────────────────────────────────────────────────────────

    def generar_persona(self, forzar_vigente: bool = True) -> dict:
        """
        Genera nombre, apellidos, sexo, NIF y fechas coherentes para una persona.

        Args:
            forzar_vigente: Si True, garantiza que el DNI no esté caducado.

        Returns:
            Diccionario con todas las claves necesarias para DatosDNI.
        """
        sexo = self._rng.choice(["M", "F"])
        nombre = (
            self._fake.first_name_male().upper()
            if sexo == "M"
            else self._fake.first_name_female().upper()
        )
        apellido1 = self._fake.last_name().upper()
        apellido2 = self._fake.last_name().upper()
        apellidos = f"{apellido1} {apellido2}"

        nif = self.generar_nif()
        fecha_nac = self.generar_fecha_nacimiento()
        emision, caducidad = self.generar_fecha_emision_y_caducidad(
            fecha_nac, forzar_vigente=forzar_vigente
        )

        # Lugar de nacimiento: ciudad distinta a la de residencia habitual
        ciudad_nac, provincia_nac, _ = self._rng.choice(_MUNICIPIOS)
        lugar_nacimiento = f"{ciudad_nac} ({provincia_nac})"

        return {
            "nombre": nombre,
            "apellidos": apellidos,
            "sexo": sexo,
            "nif": nif,
            "fecha_nacimiento": fecha_nac,
            "fecha_emision": emision,
            "fecha_caducidad": caducidad,
            "lugar_nacimiento": lugar_nacimiento,
            "numero_soporte": self.generar_numero_soporte(),
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
        Normaliza texto para MRZ: mayúsculas, sustituye acentos/espacios/guiones
        por '<', elimina caracteres no permitidos y rellena hasta la longitud.
        """
        replacements = {
            "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
            "Ñ": "N", " ": "<", "-": "<", "'": "<", ".": "<",
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
        Genera las dos líneas MRZ del DNI español en formato TD1 (30 chars c/u).

        Línea 1: IDESP + nº doc (9) + dc + opcionales (15) + dc_comp
        Línea 2: f_nac (6) + dc + sexo + f_cad (6) + dc + nat (3) + opc (11) + dc_comp
        Línea 3: apellidos << nombre (30 chars — se devuelve pero no se imprime en img)

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

        # Línea 1: IDESP + num_doc(9) + dc(1) + opcionales(15) + dc_comp(1) = 30
        opcional1 = "<" * 15
        dc_comp1 = self._mrz_check_digit(opcional1)
        linea_1 = f"IDESP{num_doc}{dc_doc}{opcional1}{dc_comp1}"

        # Línea 2: fn(6)+dc(1)+sexo(1)+fc(6)+dc(1)+nat(3)+opc(11)+dc_comp(1) = 30
        opcional2 = "<" * 11
        cuerpo_l2 = f"{fn}{dc_fn}{sexo_mrz}{fc}{dc_fc}ESP{opcional2}"
        dc_comp2 = self._mrz_check_digit(
            linea_1[5:] + cuerpo_l2[:7] + cuerpo_l2[8:15]
        )
        linea_2 = f"{cuerpo_l2}{dc_comp2}"

        # Asegurar longitudes exactas
        linea_1 = linea_1[:30].ljust(30, "<")
        linea_2 = linea_2[:30].ljust(30, "<")

        logger.debug("MRZ generada: L1=%s | L2=%s", linea_1, linea_2)
        return linea_1, linea_2

    # ── Préstamo ──────────────────────────────────────────────────────────────

    def calcular_cuota_francesa(
        self, importe: float, tae: float, plazo_meses: int
    ) -> float:
        """
        Calcula la cuota mensual por amortización francesa (cuota constante).

            C = P * (r * (1+r)^n) / ((1+r)^n - 1)

        Args:
            importe: Capital prestado en euros.
            tae: Tasa Anual Equivalente en porcentaje (ej. 5.5 para 5,5%).
            plazo_meses: Número de cuotas mensuales.

        Returns:
            Cuota mensual en euros, redondeada a 2 decimales.
        """
        r = (tae / 100) / 12
        if r == 0:
            return round(importe / plazo_meses, 2)
        factor = (1 + r) ** plazo_meses
        cuota = importe * (r * factor) / (factor - 1)
        return round(cuota, 2)

    def generar_ingresos(self, situacion: str, edad_anos: int = 35) -> float:
        """
        Genera ingresos netos mensuales coherentes con la situación laboral
        y la edad del solicitante (a mayor edad, mayor experiencia → más ingresos).

        Args:
            situacion: Tipo de contrato/situación laboral.
            edad_anos: Edad del solicitante en años.

        Returns:
            Ingresos netos mensuales en euros.
        """
        rangos_base: dict[str, tuple[float, float]] = {
            "indefinido": (1_400.0, 4_500.0),
            "temporal":   (1_100.0, 2_200.0),
            "autónomo":   (1_200.0, 5_500.0),
            "funcionario":(1_700.0, 4_200.0),
            "pensionista":(800.0,   2_400.0),
        }
        key = next(
            (k for k in rangos_base if k in situacion.lower()),
            "indefinido",
        )
        lo, hi = rangos_base[key]

        # Factor de experiencia: entre 25 y 55 años aumentan los ingresos un 30%
        factor = 1.0 + min(0.30, max(0.0, (edad_anos - 25) / 100))
        lo = round(lo * factor, 2)
        hi = round(hi * factor, 2)

        return round(self._rng.uniform(lo, hi), 2)

    def generar_prestamo(
        self, ingresos_netos: float
    ) -> tuple[float, int, str, float, float]:
        """
        Genera parámetros de un préstamo coherente con los ingresos.

        La cuota mensual no superará el MAX_DEBT_RATIO de ingresos netos
        (regla de oro bancaria del 35%).

        Returns:
            Tupla (importe, plazo_meses, finalidad, cuota, tae).
        """
        cuota_maxima = ingresos_netos * MAX_DEBT_RATIO
        tae = round(self._rng.uniform(3.5, 12.0), 2)
        finalidad = self._rng.choice(_FINALIDADES_PRESTAMO)

        for _ in range(200):
            importe = round(self._rng.uniform(MIN_LOAN_AMOUNT, MAX_LOAN_AMOUNT), 2)
            plazo = self._rng.randint(MIN_LOAN_TERM_MONTHS, MAX_LOAN_TERM_MONTHS)
            cuota = self.calcular_cuota_francesa(importe, tae, plazo)
            if cuota <= cuota_maxima:
                return importe, plazo, finalidad, cuota, tae

        # Fallback conservador
        plazo = MAX_LOAN_TERM_MONTHS
        importe = max(MIN_LOAN_AMOUNT, round(cuota_maxima * plazo * 0.75, 2))
        importe = min(importe, MAX_LOAN_AMOUNT)
        cuota = self.calcular_cuota_francesa(importe, tae, plazo)
        return importe, plazo, finalidad, cuota, tae

    # ── Domicilio y contacto ──────────────────────────────────────────────────

    def generar_domicilio_completo(self) -> dict:
        """
        Genera una dirección postal española completa y coherente.

        Returns:
            Diccionario con claves: direccion, municipio, provincia, cp.
        """
        municipio, provincia, cp_base = self._rng.choice(_MUNICIPIOS)
        tipo_via = self._rng.choice(_TIPOS_VIA)
        nombre_via = self._rng.choice(_NOMBRES_VIA)
        numero = self._rng.randint(1, 200)

        # Piso y puerta opcionales
        tiene_piso = self._rng.random() < 0.65
        if tiene_piso:
            planta = self._rng.randint(1, 9)
            puerta = self._rng.choice(["A", "B", "C", "D", "Dcha.", "Izda."])
            complemento = f", {planta}º {puerta}"
        else:
            complemento = ""

        # CP: base de 2 dígitos + 3 dígitos aleatorios (ej. "28045")
        cp_sufijo = str(self._rng.randint(0, 999)).zfill(3)
        cp = f"{cp_base}{cp_sufijo}"

        direccion = f"{tipo_via} {nombre_via}, {numero}{complemento}"
        return {
            "direccion": direccion,
            "municipio": municipio,
            "provincia": provincia,
            "cp": cp,
        }

    def generar_telefono(self) -> str:
        """Genera un número de teléfono móvil español (6xx/7xx)."""
        prefijo = self._rng.choice(["6", "7"])
        # Formato: 6XX XXX XXX — generado con dígitos realistas
        bloque1 = str(self._rng.randint(10, 99))
        bloque2 = str(self._rng.randint(100, 999))
        bloque3 = str(self._rng.randint(100, 999))
        return f"{prefijo}{bloque1}{bloque2}{bloque3}"

    def generar_email(self, nombre: str, apellidos: str) -> str:
        """Genera un email sintético basado en el nombre del titular."""
        def limpiar(s: str) -> str:
            tabla = str.maketrans(
                "ÁÉÍÓÚáéíóúÑñÜü", "AEIOUaeiouNnUu"
            )
            return s.translate(tabla).replace(" ", "").lower()

        n = limpiar(nombre)[:8]
        a = limpiar(apellidos.split()[0])[:10]
        variante = self._rng.choice([
            f"{n}.{a}",
            f"{n}{a}",
            f"{a}.{n}",
            f"{n}.{a}{self._rng.randint(1, 99)}",
            f"{n[0]}{a}",
        ])
        dominio = self._rng.choice([
            "gmail.com", "hotmail.com", "yahoo.es",
            "outlook.com", "telefonica.net", "orange.es",
        ])
        return f"{variante}@{dominio}"

    def _empresa_por_situacion(self, situacion: str) -> str:
        """Selecciona una empresa coherente con la situación laboral."""
        key = next(
            (k for k in _SECTORES_EMPRESA if k in situacion.lower()),
            "indefinido",
        )
        return self._rng.choice(_SECTORES_EMPRESA[key])

    # ── Expediente completo ───────────────────────────────────────────────────

    def generar_expediente(
        self, forzar_inconsistente: Optional[bool] = None
    ) -> "Expediente":
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

        es_inconsistente = (
            forzar_inconsistente
            if forzar_inconsistente is not None
            else self._rng.random() < self._inconsistency_rate
        )

        # ── Persona base ──────────────────────────────────────────────────────
        # Documentos consistentes siempre vigentes; inconsistentes pueden estar caducados
        forzar_vigente = not es_inconsistente or self._rng.random() > 0.2
        persona = self.generar_persona(forzar_vigente=forzar_vigente)

        nombre: str = persona["nombre"]
        apellidos: str = persona["apellidos"]
        sexo: str = persona["sexo"]
        nif: str = persona["nif"]
        fecha_nac: date = persona["fecha_nacimiento"]
        fecha_emision: date = persona["fecha_emision"]
        fecha_caducidad: date = persona["fecha_caducidad"]
        lugar_nacimiento: str = persona["lugar_nacimiento"]
        numero_soporte: str = persona["numero_soporte"]

        # Edad para correlacionar con ingresos
        hoy = date.today()
        edad_anos = (hoy - fecha_nac).days // 365

        # ── MRZ ───────────────────────────────────────────────────────────────
        mrz_l1, mrz_l2 = self.generar_mrz(
            nombre, apellidos, nif, fecha_nac, fecha_caducidad, sexo
        )

        # ── DNI ───────────────────────────────────────────────────────────────
        dni = DatosDNI(
            numero_dni=nif,
            nombre=nombre,
            apellidos=apellidos,
            fecha_nacimiento=fecha_nac.strftime("%d %m %Y"),
            fecha_caducidad=fecha_caducidad.strftime("%d %m %Y"),
            fecha_emision=fecha_emision.strftime("%d %m %Y"),
            nacionalidad="ESPAÑOLA",
            sexo=sexo,
            lugar_nacimiento=lugar_nacimiento,
            numero_soporte=numero_soporte,
            mrz_linea_1=mrz_l1,
            mrz_linea_2=mrz_l2,
        )

        # ── Formulario (inicialmente coherente con el DNI) ────────────────────
        situacion = self._rng.choice(_SITUACIONES_LABORALES)
        empresa = self._empresa_por_situacion(situacion)
        ingresos = self.generar_ingresos(situacion, edad_anos)
        importe, plazo, finalidad, cuota, tae = self.generar_prestamo(ingresos)
        domicilio = self.generar_domicilio_completo()
        telefono = self.generar_telefono()
        email = self.generar_email(nombre, apellidos)
        ratio = round(cuota / ingresos * 100, 2)

        form_nombre = nombre
        form_apellidos = apellidos
        form_nif = nif
        form_fecha_nac = fecha_nac.strftime("%d/%m/%Y")

        # ── Inconsistencias ───────────────────────────────────────────────────
        inconsistencias: list[str] = []
        if es_inconsistente:
            campos = self._rng.sample(
                ["nombre", "apellidos", "NIF", "fecha_nacimiento"],
                k=self._rng.randint(1, 3),
            )
            inconsistencias = [f"Discrepancia en campo: {c}" for c in campos]

            for campo in campos:
                if campo == "nombre":
                    otra = self.generar_persona()
                    form_nombre = otra["nombre"]
                elif campo == "apellidos":
                    otra = self.generar_persona()
                    form_apellidos = otra["apellidos"]
                elif campo == "NIF":
                    form_nif = self._corromper_nif(nif)
                elif campo == "fecha_nacimiento":
                    otra = self.generar_persona()
                    form_fecha_nac = otra["fecha_nacimiento"].strftime("%d/%m/%Y")

        formulario = DatosFormulario(
            sol_nombre=form_nombre,
            sol_apellidos=form_apellidos,
            sol_nif=form_nif,
            sol_fecha_nacimiento=form_fecha_nac,
            sol_domicilio=domicilio["direccion"],
            sol_municipio=domicilio["municipio"],
            sol_provincia=domicilio["provincia"],
            sol_codigo_postal=domicilio["cp"],
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
            ratio_endeudamiento=ratio,
        )

        expediente = Expediente(
            expediente_id=exp_id,
            dni=dni,
            formulario=formulario,
            es_consistente=not es_inconsistente,
            inconsistencias=inconsistencias,
        )
        logger.debug(
            "Expediente %s generado — consistente=%s", exp_id, not es_inconsistente
        )
        return expediente

    def generar_lote(self, n: int) -> list["Expediente"]:
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
            + [self.generar_expediente(forzar_inconsistente=True)  for _ in range(n_inconsistentes)]
        )
        self._rng.shuffle(expedientes)
        logger.info(
            "Lote: %d expedientes (%d consistentes, %d inconsistentes)",
            n, n_consistentes, n_inconsistentes,
        )
        return expedientes

    # ── Utilidades estáticas ──────────────────────────────────────────────────

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
        return nif[8].upper() == _NIF_LETRAS[numero % 23]
