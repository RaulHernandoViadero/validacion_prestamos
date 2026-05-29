"""
test_fake_data_factory.py — Tests unitarios de la factoría de datos sintéticos.

Ejecutar con:
    python -m pytest tests/test_fake_data_factory.py -v
"""

import pytest
from datetime import date

from src.data_generation.fake_data_factory import (
    FakeDataFactory,
    _NIF_LETRAS,
)
from src.config import MAX_DEBT_RATIO, MIN_AGE_YEARS


@pytest.fixture
def factory() -> FakeDataFactory:
    """Factoría con semilla fija para resultados reproducibles."""
    return FakeDataFactory(seed=42, inconsistency_rate=0.15)


# ─── Tests de NIF ─────────────────────────────────────────────────────────────

class TestNIF:

    def test_formato_nif(self, factory):
        """El NIF debe tener 8 dígitos + 1 letra."""
        nif = factory.generar_nif()
        assert len(nif) == 9
        assert nif[:8].isdigit()
        assert nif[8].isalpha()

    def test_digito_control_valido(self, factory):
        """La letra de control debe ser la correcta según el algoritmo oficial."""
        for _ in range(50):
            nif = factory.generar_nif()
            assert FakeDataFactory.validar_nif(nif), f"NIF inválido generado: {nif}"

    def test_nif_corrompido_invalido(self, factory):
        """Un NIF corrompido debe fallar la validación."""
        nif = factory.generar_nif()
        nif_malo = factory._corromper_nif(nif)
        assert not FakeDataFactory.validar_nif(nif_malo)

    def test_validar_nif_longitud_incorrecta(self):
        """NIF con longitud incorrecta debe devolver False."""
        assert not FakeDataFactory.validar_nif("1234")
        assert not FakeDataFactory.validar_nif("1234567890X")

    def test_validar_nif_letras_invalidas(self):
        """NIF con caracteres no numéricos en la parte numérica devuelve False."""
        assert not FakeDataFactory.validar_nif("ABCDEFGHX")


# ─── Tests de fechas ──────────────────────────────────────────────────────────

class TestFechas:

    def test_mayor_de_edad(self, factory):
        """La fecha de nacimiento garantiza que el titular es mayor de edad."""
        hoy = date.today()
        for _ in range(50):
            persona = factory.generar_persona()
            fecha_nac_str = persona["fecha_nacimiento"].strftime("%d/%m/%Y")
            # Reconstruir la fecha
            d, m, y = map(int, fecha_nac_str.split("/"))
            fecha_nac = date(y, m, d)
            edad = (hoy - fecha_nac).days // 365
            assert edad >= MIN_AGE_YEARS, f"Persona menor de edad: {edad} años"

    def test_caducidad_posterior_a_emision(self, factory):
        """La fecha de caducidad debe ser posterior a la de emisión."""
        persona = factory.generar_persona()
        emision: date = persona["fecha_emision"]
        caducidad: date = persona["fecha_caducidad"]
        assert caducidad > emision

    def test_caducidad_10_años_despues(self, factory):
        """La caducidad debe ser exactamente 10 años posterior a la emisión."""
        persona = factory.generar_persona()
        emision: date = persona["fecha_emision"]
        caducidad: date = persona["fecha_caducidad"]
        assert caducidad.year - emision.year == 10


# ─── Tests de préstamo ────────────────────────────────────────────────────────

class TestPrestamo:

    def test_cuota_no_supera_35_por_ciento(self, factory):
        """La cuota mensual no debe superar el 35% de los ingresos netos."""
        for _ in range(100):
            ingresos = factory.generar_ingresos("indefinido")
            importe, plazo, _, cuota, tae = factory.generar_prestamo(ingresos)
            ratio = cuota / ingresos
            assert ratio <= MAX_DEBT_RATIO + 0.001, (
                f"Cuota {cuota:.2f}€ supera el 35% de ingresos {ingresos:.2f}€ "
                f"(ratio={ratio:.1%})"
            )

    def test_cuota_francesa_formula(self, factory):
        """Verificar la fórmula de amortización francesa con caso conocido."""
        # Préstamo 10.000€, TAE 6%, 12 meses → cuota ~860.66€
        cuota = factory.calcular_cuota_francesa(10_000, 6.0, 12)
        assert abs(cuota - 860.66) < 1.0, f"Cuota calculada incorrecta: {cuota}"

    def test_cuota_sin_intereses(self, factory):
        """Con TAE=0 la cuota es simplemente importe/plazo."""
        cuota = factory.calcular_cuota_francesa(1_200, 0.0, 12)
        assert abs(cuota - 100.0) < 0.01

    def test_importe_dentro_de_limites(self, factory):
        """El importe generado debe estar dentro de los límites configurados."""
        from src.config import MIN_LOAN_AMOUNT, MAX_LOAN_AMOUNT
        for _ in range(50):
            ingresos = factory.generar_ingresos("funcionario")
            importe, _, _, _, _ = factory.generar_prestamo(ingresos)
            assert MIN_LOAN_AMOUNT <= importe <= MAX_LOAN_AMOUNT


# ─── Tests de MRZ ─────────────────────────────────────────────────────────────

class TestMRZ:

    def test_longitud_lineas_mrz(self, factory):
        """Cada línea MRZ debe tener exactamente 30 caracteres."""
        persona = factory.generar_persona()
        l1, l2 = factory.generar_mrz(
            persona["nombre"],
            persona["apellidos"],
            persona["nif"],
            persona["fecha_nacimiento"],
            persona["fecha_caducidad"],
            persona["sexo"],
        )
        assert len(l1) == 30, f"Línea 1 MRZ tiene {len(l1)} caracteres"
        assert len(l2) == 30, f"Línea 2 MRZ tiene {len(l2)} caracteres"

    def test_mrz_empieza_por_idesp(self, factory):
        """La línea 1 del DNI español debe empezar por 'IDESP'."""
        persona = factory.generar_persona()
        l1, _ = factory.generar_mrz(
            persona["nombre"], persona["apellidos"], persona["nif"],
            persona["fecha_nacimiento"], persona["fecha_caducidad"], persona["sexo"],
        )
        assert l1.startswith("IDESP")

    def test_mrz_sexo_correcto(self, factory):
        """El sexo en la línea 2 de MRZ debe coincidir con el del titular."""
        for _ in range(20):
            persona = factory.generar_persona()
            _, l2 = factory.generar_mrz(
                persona["nombre"], persona["apellidos"], persona["nif"],
                persona["fecha_nacimiento"], persona["fecha_caducidad"], persona["sexo"],
            )
            sexo_mrz = l2[7]  # posición 7 en la línea 2
            assert sexo_mrz == persona["sexo"], (
                f"Sexo en MRZ '{sexo_mrz}' != sexo real '{persona['sexo']}'"
            )


# ─── Tests de expediente ──────────────────────────────────────────────────────

class TestExpediente:

    def test_expediente_consistente_datos_iguales(self, factory):
        """En un expediente consistente, DNI y formulario deben tener los mismos datos."""
        exp = factory.generar_expediente(forzar_inconsistente=False)
        assert exp.es_consistente is True
        assert exp.inconsistencias == []
        assert exp.dni.nombre == exp.formulario.sol_nombre
        assert exp.dni.apellidos == exp.formulario.sol_apellidos
        assert exp.dni.numero_dni == exp.formulario.sol_nif
        assert exp.dni.fecha_nacimiento == exp.formulario.sol_fecha_nacimiento

    def test_expediente_inconsistente_tiene_diferencias(self, factory):
        """Un expediente inconsistente debe tener al menos una discrepancia."""
        exp = factory.generar_expediente(forzar_inconsistente=True)
        assert exp.es_consistente is False
        assert len(exp.inconsistencias) >= 1

        campos_cambiados = 0
        if exp.dni.nombre != exp.formulario.sol_nombre:
            campos_cambiados += 1
        if exp.dni.apellidos != exp.formulario.sol_apellidos:
            campos_cambiados += 1
        if exp.dni.numero_dni != exp.formulario.sol_nif:
            campos_cambiados += 1
        if exp.dni.fecha_nacimiento != exp.formulario.sol_fecha_nacimiento:
            campos_cambiados += 1

        assert campos_cambiados >= 1, "Expediente marcado como inconsistente pero sin diferencias reales"

    def test_expediente_id_unico(self, factory):
        """Los IDs de expediente deben ser únicos."""
        ids = [factory.generar_expediente().expediente_id for _ in range(20)]
        assert len(set(ids)) == len(ids), "IDs de expediente duplicados"

    def test_expediente_serializable_a_dict(self, factory):
        """El expediente debe poder serializarse a dict (para guardar como JSON)."""
        exp = factory.generar_expediente()
        d = exp.to_dict()
        assert "expediente_id" in d
        assert "dni" in d
        assert "formulario" in d
        assert "es_consistente" in d
        assert "inconsistencias" in d

    def test_nif_del_formulario_valido_en_consistente(self, factory):
        """En expedientes consistentes, el NIF del formulario debe ser válido."""
        for _ in range(20):
            exp = factory.generar_expediente(forzar_inconsistente=False)
            assert FakeDataFactory.validar_nif(exp.formulario.sol_nif)

    def test_nif_del_formulario_invalido_en_inconsistente_con_nif(self, factory):
        """Si la inconsistencia es en el NIF, el NIF del formulario debe ser inválido."""
        encontrado = False
        for _ in range(50):
            exp = factory.generar_expediente(forzar_inconsistente=True)
            if any("NIF" in inc for inc in exp.inconsistencias):
                assert not FakeDataFactory.validar_nif(exp.formulario.sol_nif)
                encontrado = True
                break
        assert encontrado, "No se generó ningún expediente con inconsistencia en NIF en 50 intentos"


# ─── Tests de lote ────────────────────────────────────────────────────────────

class TestLote:

    def test_tasa_inconsistencias_aproximada(self, factory):
        """La tasa de inconsistencias del lote debe estar cerca del 15%."""
        lote = factory.generar_lote(200)
        n_inconsistentes = sum(1 for e in lote if not e.es_consistente)
        tasa_real = n_inconsistentes / len(lote)
        # Tolerancia ±5%
        assert abs(tasa_real - 0.15) <= 0.05, (
            f"Tasa de inconsistencias {tasa_real:.1%} muy alejada del 15%"
        )

    def test_lote_tamanyo_correcto(self, factory):
        """El lote debe tener exactamente el número de expedientes solicitado."""
        for n in [1, 10, 50, 100]:
            lote = factory.generar_lote(n)
            assert len(lote) == n

    def test_reproducibilidad_con_misma_semilla(self):
        """Dos factorías con la misma semilla deben generar los mismos datos."""
        f1 = FakeDataFactory(seed=99)
        f2 = FakeDataFactory(seed=99)
        exp1 = f1.generar_expediente()
        exp2 = f2.generar_expediente()
        assert exp1.dni.numero_dni == exp2.dni.numero_dni
        assert exp1.dni.nombre == exp2.dni.nombre
