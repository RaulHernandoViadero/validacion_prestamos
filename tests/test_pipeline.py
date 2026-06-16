"""
test_pipeline.py — Tests unitarios del pipeline de verificación documental.

Prueba el pipeline end-to-end en modo sin modelos (YOLO y clasificador
deshabilitados) para que los tests pasen sin pesos entrenados.

Cobertura:
  - DocumentPipeline.procesar() con imágenes dummy
  - ExpedienteResult estructura y serialización
  - Tolerancia a fallos: cada etapa falla graciosamente
  - PipelineConfig personalización
  - business_rules: cuota francesa, MRZ, perfil riesgo, coherencia fechas
  - cross_validator: R01–R09 con campos conocidos
  - verdict_engine: veredicto con confianzas sintéticas
"""

import pytest
from datetime import date
from PIL import Image

from src.pipeline.document_pipeline import DocumentPipeline, PipelineConfig, ExpedienteResult
from src.validation.business_rules import (
    calcular_digito_control_mrz,
    calcular_cuota_francesa,
    verificar_cuota_francesa,
    evaluar_perfil_riesgo,
    verificar_coherencia_fechas,
    validar_limites_prestamo,
    validar_finalidad,
)
from src.validation.cross_validator import CrossValidator
from src.validation.verdict_engine import VerdictEngine, VEREDICTO_APTO, VEREDICTO_INCONSISTENTE
from src.ocr.text_postprocessor import CampoNormalizado


# ── Helpers ───────────────────────────────────────────────────────────────────

def _img_dummy(w: int = 200, h: int = 150) -> Image.Image:
    return Image.new("RGB", (w, h), color=(240, 240, 240))


def _campo(clase: str, texto: str, valido: bool = True, conf: float = 0.9) -> CampoNormalizado:
    return CampoNormalizado(
        clase=clase,
        texto_raw=texto,
        texto_norm=texto,
        confianza_ocr=conf,
        valido=valido,
    )


def _pipeline_test() -> DocumentPipeline:
    """Pipeline con YOLO y clasificador deshabilitados."""
    return DocumentPipeline(config=PipelineConfig(
        usar_yolo=False,
        usar_clasificador=False,
        anotar_imagen=False,
    ))


# ── Tests del pipeline ────────────────────────────────────────────────────────

class TestDocumentPipeline:

    def test_procesar_devuelve_expediente_result(self):
        """procesar() devuelve un ExpedienteResult con todos los campos."""
        p = _pipeline_test()
        r = p.procesar(_img_dummy(), _img_dummy(), expediente_id="UT-001")
        assert isinstance(r, ExpedienteResult)
        assert r.expediente_id == "UT-001"

    def test_procesar_id_autogenerado(self):
        """Si no se pasa ID se genera automáticamente."""
        p = _pipeline_test()
        r = p.procesar(_img_dummy(), _img_dummy())
        assert r.expediente_id and len(r.expediente_id) > 0

    def test_procesar_siempre_tiene_veredicto(self):
        """procesar() siempre produce un veredicto (nunca None)."""
        p = _pipeline_test()
        r = p.procesar(_img_dummy(), _img_dummy())
        assert r.veredicto is not None
        assert r.veredicto.resultado in (VEREDICTO_APTO, VEREDICTO_INCONSISTENTE)

    def test_procesar_mide_tiempos(self):
        """Las métricas de tiempo se registran en todas las etapas."""
        p = _pipeline_test()
        r = p.procesar(_img_dummy(), _img_dummy())
        assert r.tiempo_total_ms > 0
        etapas_esperadas = [
            "deteccion_ms", "ocr_ms", "clasificacion_ms",
            "validacion_ms", "reglas_negocio_ms", "veredicto_ms",
        ]
        for etapa in etapas_esperadas:
            assert etapa in r.tiempos_etapas, f"Falta etapa '{etapa}'"

    def test_to_dict_serializable(self):
        """to_dict() devuelve un diccionario JSON-serializable."""
        import json
        p = _pipeline_test()
        r = p.procesar(_img_dummy(), _img_dummy())
        d = r.to_dict()
        # Debe serializar a JSON sin errores
        json_str = json.dumps(d)
        assert len(json_str) > 100

    def test_advertencias_son_lista(self):
        """Las advertencias son siempre una lista (nunca None)."""
        p = _pipeline_test()
        r = p.procesar(_img_dummy(), _img_dummy())
        assert isinstance(r.advertencias, list)

    def test_config_personalizada(self):
        """PipelineConfig respeta los parámetros pasados."""
        cfg = PipelineConfig(tolerancia_nombres=0.70, umbral_confianza_ocr=0.30)
        p = DocumentPipeline(config=cfg)
        assert p._cfg.tolerancia_nombres == 0.70
        assert p._cfg.umbral_confianza_ocr == 0.30


# ── Tests de business_rules ───────────────────────────────────────────────────

class TestBusinessRules:

    # MRZ
    def test_digito_control_mrz_numeros(self):
        assert calcular_digito_control_mrz("740812") == 2

    def test_digito_control_mrz_relleno(self):
        """El carácter '<' vale 0."""
        assert calcular_digito_control_mrz("<<<<<<<<") == 0

    def test_digito_control_mrz_letras(self):
        """'A' = 10, peso 7 → 70 % 10 = 0."""
        assert calcular_digito_control_mrz("A") == 0

    # Cuota francesa
    def test_cuota_francesa_calculo(self):
        c = calcular_cuota_francesa(10000, 5.5, 36)
        assert abs(c - 301.96) < 0.05

    def test_cuota_francesa_sin_intereses(self):
        """Con TAE=0 la cuota es capital / plazo."""
        c = calcular_cuota_francesa(12000, 0.0, 12)
        assert c == 1000.0

    def test_cuota_francesa_parametros_invalidos(self):
        assert calcular_cuota_francesa(0, 5.5, 36) == 0.0
        assert calcular_cuota_francesa(10000, 5.5, 0) == 0.0

    def test_verificar_cuota_correcta(self):
        r = verificar_cuota_francesa(301.96, 10000, 5.5, 36)
        assert r.valido

    def test_verificar_cuota_incorrecta(self):
        r = verificar_cuota_francesa(999.99, 10000, 5.5, 36)
        assert not r.valido

    # Perfil de riesgo
    def test_perfil_riesgo_bajo(self):
        p = evaluar_perfil_riesgo(3000, 300, 10000, 36)
        assert p.nivel == "BAJO"
        assert abs(p.ratio_deuda - 0.10) < 0.01

    def test_perfil_riesgo_medio(self):
        p = evaluar_perfil_riesgo(2000, 500, 10000, 36)
        assert p.nivel == "MEDIO"

    def test_perfil_riesgo_alto(self):
        p = evaluar_perfil_riesgo(1000, 400, 10000, 36)
        assert p.nivel == "ALTO"

    def test_perfil_riesgo_con_fecha(self):
        p = evaluar_perfil_riesgo(2000, 300, 10000, 36, "15/06/1985")
        assert p.edad > 0

    # Límites del préstamo
    def test_limites_validos(self):
        ok, errs = validar_limites_prestamo(15000, 60, 6.5)
        assert ok and not errs

    def test_limites_importe_bajo(self):
        ok, errs = validar_limites_prestamo(500, 60, 6.5)
        assert not ok
        assert any("mporte" in e for e in errs)

    def test_limites_plazo_excesivo(self):
        ok, errs = validar_limites_prestamo(15000, 400, 6.5)
        assert not ok

    def test_limites_tae_invalida(self):
        ok, errs = validar_limites_prestamo(15000, 60, 35.0)
        assert not ok

    # Coherencia de fechas
    def test_coherencia_fechas_ok(self):
        resultados = verificar_coherencia_fechas("15/06/1985", "20/03/2019", "20/03/2029")
        assert all(v for _, v, _ in resultados)

    def test_coherencia_fechas_caducado(self):
        resultados = verificar_coherencia_fechas("15/06/1985", "20/03/2010", "20/03/2020")
        # La fecha 2020 es anterior a hoy → debe fallar "Documento vigente"
        vigencia = [(d, v, _) for d, v, _ in resultados if "vigente" in d.lower() or "vigent" in d.lower()]
        assert vigencia and not vigencia[0][1]

    # Finalidad
    def test_finalidad_valida(self):
        ok, _ = validar_finalidad("VIVIENDA")
        assert ok

    def test_finalidad_invalida(self):
        ok, _ = validar_finalidad("VACACIONES EN MARTE")
        assert not ok


# ── Tests del CrossValidator ──────────────────────────────────────────────────

class TestCrossValidator:

    def _campos_ok(self) -> tuple[dict, dict]:
        """Devuelve campos DNI y formulario perfectamente coherentes."""
        hoy = date.today()
        cad = date(hoy.year + 5, hoy.month, hoy.day)
        nac = date(1985, 6, 15)

        campos_dni = {
            "nombre":           _campo("nombre",           "JUAN"),
            "apellidos":        _campo("apellidos",        "GARCIA LOPEZ"),
            "numero_dni":       _campo("numero_dni",       "12345678Z"),
            "fecha_nacimiento": _campo("fecha_nacimiento", f"{nac.day:02d}/{nac.month:02d}/{nac.year}"),
            "fecha_caducidad":  _campo("fecha_caducidad",  f"{cad.day:02d}/{cad.month:02d}/{cad.year}"),
        }
        campos_form = {
            "sol_nombre":           _campo("sol_nombre",           "JUAN"),
            "sol_apellidos":        _campo("sol_apellidos",        "GARCIA LOPEZ"),
            "sol_nif":              _campo("sol_nif",              "12345678Z"),
            "sol_fecha_nacimiento": _campo("sol_fecha_nacimiento",
                                           f"{nac.day:02d}/{nac.month:02d}/{nac.year}"),
            "sol_ingresos_netos":   _campo("sol_ingresos_netos",   "2000.00"),
            "prestamo_cuota":       _campo("prestamo_cuota",       "350.00"),
            "prestamo_importe":     _campo("prestamo_importe",     "15000.00"),
        }
        return campos_dni, campos_form

    def test_expediente_consistente(self):
        """Campos perfectamente coherentes → todas las reglas pasan."""
        v = CrossValidator()
        dni, form = self._campos_ok()
        r = v.validar(dni, form)
        # R09 pasa como WARNING sin clasificador
        assert r.n_reglas_ok >= 8

    def test_r01_nombre_falla(self):
        """R01 falla si los nombres no coinciden."""
        v = CrossValidator()
        dni, form = self._campos_ok()
        form["sol_nombre"] = _campo("sol_nombre", "PEDRO")
        r = v.validar(dni, form)
        assert "R01" in r.reglas_fallidas

    def test_r03_nif_invalido_falla(self):
        """R03 falla si el NIF no es válido."""
        v = CrossValidator()
        dni, form = self._campos_ok()
        dni["numero_dni"]  = _campo("numero_dni",  "00000000A")
        form["sol_nif"]    = _campo("sol_nif",     "00000000A")
        r = v.validar(dni, form)
        assert "R03" in r.reglas_fallidas

    def test_r07_ratio_alto_falla(self):
        """R07 falla si la cuota supera el 35% de los ingresos."""
        v = CrossValidator()
        dni, form = self._campos_ok()
        form["sol_ingresos_netos"] = _campo("sol_ingresos_netos", "1000.00")
        form["prestamo_cuota"]     = _campo("prestamo_cuota",     "400.00")  # 40%
        r = v.validar(dni, form)
        assert "R07" in r.reglas_fallidas

    def test_r09_manipulado_falla(self):
        """R09 falla si el clasificador detecta manipulación."""
        v = CrossValidator()
        dni, form = self._campos_ok()
        r = v.validar(dni, form,
                      autenticidad_dni={"etiqueta": "MANIPULADO", "confianza": 0.92},
                      autenticidad_form={"etiqueta": "LEGITIMO",  "confianza": 0.88})
        assert "R09" in r.reglas_fallidas


# ── Tests del VerdictEngine ───────────────────────────────────────────────────

class TestVerdictEngine:

    def _resultado_validacion_ok(self):
        from src.validation.cross_validator import ResultadoValidacion, ResultadoRegla
        rv = ResultadoValidacion()
        for codigo in ["R01","R02","R03","R04","R05","R06","R07","R08","R09"]:
            rv.agregar_regla(ResultadoRegla(
                codigo=codigo, nombre=f"Regla {codigo}",
                pasada=True, detalle="OK",
            ))
        return rv

    def _resultado_validacion_con_fallos(self, codigos_fallo: list):
        from src.validation.cross_validator import ResultadoValidacion, ResultadoRegla
        rv = ResultadoValidacion()
        for codigo in ["R01","R02","R03","R04","R05","R06","R07","R08","R09"]:
            rv.agregar_regla(ResultadoRegla(
                codigo=codigo, nombre=f"Regla {codigo}",
                pasada=(codigo not in codigos_fallo),
                detalle="OK" if codigo not in codigos_fallo else "FALLO",
            ))
        return rv

    def test_veredicto_apto(self):
        engine = VerdictEngine()
        campos = {
            "numero_dni": _campo("numero_dni", "12345678Z", conf=0.95),
            "nombre":     _campo("nombre",     "JUAN",       conf=0.91),
        }
        rv     = self._resultado_validacion_ok()
        v      = engine.emitir("EXP-001", rv, campos, campos)
        assert v.resultado == VEREDICTO_APTO
        assert v.confianza > 0.7

    def test_veredicto_inconsistente(self):
        engine = VerdictEngine()
        campos = {}
        rv     = self._resultado_validacion_con_fallos(["R01", "R07"])
        v      = engine.emitir("EXP-002", rv, campos, campos)
        assert v.resultado == VEREDICTO_INCONSISTENTE
        assert "R01" in v.reglas_fallidas
        assert "R07" in v.reglas_fallidas

    def test_confianza_penalizada_con_ocr_bajo(self):
        """Con confianza OCR muy baja, la confianza del veredicto se reduce."""
        engine = VerdictEngine(umbral_confianza_ocr=0.5)
        campos = {
            "numero_dni": _campo("numero_dni", "12345678Z", conf=0.1),
        }
        rv     = self._resultado_validacion_ok()
        v      = engine.emitir("EXP-003", rv, campos, {})
        # Con OCR bajo debe penalizarse (×0.7)
        assert v.confianza < 0.9

    def test_veredicto_tiene_timestamp(self):
        engine = VerdictEngine()
        rv     = self._resultado_validacion_ok()
        v      = engine.emitir("EXP-004", rv, {}, {})
        assert v.timestamp and len(v.timestamp) > 10
