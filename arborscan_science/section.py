"""Bending capacity of one assumed healthy, solid, circular section in SI.

Independent section-modulus implementation: M = effective_MOR * I / radius.
The caller supplies effective strength, including any justified modifiers.
No species defaults, empirical wind model or copied ForestGALES code is used.
Reference: Hale et al. 2015, doi:10.1016/j.envsoft.2015.01.016,
stem-breakage expression; material/section applicability must be established
separately. This is not whole-tree stability or experimental validation.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


METHOD_ID = 'healthy_circular_section_mor_v1'
_ORIGINS = {'measured', 'derived', 'external_prior', 'scenario_assumption'}


class ScienceInputError(ValueError):
    def __init__(self, code: str, field: str, message: str):
        super().__init__(message)
        self.code, self.field, self.message = code, field, message

    def detail(self) -> dict[str, str]:
        return {'code': self.code, 'field': self.field, 'message': self.message}


def _positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScienceInputError('invalid_number', field, 'Нужно конечное положительное число.')
    try:
        number = float(value)
    except (ValueError, OverflowError):
        number = math.inf
    if not math.isfinite(number) or number <= 0:
        raise ScienceInputError('invalid_number', field, 'Нужно конечное положительное число.')
    return number


def _parameter(value: Any, field: str, unit: str) -> float:
    if not isinstance(value, dict) or value.get('unit') != unit:
        raise ScienceInputError('invalid_unit', field, f'Вычислительное ядро принимает {unit}.')
    if not isinstance(value.get('origin'), str) or value['origin'] not in _ORIGINS:
        raise ScienceInputError('missing_provenance', field, 'Укажите происхождение величины.')
    source = value.get('source')
    if not isinstance(source, str) or not source.strip() or len(source) > 512:
        raise ScienceInputError('missing_provenance', field, 'Укажите источник величины, до 512 символов.')
    # A scalar with no definition is not a standard deviation or confidence interval.
    if value.get('uncertainty') is not None:
        raise ScienceInputError('undefined_uncertainty', field,
                                'Неопределённость требует отдельного определения; неизвестная не равна нулю.')
    return _positive(value.get('value'), field)


def _moment(diameter: float, strength: float) -> float:
    try:
        radius = diameter / 2
        moment = strength * (math.pi * radius ** 4 / 4) / radius
    except (OverflowError, ZeroDivisionError):
        moment = math.inf
    if not math.isfinite(moment) or moment <= 0:
        raise ScienceInputError('numerical_domain', 'diameter/effective_mor',
                                'Величины вне поддержанного численного диапазона.')
    return moment


def section_capacity(inputs: Any) -> dict[str, Any]:
    """Return a deterministic scenario without mutating supplied measurements."""
    if not isinstance(inputs, dict) or type(inputs.get('schema_version')) is not int or inputs['schema_version'] != 1:
        raise ScienceInputError('unsupported_schema', 'schema_version', 'Неподдержанная версия научных входов.')
    allowed = {'schema_version', 'diameter', 'effective_mor', 'section_model', 'strength_definition'}
    if set(inputs) - allowed:
        raise ScienceInputError('unknown_field', 'inputs', 'Сценарий содержит неподдержанное поле.')
    if inputs.get('section_model') != 'healthy_solid_circular':
        raise ScienceInputError('unsupported_domain', 'section_model',
                                'Поддержан сценарий здорового сплошного круглого сечения.')
    definition = inputs.get('strength_definition')
    if not isinstance(definition, str) or not definition.strip() or len(definition) > 1024:
        raise ScienceInputError('missing_strength_definition', 'strength_definition',
                                'Объясните effective MOR и уже учтённые поправки.')
    diameter = _parameter(inputs.get('diameter'), 'diameter', 'm')
    strength = _parameter(inputs.get('effective_mor'), 'effective_mor', 'Pa')
    value = _moment(diameter, strength)
    try:
        canonical = json.dumps(inputs, sort_keys=True, ensure_ascii=False,
                               separators=(',', ':'), allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        raise ScienceInputError('invalid_input_structure', 'inputs', 'Неподдержанная структура входов.') from None
    if len(canonical.encode('utf-8')) > 8192:
        raise ScienceInputError('input_too_large', 'inputs', 'Слишком большой пакет входных данных.')
    return {
        'schema_version': 1,
        'method_id': METHOD_ID,
        'method_version': '1.0.0',
        'input_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
        'input_snapshot': json.loads(canonical),
        'capacity_nm': value,
        'unit': 'N m',
        'quantity': 'single_section_bending_capacity',
        'validation_status': 'scenario_software_only',
        'field_validated': False,
        'uncertainty_interval': None,
        'uncertainty_reason': 'Неопределённость входов не задана в поддержанной форме.',
        'beta_kg_s': None,
        'critical_wind_m_s': None,
        'root_capacity_nm': None,
        'annual_failure_probability': None,
        'source_doi': '10.1016/j.envsoft.2015.01.016',
        'limitations': [
            'Результат относится только к одному заданному круглому сечению и effective MOR.',
            'Не оценивает скрытые дефекты, другие сечения, корни, кручение или сжатие.',
            'Не определяет нагрузку, критический ветер, beta или вероятность отказа.',
            'Не подтверждает пригодность введённых свойств конкретному дереву.',
        ],
    }
