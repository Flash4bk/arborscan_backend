"""Independent SI checks for the first supported scientific scenario.

No tree, field accuracy, beta, root capacity or wind probability is inferred.
"""
import copy
import math
import unittest

from arborscan_science.section import ScienceInputError, section_capacity


def scenario(diameter=0.2, mor=50_000_000.0):
    return {
        'schema_version': 1,
        'diameter': {'value': diameter, 'unit': 'm',
                     'origin': 'scenario_assumption', 'source': 'DEMO known circular section'},
        'effective_mor': {'value': mor, 'unit': 'Pa',
                          'origin': 'scenario_assumption', 'source': 'DEMO supplied effective bending strength'},
        'section_model': 'healthy_solid_circular',
        'strength_definition': 'Effective MOR supplied; modifiers are already included once.',
    }


class CircularSectionCapacityTests(unittest.TestCase):
    def test_known_circle_independent_section_modulus(self):
        result = section_capacity(scenario())
        # Closed-form diameter expression, independent of the I/radius implementation.
        expected = math.pi * 50_000_000 * .2**3 / 32
        self.assertAlmostEqual(result['capacity_nm'], expected, places=8)
        self.assertEqual(result['unit'], 'N m')
        self.assertEqual(result['method_id'], 'healthy_circular_section_mor_v1')
        self.assertEqual(result['validation_status'], 'scenario_software_only')
        self.assertIsNone(result['uncertainty_interval'])

    def test_diameter_cubed_and_strength_linear(self):
        a = section_capacity(scenario())['capacity_nm']
        self.assertAlmostEqual(section_capacity(scenario(.4))['capacity_nm'], 8*a)
        self.assertAlmostEqual(section_capacity(scenario(mor=25_000_000))['capacity_nm'], .5*a)

    def test_input_is_not_mutated_and_content_hash_is_stable(self):
        value = scenario(); before = copy.deepcopy(value)
        first = section_capacity(value)
        second = section_capacity(dict(reversed(list(value.items()))))
        self.assertEqual(value, before)
        self.assertEqual(first['input_sha256'], second['input_sha256'])
        self.assertNotEqual(first['input_sha256'], section_capacity(scenario(.21))['input_sha256'])

    def test_no_derived_beta_general_wind_or_safe_verdict(self):
        result = section_capacity(scenario())
        for key in ['beta_kg_s', 'critical_wind_m_s', 'root_capacity_nm', 'annual_failure_probability']:
            self.assertIsNone(result[key])
        self.assertFalse(result['field_validated'])
        self.assertTrue(result['limitations'])

    def test_invalid_numeric_inputs(self):
        for bad in [0, -1, True, '0.2', None, float('nan'), float('inf')]:
            for field in ['diameter', 'effective_mor']:
                data = scenario(); data[field]['value'] = bad
                with self.subTest(field=field, value=str(bad)):
                    with self.assertRaises(ScienceInputError): section_capacity(data)

    def test_si_units_and_provenance_required(self):
        for field, unit in [('diameter', 'cm'), ('effective_mor', 'MPa')]:
            data = scenario(); data[field]['unit'] = unit
            with self.assertRaises(ScienceInputError): section_capacity(data)
        for field in ['diameter', 'effective_mor']:
            for key in ['source', 'origin']:
                data = scenario(); data[field].pop(key)
                with self.assertRaises(ScienceInputError): section_capacity(data)

    def test_untrusted_structure_never_bypasses_validation(self):
        for origin in [[], {}, 1, True]:
            data = scenario(); data['diameter']['origin'] = origin
            with self.subTest(origin=str(origin)):
                with self.assertRaises(ScienceInputError): section_capacity(data)
        for field in ['beta_kg_s', 'capacity_nm', 'verified']:
            data = scenario(); data[field] = 123
            with self.assertRaises(ScienceInputError): section_capacity(data)

    def test_domain_unknown_or_defective_section_cannot_be_silently_healthy(self):
        for model in ['unknown', 'hollow', 'decayed', 'dbh', None]:
            data = scenario(); data['section_model'] = model
            with self.assertRaises(ScienceInputError): section_capacity(data)
        data = scenario(); data['strength_definition'] = ''
        with self.assertRaises(ScienceInputError): section_capacity(data)

    def test_unknown_uncertainty_is_not_zero_and_bad_uncertainty_is_rejected(self):
        data = scenario(); data['diameter']['uncertainty'] = None
        self.assertIsNone(section_capacity(data)['uncertainty_interval'])
        data['diameter']['uncertainty'] = -1
        with self.assertRaises(ScienceInputError): section_capacity(data)

    def test_overflow_underflow_and_unknown_schema_are_explicit_errors(self):
        for d, mor in [(1e200, 1e200), (1e-200, 1e-200)]:
            with self.assertRaises(ScienceInputError): section_capacity(scenario(d, mor))
        data = scenario(); data['schema_version'] = 999
        with self.assertRaises(ScienceInputError): section_capacity(data)


if __name__ == '__main__':
    unittest.main()
