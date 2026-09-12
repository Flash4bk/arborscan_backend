import unittest

from arborscan_v4.species_profiles import get_mechanical_profile


class SpeciesProfileTests(unittest.TestCase):
    def test_horse_chestnut_is_not_mapped_to_pine(self):
        result = get_mechanical_profile("Aesculus hippocastanum")
        self.assertFalse(result["available"])
        self.assertEqual(result["scientific_name"], "Aesculus hippocastanum")
        self.assertEqual(result["properties"], {})
        self.assertEqual(result["reason"], "mechanical_profile_not_validated")
        self.assertNotIn("Сосна", str(result))

    def test_missing_species_has_explicit_reason(self):
        result = get_mechanical_profile(None)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "species_not_identified")


if __name__ == "__main__":
    unittest.main()
