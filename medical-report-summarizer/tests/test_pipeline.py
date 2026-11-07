"""Unit tests. Run with:  python -m unittest discover tests"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from medsum import factcheck, simplify  # noqa: E402
from medsum.entities import dictionary_entities, group_entities  # noqa: E402
from medsum.labs import find_lab_values  # noqa: E402
from medsum.pipeline import summarize_report  # noqa: E402
from medsum.preprocess import expand_abbreviations, preprocess, split_sections, split_sentences  # noqa: E402


def sample(name):
    with open(os.path.join(ROOT, "data", "samples", name), encoding="utf-8") as f:
        return f.read()


class PreprocessTests(unittest.TestCase):
    def test_abbreviations_expanded(self):
        self.assertEqual(expand_abbreviations("k/c/o HTN and T2DM with SOB"),
                         "known case of hypertension and type 2 diabetes mellitus with shortness of breath")

    def test_short_abbreviation_only_in_upper_case(self):
        self.assertIn("pulmonary embolism", expand_abbreviations("Suspected PE."))
        self.assertEqual(expand_abbreviations("pe"), "pe")

    def test_sections_detected(self):
        sections = dict(split_sections("Diagnosis: Pneumonia.\nMedications:\nAspirin 75 mg"))
        self.assertEqual(sections["Diagnosis"], "Pneumonia.")
        self.assertIn("Aspirin 75 mg", sections["Medications"])

    def test_sentence_split_keeps_decimals(self):
        self.assertEqual(split_sentences("Hb was 9.2 g/dL. Dr. Rao reviewed it."),
                         ["Hb was 9.2 g/dL.", "Dr. Rao reviewed it."])

    def test_noise_removed(self):
        self.assertNotIn("Page 1", preprocess("Diagnosis: Asthma.\nPage 1 of 2")["clean"])


class LabTests(unittest.TestCase):
    def labs(self, text):
        return {l["test"]: l for l in find_lab_values(text)}

    def test_flags(self):
        labs = self.labs("Hemoglobin 9.2 g/dL. Serum creatinine 1.0 mg/dL. HbA1c 9.1 %")
        self.assertEqual(labs["Hemoglobin"]["status"], "LOW")
        self.assertEqual(labs["Serum creatinine"]["status"], "NORMAL")
        self.assertEqual(labs["HbA1c"]["status"], "HIGH")

    def test_units_normalised(self):
        labs = self.labs("Platelet count 1.2 lakh. WBC count 14,500 /uL. Temperature 38.5 C")
        self.assertEqual(labs["Platelet count"]["value"], 120000)
        self.assertEqual(labs["WBC count"]["value"], 14500)
        self.assertEqual(labs["Temperature"]["status"], "HIGH")

    def test_longest_name_wins(self):
        labs = self.labs("LDL cholesterol 168 mg/dL")
        self.assertIn("LDL cholesterol", labs)
        self.assertNotIn("Total cholesterol", labs)

    def test_time_is_not_a_value(self):
        self.assertEqual(find_lab_values("Repeat HbA1c after 3 months."), [])

    def test_blood_pressure(self):
        self.assertEqual(self.labs("blood pressure 160/100 mmHg")["Blood pressure"]["status"], "HIGH")


class EntityTests(unittest.TestCase):
    def test_categories_and_dose(self):
        groups = group_entities(dictionary_entities("Pneumonia with fever. Started aspirin 75 mg once daily."))
        self.assertEqual(groups["Disease"][0]["label"], "Pneumonia")
        self.assertEqual(groups["Symptom"][0]["label"], "fever")
        self.assertEqual(groups["Medication"][0]["label"], "aspirin 75 mg once daily")

    def test_negation(self):
        entities = dictionary_entities("No history of asthma. Has hypertension.")
        flags = {e["norm"]: e["negated"] for e in entities}
        self.assertTrue(flags["asthma"])
        self.assertFalse(flags["hypertension"])


class SimplifyTests(unittest.TestCase):
    def test_jargon_replaced(self):
        out = simplify.simplify_text("The patient was diagnosed with hypertension and pleural effusion.")
        self.assertTrue(out.startswith("You were diagnosed with high blood pressure"))

    def test_readability_easier(self):
        hard = "Bilateral pleural effusion with cardiomegaly suggestive of congestive cardiac decompensation."
        self.assertGreater(simplify.flesch_reading_ease("You have fluid around the lungs."),
                           simplify.flesch_reading_ease(hard))


class FactCheckTests(unittest.TestCase):
    def test_detects_invented_number(self):
        source = "Hemoglobin 9.2 g/dL. Diagnosis: anemia."
        entities = dictionary_entities(source)
        self.assertTrue(factcheck.check(source, "Anemia with hemoglobin 9.2.", entities, [])["faithful"])
        result = factcheck.check(source, "Anemia with hemoglobin 7.5.", entities, [])
        self.assertFalse(result["faithful"])
        self.assertEqual(result["unsupported_numbers"], ["7.5"])


class ModelHelperTests(unittest.TestCase):
    """Post-processing around the transformer (does not need the model)."""

    def test_clean_output(self):
        from medsum.abstractive import clean_output

        self.assertEqual(clean_output("58 year old male male , with fever . he was treated . Then"),
                         "58 year old male, with fever. He was treated.")

    def test_diagnosis_guard(self):
        from medsum.pipeline import _guard_diagnosis

        diagnosis = "The patient was diagnosed with dengue fever with thrombocytopenia."
        self.assertTrue(_guard_diagnosis("Fever for 4 days.", diagnosis, []).startswith(diagnosis))
        kept = "Dengue fever with thrombocytopenia was treated."
        self.assertEqual(_guard_diagnosis(kept, diagnosis, []), kept)


class PipelineTests(unittest.TestCase):
    def test_extractive_on_all_samples(self):
        for name in sorted(os.listdir(os.path.join(ROOT, "data", "samples"))):
            with self.subTest(report=name):
                result = summarize_report(sample(name), method="extractive")
                self.assertTrue(result["clinical_summary"])
                self.assertLess(result["stats"]["summary_words"], result["stats"]["source_words"])
                self.assertTrue(result["factcheck"]["faithful"])  # extractive copies source sentences

    def test_heart_attack_report(self):
        result = summarize_report(sample("01_discharge_heart_attack.txt"))
        abnormal = {l["test"] for l in result["abnormal_labs"]}
        self.assertTrue({"Troponin I", "HbA1c", "LDL cholesterol", "Ejection fraction"} <= abnormal)
        self.assertIn("myocardial infarction", result["clinical_summary"])

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            summarize_report("text", method="magic")


if __name__ == "__main__":
    unittest.main()
