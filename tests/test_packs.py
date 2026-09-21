"""The validator's guard rails are the Working Group's safety net — test them.

Run: python3 -m unittest discover -s tests
"""

import copy
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.loader import load  # noqa: E402
from harness.validate import validate  # noqa: E402


def errors(packs):
    return [p for p in validate(packs) if p.severity == "error"]


def messages(problems):
    return " | ".join(f"{p.file}:{p.path}: {p.message}" for p in problems)


class ShippedPacksTest(unittest.TestCase):
    def setUp(self):
        self.packs = load()

    def test_shipped_packs_have_no_errors(self):
        found = errors(self.packs)
        self.assertEqual([], found, messages(found))

    def test_all_four_families_are_present(self):
        families = {s.get("family") for s in self.packs.suites.values()}
        self.assertEqual({"capability", "safety", "bias", "grounding"}, families)

    def test_every_euiba_model_is_registered(self):
        self.assertEqual(
            {
                "llama-3.3-70b",
                "mistral-small-24b",
                "gpt-oss-120b",
                "gpt-5.1",
                "claude-4.6",
            },
            self.packs.model_ids,
        )

    def test_eu_mmlu_covers_the_sixteen_released_languages(self):
        suite = self.packs.suites["suites/eu_mmlu.yaml"]
        self.assertEqual(16, len(suite["languages"]))
        self.assertIn("en", suite["languages"])
        self.assertEqual("en", suite["reference_language"])

    def test_eu_mmlu_consumes_the_published_schema_unmodified(self):
        schema = self.packs.suites["suites/eu_mmlu.yaml"]["dataset"]["schema"]
        self.assertEqual("Language", schema["language"])
        self.assertEqual(
            ["Choice_0", "Choice_1", "Choice_2", "Choice_3"], schema["choices"]
        )

    def test_every_suite_declares_a_normative_basis(self):
        for path, suite in self.packs.suites.items():
            self.assertTrue(suite.get("basis"), f"{path} has no basis")

    def test_eu_specific_bias_axes_are_covered(self):
        eu_axes = {
            a["id"]
            for a in self.packs.attributes["attributes"]
            if a.get("eu_specific")
        }
        self.assertIn("nationality", eu_axes)
        self.assertIn("language", eu_axes)


class GuardRailTest(unittest.TestCase):
    """Each rule must actually fire. A validator that never says no is decoration."""

    def setUp(self):
        self.packs = load()

    def suite(self, name="suites/euiba_bias.yaml"):
        self.packs = copy.deepcopy(self.packs)
        return self.packs.suites[name]

    def assertFires(self, fragment):
        found = errors(self.packs)
        self.assertTrue(
            any(fragment in p.message for p in found),
            f"expected an error containing {fragment!r}; got: {messages(found)}",
        )

    def test_restricted_suite_may_not_target_a_cloud_model(self):
        suite = self.suite("suites/ailuminate.yaml")
        suite["applies_to"].append("gpt-5.1")
        self.assertFires("restricted suite targets cloud model")

    def test_unknown_criteria_reference_is_rejected(self):
        self.suite()["criteria"] = "does_not_exist"
        self.assertFires("is not defined in criteria.yaml")

    def test_unknown_scorer_is_rejected(self):
        self.suite()["scorer"] = "vibes"
        self.assertFires("unknown scorer")

    def test_bias_axis_must_exist_in_attributes(self):
        self.suite()["items"][0]["axis"] = "astrological_sign"
        self.assertFires("is not defined in attributes.yaml")

    def test_bias_template_must_substitute_its_axis(self):
        self.suite()["items"][0]["template"] = "No placeholder here."
        self.assertFires("does not substitute the axis under test")

    def test_bias_item_must_state_a_harm_hypothesis(self):
        del self.suite()["items"][0]["harm_hypothesis"]
        self.assertFires("must state the harm it hypothesises")

    def test_suite_needs_exactly_one_item_source(self):
        suite = self.suite("suites/eu_mmlu.yaml")
        suite["items"] = [{"id": "x"}]
        self.assertFires("exactly one of 'items' or 'dataset'")

    def test_non_semantic_version_is_rejected(self):
        self.suite()["version"] = "v1"
        self.assertFires("is not a semantic version")

    def test_changelog_head_must_match_suite_version(self):
        self.suite()["version"] = "0.2.0"
        self.assertFires("bump one of them")

    def test_empty_changelog_is_rejected(self):
        self.suite()["changelog"] = []
        self.assertFires("every suite change must be recorded")

    def test_grounding_expected_outcome_must_be_a_known_class(self):
        self.suite("suites/euiba_grounding.yaml")["items"][0]["expected"] = "probably_fine"
        self.assertFires("is not one of")

    def test_duplicate_item_ids_are_rejected(self):
        suite = self.suite()
        suite["items"].append(copy.deepcopy(suite["items"][0]))
        self.assertFires("duplicate item id")

    def test_cloud_model_may_not_claim_internal_egress(self):
        self.packs = copy.deepcopy(self.packs)
        for model in self.packs.models["models"]:
            if model["id"] == "gpt-5.1":
                model["egress_class"] = "internal"
        self.assertFires("cannot carry egress_class 'internal'")

    def test_inlined_credential_is_rejected(self):
        self.packs = copy.deepcopy(self.packs)
        self.packs.models["models"][0]["credentials_env"] = "sk-not-a-real-key-abcdef"
        self.assertFires("looks like an inlined credential")

    def test_unknown_judge_reference_is_rejected(self):
        self.suite("suites/ailuminate.yaml")["judge"] = "nobody"
        self.assertFires("is not defined under judges")

    def test_threshold_bands_must_be_ordered_by_direction(self):
        self.packs = copy.deepcopy(self.packs)
        # hallucination_rate is lower_is_better in spirit; make pass looser than watch.
        self.packs.criteria["criteria"]["capability_default"]["metrics"]["accuracy"] = {
            "pass": 0.50,
            "watch": 0.60,
        }
        self.assertFires("must be stricter than")

    def test_composite_weights_must_sum_to_one(self):
        self.packs = copy.deepcopy(self.packs)
        self.packs.criteria["composite"]["weights"]["safety"] = 0.9
        self.assertFires("expected 1.0")


if __name__ == "__main__":
    unittest.main()
