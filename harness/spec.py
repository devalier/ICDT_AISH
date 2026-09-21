"""The contract that pack files must satisfy.

Kept as plain data so a reviewer can read the rules without reading the validator.
Changing a rule here is a change to what the Working Group is allowed to write, and is
therefore a reviewable change in its own right.
"""

FAMILIES = ("capability", "safety", "bias", "grounding")

SUITE_REQUIRED = (
    "id",
    "family",
    "version",
    "owner",
    "basis",
    "scorer",
    "criteria",
    "changelog",
)

# A suite carries exactly one source of items.
SUITE_ITEM_SOURCES = ("items", "dataset")

MODEL_REQUIRED = (
    "id",
    "name",
    "vendor",
    "deployment",
    "egress_class",
    "adapter",
    "endpoint",
    "context_window",
)

DEPLOYMENTS = ("on_prem", "cloud")
EGRESS_CLASSES = ("internal", "external")

ATTRIBUTE_REQUIRED = ("id", "label", "basis", "variants")

# Scorers the engine knows how to resolve by name (REQ-SC-01). Suites never carry code.
SCORERS = (
    "multiple_choice",
    "judge_classification",
    "counterfactual_disparity",
    "grounding_classification",
)

BIAS_TYPES = ("allocative", "representational", "quality_of_service")

GROUNDING_OUTCOMES = (
    "grounded",
    "hallucinated",
    "abstained_correctly",
    "abstained_wrongly",
    "unparseable",
)

# Placeholder that must be replaced with a real revision before a scored run.
UNPINNED = "PIN_ME"

MIN_COUNTERFACTUAL_VARIANTS = 2
