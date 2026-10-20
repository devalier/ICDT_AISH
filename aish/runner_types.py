"""Exception shared by the runner and the judge (kept apart to avoid an import cycle)."""


class RunError(Exception):
    """A run could not be built or executed. Message is safe to show a user."""
