# ItemRepository integration tests require a live database.
# They will be added when a test-database fixture is introduced.
# For now, verify the module imports cleanly.

from ai_intel.sources.repository import ItemRepository  # noqa: F401
