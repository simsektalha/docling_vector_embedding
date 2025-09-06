import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_placeholder_ingest_search_smoke(tmp_path: Path):
    # Placeholder integration test to keep CI light. Can be expanded to spin up Qdrant.
    assert True


