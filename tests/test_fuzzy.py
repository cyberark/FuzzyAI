import sys
import os

# Add the fuzzy module directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'fuzzy'))

from fuzzy.fuzzer import Fuzzer

def test_fuzzer_initialization():
    """A simple test to ensure the Fuzzer is initialized properly."""
    fuzzer = Fuzzer(db_address="localhost:27017")
    assert fuzzer is not None

