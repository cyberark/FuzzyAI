
import threading
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_cli_execution.log"),
        logging.StreamHandler()
    ]
)

def run_pytest_command():
    """Run the pytest command."""
    os.system('pytest tests/test_cli_execution.py --maxfail=1 --disable-warnings')

def test_cli_execution_with_timeout():
    """Test CLI execution with timeout."""
    try:
        # Log the start of the test
        logging.info("Starting CLI execution test with timeout...")

        # Create a thread to run the pytest command
        pytest_thread = threading.Thread(target=run_pytest_command)

        # Start the thread
        pytest_thread.start()

        # Wait for the thread to finish with a timeout
        pytest_thread.join(timeout=300)  # 300 seconds (5 minutes)

        # Check if the thread is still alive (did not finish in time)
        if pytest_thread.is_alive():
            logging.error("CLI command timed out after 300 seconds.")
            assert False, "CLI command timed out after 300 seconds."
        else:
            logging.info("CLI execution test passed successfully.")

    except Exception as e:
        logging.critical("An unexpected error occurred during CLI execution: %s", e)
        assert False, f"An unexpected error occurred: {e}"
