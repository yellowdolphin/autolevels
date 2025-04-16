from pathlib import Path
from test_cli import ICC_PROFILE, PNG_IMAGE, TIFF_IMAGE


def pytest_sessionfinish(session, exitstatus):
    # Teardown code: Delete files created during tests.
    files_to_delete = [ICC_PROFILE, PNG_IMAGE, TIFF_IMAGE]
    for file_path in files_to_delete:
        Path(file_path).unlink(missing_ok=True)
