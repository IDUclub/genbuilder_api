import os
import shutil
import tempfile

from fastapi import HTTPException

from app.dependencies import config


def get_log_file_path() -> str:
    return f"{config.get('LOG_FILE')}.log"


def create_temp_log_copy() -> tuple[str, str]:
    log_file_path = get_log_file_path()
    if not os.path.exists(log_file_path):
        raise HTTPException(
            status_code=404, detail=f"Log file not found: {log_file_path}"
        )
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        shutil.copy(log_file_path, temp_file.name)
        return temp_file.name, os.path.basename(log_file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during copying log file: {log_file_path}, error: {str(e)}",
        )


def clear_log_file() -> dict:
    log_file_path = get_log_file_path()
    if not os.path.exists(log_file_path):
        raise HTTPException(
            status_code=404, detail=f"Log file not found: {log_file_path}"
        )
    try:
        with open(log_file_path, "w") as f:
            f.truncate(0)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear log file: {str(e)}"
        )
    return {"message": "Log file cleared"}


def cleanup_temp_file(file_path: str):
    try:
        os.remove(file_path)
    except Exception:
        pass
