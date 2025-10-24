from fastapi import APIRouter
from starlette.responses import FileResponse
from starlette.background import BackgroundTask
from app.logic.logs_logic import create_temp_log_copy, clear_log_file, cleanup_temp_file

logs_router = APIRouter()

@logs_router.get("/logs", summary="Get logs")
async def get_logs():
    temp_file, original_filename = create_temp_log_copy()
    return FileResponse(
        path=temp_file,
        media_type="application/octet-stream",
        filename=original_filename,
        background=BackgroundTask(cleanup_temp_file, temp_file)
    )

@logs_router.delete("/logs", summary="Remove logs")
async def clear_logs():
    return clear_log_file()
