"""GUI of sports betting."""

import reflex as rx
from fastapi.responses import FileResponse

from .pages.dataloader.creation import dataloader_creation
from .pages.dataloader.loading import dataloader_loading


async def dataloader() -> FileResponse:
    """Dataloader endpoint."""
    return FileResponse(filename='dataloader.pkl', path='dataloader.pkl', media_type='application/octet-stream')


app = rx.App()
app.api.add_api_route("/dataloader", dataloader)
app.api.add_api_route("/dataloader/creation", dataloader_creation)
app.api.add_api_route("/dataloader/loading", dataloader_loading)
