"""GUI of sports betting."""

import reflex as rx
from fastapi.responses import FileResponse

from .dataloader_creation import dataloader_creation_page
from .dataloader_loading import dataloader_loading_page


async def dataloader() -> FileResponse:
    """Dataloader endpoint."""
    return FileResponse(filename='dataloader.pkl', path='dataloader.pkl', media_type='application/octet-stream')


app = rx.App()
app.api.add_api_route("/dataloader", dataloader)
app.api.add_api_route("/dataloader/creation", dataloader_creation_page)
app.api.add_api_route("/dataloader/loading", dataloader_loading_page)
