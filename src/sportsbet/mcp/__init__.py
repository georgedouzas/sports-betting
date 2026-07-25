"""Drive the library from an MCP server."""

from __future__ import annotations

from ._server import run, server

__all__: list[str] = ['run', 'server']
