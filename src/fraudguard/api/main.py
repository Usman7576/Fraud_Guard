"""ASGI entry point used by `fraudguard serve` (uvicorn fraudguard.api.main:app)."""

from fraudguard.api.app import create_app
from fraudguard.config import get_settings
from fraudguard.log import setup_logging

setup_logging(get_settings().log_level)
app = create_app(get_settings())
