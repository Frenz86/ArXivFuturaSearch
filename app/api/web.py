"""Web interface routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root_web(request: Request):
    """Serve the web interface."""
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")

    # Get health data for the template
    from app.api.monitoring import health_check
    health_data = await health_check()

    return templates.TemplateResponse("index.html", {"request": request, "health": health_data})


@router.get("/api")
async def api_info():
    """Root API endpoint with basic info."""
    return {
        "name": "ArXiv Futura Search",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/web/search", response_class=HTMLResponse)
async def web_search(request: Request):
    """Search page."""
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")

    # Get health data for the template
    from app.api.monitoring import health_check
    health_data = await health_check()

    return templates.TemplateResponse("index.html", {"request": request, "health": health_data})


@router.get("/web/build", response_class=HTMLResponse)
async def web_build(request: Request):
    """Build index page."""
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")

    # Get health data for the template
    from app.api.monitoring import health_check
    health_data = await health_check()

    return templates.TemplateResponse("build.html", {"request": request, "health": health_data})


@router.get("/web/index", response_class=HTMLResponse)
async def web_index_view(request: Request):
    """View indexed papers page."""
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    return templates.TemplateResponse("index_view.html", {"request": request})


@router.get("/web/config", response_class=HTMLResponse)
async def web_config_view(request: Request):
    """View configuration page."""
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    return templates.TemplateResponse("config_view.html", {"request": request})
