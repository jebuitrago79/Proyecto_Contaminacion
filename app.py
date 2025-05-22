# app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from modelo import run_analysis

app = FastAPI()

# Montar la carpeta static para imágenes
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurar templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    resultados = run_analysis()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "correlation_matrix": resultados["correlation_matrix"],
        "r2_scores": resultados["r2_scores"],
        "yearly_avg": resultados["yearly_avg"],
        "imagenes": {
            "heatmap": "static/heatmap_correlacion.png",
            "dispersion_co": "static/dispersion_pm25_vs_co.png",
            "dispersion_pm10": "static/dispersion_pm25_vs_pm10.png",
            "regresion_co": "static/regresion_co.png",
            "regresion_pm10": "static/regresion_pm10.png",
            "pm25_anual": "static/pm25_anual.png"
        }
    })
