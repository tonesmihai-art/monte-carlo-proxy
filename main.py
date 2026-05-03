"""
Proxy FastAPI — Monte Carlo Stocks
Punct de intrare: inregistreaza toate routerele si configureaza CORS.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import proxy, sector, validate, validate_clean, iv, finnhub, heston_routes, misc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(proxy.router)
app.include_router(sector.router)
app.include_router(validate.router)
app.include_router(validate_clean.router)
app.include_router(iv.router)
app.include_router(finnhub.router)
app.include_router(heston_routes.router)
app.include_router(misc.router)
