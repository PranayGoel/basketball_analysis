"""FastAPI app factory: CORS config + router mounting."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import jobs, reports, search, videos
from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title="Basketball Analysis API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(videos.router)
    app.include_router(jobs.router)
    app.include_router(reports.router)
    app.include_router(search.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
