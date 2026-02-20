from fastapi import APIRouter
from fastapi.responses import JSONResponse

from cognee.api.v1.health import health_checker, HealthStatus


def get_health_router():
    health_router = APIRouter()

    @health_router.get("", response_model=dict)
    async def health_check():
        """
        Health check endpoint for liveness/readiness probes.
        """
        try:
            health_status = await health_checker.get_health_status(detailed=False)
            status_code = 503 if health_status.status == HealthStatus.UNHEALTHY else 200

            return JSONResponse(
                status_code=status_code,
                content={
                    "status": "ready" if status_code == 200 else "not ready",
                    "health": health_status.status,
                    "version": health_status.version,
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "reason": f"health check failed: {str(e)}"},
            )

    @health_router.get("/detailed", response_model=dict)
    async def detailed_health_check():
        """
        Comprehensive health status with component details.
        """
        try:
            health_status = await health_checker.get_health_status(detailed=True)
            status_code = 200
            if health_status.status == HealthStatus.UNHEALTHY:
                status_code = 503
            elif health_status.status == HealthStatus.DEGRADED:
                status_code = 503

            return JSONResponse(status_code=status_code, content=health_status.model_dump())
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": f"Health check system failure: {str(e)}"},
            )

    return health_router
