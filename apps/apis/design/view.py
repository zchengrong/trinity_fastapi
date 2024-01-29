import time

from apps.apis.routers import router


@router.get("/design")
async def design():
    result = time.time()
    return {"result": result}
