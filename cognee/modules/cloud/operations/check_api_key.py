import aiohttp

from cognee.modules.cloud.exceptions import CloudConnectionError


async def check_api_key(auth_token: str):
    cloud_base_url = "http://localhost:8001"

    url = f"{cloud_base_url}/api/api-keys/check"
    headers = {"X-Api-Key": auth_token}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers) as response:
                if response.status == 200:
                    return
                else:
                    error_text = await response.text()

                    raise CloudConnectionError(
                        f"Failed to connect to cloud instance: {response.status} - {error_text}"
                    )

    except Exception as e:
        raise CloudConnectionError(f"Failed to connect to cloud instance: {str(e)}")
