import aiohttp

from cognee.modules.cloud.exceptions import CloudConnectionError
from cognee.shared.utils import create_secure_ssl_context


async def check_api_key(auth_token: str):
    cloud_base_url = "http://localhost:8001"

    url = f"{cloud_base_url}/api/api-keys/check"
    headers = {"X-Api-Key": auth_token}

    try:
        ssl_context = create_secure_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
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
