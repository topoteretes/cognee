from cognee.infrastructure.loaders import LoaderInterface
from typing import List

from cognee.modules.ingestion.exceptions.exceptions import IngestionError
from cognee.modules.ingestion import save_data_to_file


class WebUrlLoader(LoaderInterface):
    @property
    def supported_extensions(self) -> List[str]:
        """
        List of file extensions this loader supports.

        Returns:
            List of extensions including the dot (e.g., ['.txt', '.md'])
        """
        return []  # N/A, we can safely return empty since it's used in register and get_loader_info, doesn't reflect on functionality

    @property
    def supported_mime_types(self) -> List[str]:
        """
        List of MIME types this loader supports.

        Returns:
            List of MIME type strings (e.g., ['text/plain', 'application/pdf'])
        """
        return []  # N/A, we can safely return empty since it's used in register and get_loader_info, doesn't reflect on functionality

    @property
    def loader_name(self) -> str:
        """
        Unique name identifier for this loader.

        Returns:
            String identifier used for registration and configuration
        """
        return "web_url_loader"

    def can_handle(self, extension: str, mime_type: str, data_item_path: str = None) -> bool:
        """
        Check if this loader can handle the given file.

        Args:
            extension: File extension
            mime_type: MIME type of the file

        Returns:
            True if this loader can process the file, False otherwise
        """
        if data_item_path is None:
            raise IngestionError(
                "data_item_path should not be None"
            )  # TODO: Temporarily set this to default to None so that I don't update other loaders unnecessarily yet, see TODO in LoaderEngine.py
        return data_item_path.startswith(("http://", "https://"))

    async def load(self, file_path: str, **kwargs):
        """
        Load and process the file, returning standardized result.

        Args:
            file_path: Path to the file to be processed
            file_stream: If file stream is provided it will be used to process file instead
            **kwargs: Additional loader-specific configuration

        Raises:
            Exception: If file cannot be processed
        """
        loaders_config = kwargs.get("loaders_config")
        if not isinstance(loaders_config, dict):
            raise IngestionError("loaders_config must be a valid dictionary")

        web_url_loader_config = loaders_config.get(self.loader_name)
        if not isinstance(web_url_loader_config, dict):
            raise IngestionError(f"{self.loader_name} configuration must be a valid dictionary")

        try:
            from cognee.context_global_variables import tavily_config, soup_crawler_config
            from cognee.tasks.web_scraper import fetch_page_content

            _tavily_config = web_url_loader_config.get("tavily_config")
            _soup_config = web_url_loader_config.get("soup_config")

            # Set global configs for downstream access
            tavily_config.set(_tavily_config)
            soup_crawler_config.set(_soup_config)

            preferred_tool = "beautifulsoup" if _soup_config else "tavily"
            if preferred_tool == "tavily" and _tavily_config is None:
                raise IngestionError(
                    message="TavilyConfig must be set on the ingestion context when fetching HTTP URLs without a SoupCrawlerConfig."
                )
            if preferred_tool == "beautifulsoup" and _soup_config is None:
                raise IngestionError(
                    message="SoupCrawlerConfig must be set on the ingestion context when using the BeautifulSoup scraper."
                )

            data = await fetch_page_content(
                file_path,
                preferred_tool=preferred_tool,
                tavily_config=_tavily_config,
                soup_crawler_config=_soup_config,
            )
            content = ""
            for key, value in data.items():
                content += f"{key}:\n{value}\n\n"
            await save_data_to_file(content)

            return content
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(message=f"Error ingesting webpage from URL {file_path}: {str(e)}")
