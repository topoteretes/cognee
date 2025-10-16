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
            raise  # TODO: Temporarily set this to default to None so that I don't update other loaders unnecessarily yet, see TODO in LoaderEngine.py
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
        try:
            from cognee.context_global_variables import tavily_config, soup_crawler_config
            from cognee.tasks.web_scraper import fetch_page_content

            tavily = tavily_config.get()
            soup_crawler = soup_crawler_config.get()
            preferred_tool = "beautifulsoup" if soup_crawler else "tavily"
            if preferred_tool == "tavily" and tavily is None:
                raise IngestionError(
                    message="TavilyConfig must be set on the ingestion context when fetching HTTP URLs without a SoupCrawlerConfig."
                )
            if preferred_tool == "beautifulsoup" and soup_crawler is None:
                raise IngestionError(
                    message="SoupCrawlerConfig must be set on the ingestion context when using the BeautifulSoup scraper."
                )

            data = await fetch_page_content(
                file_path,
                preferred_tool=preferred_tool,
                tavily_config=tavily,
                soup_crawler_config=soup_crawler,
            )
            content = ""
            for key, value in data.items():
                content += f"{key}:\n{value}\n\n"
            await save_data_to_file(content)

            return content
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(
                message=f"Error ingesting webpage results of url {file_path}: {str(e)}"
            )
        raise NotImplementedError
