import os
import pytest
import cognee
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.loaders.LoaderEngine import LoaderEngine
from cognee.infrastructure.loaders.external.beautiful_soup_loader import BeautifulSoupLoader
from cognee.tasks.ingestion import save_data_item_to_storage
from pathlib import Path


@pytest.mark.asyncio
async def test_url_saves_as_html_file():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    try:
        original_file_path = await save_data_item_to_storage("http://example.com/")
        file_path = get_data_file_path(original_file_path)
        assert file_path.endswith(".html")
        file = Path(file_path)
        assert file.exists()
        assert file.stat().st_size > 0
    except Exception as e:
        pytest.fail(f"Failed to save data item to storage: {e}")


skip_for_tavily = pytest.mark.skipif(
    os.getenv("TAVILY_API_KEY") is not None,
    reason="Skipping as Tavily already handles parsing and outputs text",
)


@skip_for_tavily
@pytest.mark.asyncio
async def test_saved_html_is_valid():
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        pytest.fail("Test case requires bs4 installed")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    try:
        original_file_path = await save_data_item_to_storage("http://example.com/")
        file_path = get_data_file_path(original_file_path)
        content = Path(file_path).read_text()

        soup = BeautifulSoup(content, "html.parser")
        assert soup.find() is not None, "File should contain parseable HTML"

        has_html_elements = any(
            [
                soup.find("html"),
                soup.find("head"),
                soup.find("body"),
                soup.find("div"),
                soup.find("p"),
            ]
        )
        assert has_html_elements, "File should contain common HTML elements"
    except Exception as e:
        pytest.fail(f"Failed to save data item to storage: {e}")


@pytest.mark.asyncio
async def test_add_url():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add("http://example.com/")


skip_in_ci = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping in Github for now - before we get TAVILY_API_KEY",
)


@skip_in_ci
@pytest.mark.asyncio
async def test_add_url_with_tavily():
    assert os.getenv("TAVILY_API_KEY") is not None
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add("http://example.com/")


@pytest.mark.asyncio
async def test_add_url_without_incremental_loading():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    try:
        await cognee.add(
            "http://example.com/",
            incremental_loading=False,
        )
    except Exception as e:
        pytest.fail(f"Failed to add url: {e}")


@pytest.mark.asyncio
async def test_add_url_with_incremental_loading():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    try:
        await cognee.add(
            "http://example.com/",
            incremental_loading=True,
        )
    except Exception as e:
        pytest.fail(f"Failed to add url: {e}")


@pytest.mark.asyncio
async def test_add_url_can_define_preferred_loader_as_list_of_str():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(
        "http://example.com/",
        preferred_loaders=["beautiful_soup_loader"],
    )


@pytest.mark.asyncio
async def test_add_url_with_extraction_rules():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    extraction_rules = {
        "title": {"selector": "title"},
        "headings": {"selector": "h1, h2, h3", "all": True},
        "links": {"selector": "a", "attr": "href", "all": True},
        "paragraphs": {"selector": "p", "all": True},
    }

    try:
        await cognee.add(
            "http://example.com/",
            preferred_loaders={"beautiful_soup_loader": {"extraction_rules": extraction_rules}},
        )
    except Exception as e:
        pytest.fail(f"Failed to add url: {e}")


@pytest.mark.asyncio
async def test_loader_is_none_by_default():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    extraction_rules = {
        "title": {"selector": "title"},
        "headings": {"selector": "h1, h2, h3", "all": True},
        "links": {"selector": "a", "attr": "href", "all": True},
        "paragraphs": {"selector": "p", "all": True},
    }

    try:
        original_file_path = await save_data_item_to_storage("http://example.com/")
        file_path = get_data_file_path(original_file_path)
        assert file_path.endswith(".html")
        file = Path(file_path)
        assert file.exists()
        assert file.stat().st_size > 0

        loader_engine = LoaderEngine()
        preferred_loaders = {"beautiful_soup_loader": {"extraction_rules": extraction_rules}}
        loader = loader_engine.get_loader(
            file_path,
            preferred_loaders=preferred_loaders,
        )

        assert loader is None
    except Exception as e:
        pytest.fail(f"Failed to save data item to storage: {e}")


@pytest.mark.asyncio
async def test_beautiful_soup_loader_is_selected_loader_if_preferred_loader_provided():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    extraction_rules = {
        "title": {"selector": "title"},
        "headings": {"selector": "h1, h2, h3", "all": True},
        "links": {"selector": "a", "attr": "href", "all": True},
        "paragraphs": {"selector": "p", "all": True},
    }

    try:
        original_file_path = await save_data_item_to_storage("http://example.com/")
        file_path = get_data_file_path(original_file_path)
        assert file_path.endswith(".html")
        file = Path(file_path)
        assert file.exists()
        assert file.stat().st_size > 0

        loader_engine = LoaderEngine()
        bs_loader = BeautifulSoupLoader()
        loader_engine.register_loader(bs_loader)
        preferred_loaders = {"beautiful_soup_loader": {"extraction_rules": extraction_rules}}
        loader = loader_engine.get_loader(
            file_path,
            preferred_loaders=preferred_loaders,
        )

        assert loader == bs_loader
    except Exception as e:
        pytest.fail(f"Failed to save data item to storage: {e}")


@pytest.mark.asyncio
async def test_beautiful_soup_loader_works_with_and_without_arguments():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    try:
        original_file_path = await save_data_item_to_storage("http://example.com/")
        file_path = get_data_file_path(original_file_path)
        assert file_path.endswith(".html")
        file = Path(file_path)
        assert file.exists()
        assert file.stat().st_size > 0

        loader_engine = LoaderEngine()
        bs_loader = BeautifulSoupLoader()
        loader_engine.register_loader(bs_loader)
        preferred_loaders = {"beautiful_soup_loader": {}}
        await loader_engine.load_file(
            file_path,
            preferred_loaders=preferred_loaders,
        )
        extraction_rules = {
            "title": {"selector": "title"},
            "headings": {"selector": "h1, h2, h3", "all": True},
            "links": {"selector": "a", "attr": "href", "all": True},
            "paragraphs": {"selector": "p", "all": True},
        }
        preferred_loaders = {"beautiful_soup_loader": {"extraction_rules": extraction_rules}}
        await loader_engine.load_file(
            file_path,
            preferred_loaders=preferred_loaders,
        )
    except Exception as e:
        pytest.fail(f"Failed to save data item to storage: {e}")


@pytest.mark.asyncio
async def test_beautiful_soup_loader_successfully_loads_file_if_required_args_present():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    try:
        original_file_path = await save_data_item_to_storage("http://example.com/")
        file_path = get_data_file_path(original_file_path)
        assert file_path.endswith(".html")
        file = Path(file_path)
        assert file.exists()
        assert file.stat().st_size > 0

        loader_engine = LoaderEngine()
        bs_loader = BeautifulSoupLoader()
        loader_engine.register_loader(bs_loader)
        extraction_rules = {
            "title": {"selector": "title"},
            "headings": {"selector": "h1, h2, h3", "all": True},
            "links": {"selector": "a", "attr": "href", "all": True},
            "paragraphs": {"selector": "p", "all": True},
        }
        preferred_loaders = {"beautiful_soup_loader": {"extraction_rules": extraction_rules}}
        await loader_engine.load_file(
            file_path,
            preferred_loaders=preferred_loaders,
        )
    except Exception as e:
        pytest.fail(f"Failed to save data item to storage: {e}")


@pytest.mark.asyncio
async def test_beautiful_soup_loads_file_successfully():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    extraction_rules = {
        "title": {"selector": "title"},
        "headings": {"selector": "h1, h2, h3", "all": True},
        "links": {"selector": "a", "attr": "href", "all": True},
        "paragraphs": {"selector": "p", "all": True},
    }

    try:
        original_file_path = await save_data_item_to_storage("http://example.com/")
        file_path = get_data_file_path(original_file_path)
        assert file_path.endswith(".html")
        original_file = Path(file_path)
        assert original_file.exists()
        assert original_file.stat().st_size > 0

        loader_engine = LoaderEngine()
        bs_loader = BeautifulSoupLoader()
        loader_engine.register_loader(bs_loader)
        preferred_loaders = {"beautiful_soup_loader": {"extraction_rules": extraction_rules}}
        loader = loader_engine.get_loader(
            file_path,
            preferred_loaders=preferred_loaders,
        )

        assert loader == bs_loader

        cognee_loaded_txt_path = await loader_engine.load_file(
            file_path=file_path, preferred_loaders=preferred_loaders
        )

        cognee_loaded_txt_path = get_data_file_path(cognee_loaded_txt_path)

        assert cognee_loaded_txt_path.endswith(".txt")

        extracted_file = Path(cognee_loaded_txt_path)

        assert extracted_file.exists()
        assert extracted_file.stat().st_size > 0

        original_basename = original_file.stem
        extracted_basename = extracted_file.stem
        assert original_basename == extracted_basename, (
            f"Expected same base name: {original_basename} vs {extracted_basename}"
        )
    except Exception as e:
        pytest.fail(f"Failed to save data item to storage: {e}")
