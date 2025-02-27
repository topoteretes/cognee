from jinja2 import Environment, FileSystemLoader, select_autoescape
from cognee.root_dir import get_absolute_path


def render_prompt(filename: str, context: dict, base_directory: str = None) -> str:
    """Render a Jinja2 template asynchronously.
    :param filename: The name of the template file to render.
    :param context: The context to render the template with.
    :return: The rendered template as a string."""

    # Set the base directory relative to the cognee root directory
    if base_directory is None:
        base_directory = get_absolute_path("./infrastructure/llm/prompts")

    # Initialize the Jinja2 environment to load templates from the filesystem
    env = Environment(
        loader=FileSystemLoader(base_directory),
        autoescape=select_autoescape(["html", "xml", "txt"]),
    )

    # Load the template by name
    template = env.get_template(filename)

    # Render the template with the provided context
    rendered_template = template.render(context)

    return rendered_template
