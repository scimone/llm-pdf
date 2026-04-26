import logging
from llm_pdf import ocr_processor
import click
from dotenv import load_dotenv


load_dotenv(".env", override=True)
logger = logging.getLogger(__name__)
# setup_logging()


@click.group()
def cli():
    """
    Main CLI
    """

    pass


@cli.command(name="ocr")
@click.option(
    "--file_name", required=True, type=str, help="The name of the file to upload"
)
def ocr(file_name):
    ocr_processor.main(file_name)
