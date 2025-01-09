"""
Command-line interface
"""

# # Do not use this here, it breaks typer's annotations
# from __future__ import annotations
from typing import Annotated, Optional

import typer

import gcages

app = typer.Typer()


def version_callback(version: Optional[bool]) -> None:
    """
    If requested, print the version string and exit
    """
    if version:
        print(f"gcages {gcages.__version__}")
        raise typer.Exit(code=0)


@app.callback()
def cli(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            help="Print the version number and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """
    Entrypoint for the command-line interface
    """


@app.command(name="say-hi")
def say_hi_command(
    person: Annotated[
        str,
        typer.Argument(help="The person to greet"),
    ],
) -> None:
    """
    Say hi to someone

    This is just an example command,
    you will probably delete this early in your project.
    """
    print(f"Hi {person}")


if __name__ == "__main__":
    app()
