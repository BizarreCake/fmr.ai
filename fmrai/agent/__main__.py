from typing import Optional

import typer

from fmrai import fmrai
from fmrai.agent.agents.transformers import TransformersAgentAPI

run_app = typer.Typer()


app = typer.Typer()
app.add_typer(run_app, name='run')


@run_app.command(name='text')
def run_text(
        model: str,
        host: Optional[str] = typer.Option('127.0.0.1', '--host', '-h'),
        port: Optional[int] = typer.Option(8001, '--port', '-p'),
        cpu: bool = typer.Option(False, '--cpu', '-c'),
):
    with fmrai():
        try:
            api = TransformersAgentAPI(model, use_cuda=not cpu)
        except OSError:
            print('error: invalid model name:', model)
            return

        api.run(host=host, port=port)


app()
