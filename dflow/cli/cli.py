from typing import IO, List
import click
import os
from rich import print, pretty
import json

from sentence_transformers import SentenceTransformer

from dflow.language import build_model, report_model_info, merge_models
from dflow.generator import codegen as rasa_generator
from dflow.similarityCheck import similarityCheck

pretty.install()


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


@cli.command("validate", help="Model Validation")
@click.pass_context
@click.argument("model_path")
def validate(ctx, model_path):
    model = build_model(model_path)
    print("[*] Model validation success!!")
    report_model_info(model)


@cli.command("gen", help="M2T/M2M transformations")
@click.pass_context
@click.argument("model_path")
@click.argument("generator")
def generate(ctx, model_path, generator):
    if generator not in ("rasa"):
        print(f"[*] Generator {generator} not supported")
        return
    if generator == "rasa":
        out_path = rasa_generator(model_path)
    print(f"[*] M2T finished. Output: {out_path}")

@cli.command("merge", help="Merge Models")
@click.pass_context
@click.argument('models', nargs=-1, type=click.File('r'))
def merge(ctx, models):
    _models = [model.read() for model in models]
    if len(_models) < 2:
        print("[X] Number of models must be greater than two (2)")
        return

    sections = [
        'Intent',
        'responses'
    ]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    similarities = similarityCheck(_models, sections, model)
    #Intents not similar (case 2 & 4)
    if not similarities['Intent']['similar']:
         print("Add both")
    #Intents similar and responses similar (case 1)
    elif similarities['responses']['similar']:
        merged_model_str = merge_models(_models)
        out_path = f"merged.dflow"
        with open(out_path, 'w') as f:
                    f.write(merged_model_str)
        print(f"[*] Model merging finished - Output: {out_path}")
    #Intents similar and responses not similar (case 3)
    else:
         print('Warning: Intents similar and responses not similar')

def main():
    cli(prog_name="dflow")
