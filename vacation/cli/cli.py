import sys

import click

from . import dataset


# Structure of the entry_point group and adding of the subcommands
# taken from https://stackoverflow.com/a/39228156
@click.group()
def entry_point(**kwargs):
    pass


entry_point.add_command(dataset.command)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        entry_point.main(["--help"])
    entry_point()
