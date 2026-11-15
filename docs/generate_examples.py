"""Generate the example pages and their navigation from the example scripts.

Each script in `docs/examples` becomes a page. Its docstring gives the title and the introduction, and each `# %%` cell
gives a block of text and a block of code. markdown-exec runs the code at build time, in one session per page.
"""

import re
from pathlib import Path

import mkdocs_gen_files

EXAMPLES = Path('docs', 'examples')
TARGET = Path('generated', 'examples')
SECTIONS = ['sources', 'dataloaders', 'modelling']
SOURCE_URL = 'https://github.com/georgedouzas/sports-betting/blob/main/docs/examples'
UNDERLINE = re.compile(r'^(=+|-+)$')
FENCE = '```'


def _convert_headings(lines: list[str]) -> list[str]:
    """Turn underlined headings into Markdown headings."""
    converted: list[str] = []
    for line in lines:
        match = UNDERLINE.match(line.strip())
        if match and converted and converted[-1].strip():
            level = '#' if match.group(1).startswith('=') else '##'
            converted[-1] = f'{level} {converted[-1].strip()}'
        else:
            converted.append(line)
    return converted


def _split_script(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a script into its docstring and its cells of text and code."""
    docstring, _, body = text.lstrip().removeprefix('"""').partition('"""')
    chunks = re.split(r'^# %%[^\n]*\n', body, flags=re.MULTILINE)
    cells = []
    for index, chunk in enumerate(chunks):
        lines = chunk.splitlines()
        if index == 0:
            lines = [line for line in lines if not line.startswith(('# Author', '# Licence'))]
        text_lines = []
        while lines and lines[0].startswith('#'):
            text_lines.append(lines.pop(0).removeprefix('#').removeprefix(' '))
        cells.append(('\n'.join(_convert_headings(text_lines)).strip(), '\n'.join(lines).strip()))
    return docstring.strip(), cells


def _write_page(script: Path) -> str:
    """Write the page of one script and return its title."""
    docstring, cells = _split_script(script.read_text())
    intro = _convert_headings(docstring.splitlines())
    title = intro[0].removeprefix('# ')
    session = f'{script.parent.name}-{script.stem}'
    blocks = [
        '\n'.join(intro),
        f'{FENCE}python exec="1" session="{session}"\n'
        "import sys\nsys.path.insert(0, 'docs')\nfrom example_runner import run\n"
        f'{FENCE}',
    ]
    for text, code in cells:
        if text:
            blocks.append(text)
        if code:
            blocks.append(f'{FENCE}python\n{code}\n{FENCE}')
            run_code = f'print(run({code!r}, globals()))'
            blocks.append(f'{FENCE}python exec="1" session="{session}" html="1"\n{run_code}\n{FENCE}')
    relative = script.relative_to(EXAMPLES)
    blocks.append(f'The source of this example is [{relative.name}]({SOURCE_URL}/{relative.as_posix()}).')
    page = TARGET / relative.with_suffix('.md')
    with mkdocs_gen_files.open(page, 'w') as file:
        file.write('\n\n'.join(blocks) + '\n')
    mkdocs_gen_files.set_edit_path(page, Path('..') / script)
    return title


def _read_heading(readme: Path) -> tuple[str, str]:
    """Return the heading of a README and the text after it."""
    heading, _, rest = readme.read_text().partition('\n')
    return heading.removeprefix('# ').strip(), rest.strip()


def main() -> None:
    """Write every example page, the index and the navigation."""
    title, intro = _read_heading(EXAMPLES / 'README.md')
    index = [f'# {title}', intro]
    summary = ['* [Overview](index.md)']
    for section in SECTIONS:
        heading, text = _read_heading(EXAMPLES / section / 'README.md')
        index += [f'## {heading}', text]
        summary.append(f'* {heading}')
        for script in sorted((EXAMPLES / section).glob('plot_*.py')):
            page = f'{section}/{script.with_suffix(".md").name}'
            page_title = _write_page(script)
            index.append(f'- [{page_title}]({page})')
            summary.append(f'    * [{page_title}]({page})')
    with mkdocs_gen_files.open(TARGET / 'index.md', 'w') as file:
        file.write('\n\n'.join(index) + '\n')
    with mkdocs_gen_files.open(TARGET / 'SUMMARY.md', 'w') as file:
        file.write('\n'.join(summary) + '\n')


main()
