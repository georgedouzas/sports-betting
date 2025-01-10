"""Generate the API pages and navigation."""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.nav.Nav()

paths = [path for path in sorted(Path('src').rglob('*.py')) if 'cli' not in str(path) and 'gui' not in str(path)]
paths = [
    path
    for path in paths
    if not any(part.startswith('_') for part in path.parts[:-1])
    and (path.parts[-1] == '__init__.py' or not path.parts[-1].startswith('_'))
]
for path in paths:
    module_path = path.relative_to('src').with_suffix('')
    doc_path = path.relative_to('src', 'sportsbet').with_suffix('.md')
    full_doc_path = Path('api', doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == '__init__':
        parts = parts[:-1]
        doc_path = doc_path.with_name('index.md')
        full_doc_path = full_doc_path.with_name('index.md')

    ident = '.'.join(parts)
    nav[parts] = doc_path.as_posix()
    with mkdocs_gen_files.open(full_doc_path, 'w') as fd:
        fd.write(f'::: {ident}')
    mkdocs_gen_files.set_edit_path(full_doc_path, Path('../') / path)
