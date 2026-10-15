"""List or restore selected regular files from the pinned server backups.

Requires Python >=3.9 and zstd on PATH. Symlinks require the full Linux restore
procedure in docs/storage.md so that absolute server links keep their meaning.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or '\\' in name or any(':' in part for part in path.parts):
        raise ValueError('Unsafe relative path: ' + name)
    return path

def target_path(destination: Path, name: str) -> Path:
    path = destination.joinpath(*safe_relative(name).parts)
    if destination != path.resolve() and destination not in path.resolve().parents:
        raise ValueError('Destination escapes restore root: ' + name)
    # Existing symlinks/junctions must not redirect writes, even inside the root.
    current = destination
    for part in safe_relative(name).parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('Existing link in destination: ' + name)
    return path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def restore_archive(archive: Path, selected: dict, destination: Path) -> list:
    remaining = dict(selected)
    restored = []
    proc = subprocess.Popen(['zstd', '-dc', str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|', bufsize=1024**2) as stream:
            for member in stream:
                name = member.name.removeprefix('./')
                if not name.startswith('cjk/'):
                    continue
                relative = name[len('cjk/'):]
                if relative not in remaining:
                    continue
                row = remaining[relative]
                if not member.isfile() or member.size != int(row['size_bytes']):
                    raise ValueError('Archive entry differs from manifest: ' + relative)
                target = target_path(destination, relative)
                temporary = target.with_name(target.name + '.partial')
                if target.exists() or temporary.exists():
                    raise FileExistsError('Refusing to overwrite: ' + str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                created = False
                try:
                    with stream.extractfile(member) as source, temporary.open('xb') as output:
                        created = True
                        while True:
                            block = source.read(8 * 1024**2)
                            if not block:
                                break
                            output.write(block)
                            digest.update(block)
                            written += len(block)
                    if written != member.size:
                        raise ValueError('Truncated archive member: ' + relative)
                    temporary.rename(target)
                except BaseException:
                    if created:
                        temporary.unlink(missing_ok=True)
                    raise
                restored.append({'path': relative, 'size_bytes': written, 'sha256': digest.hexdigest(), 'archive': archive.name})
                del remaining[relative]
                print('Restored ' + relative, flush=True)
                if not remaining:
                    break
        if remaining:
            raise RuntimeError('Selected files missing from archive: ' + ', '.join(remaining))
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        proc.wait()
    return restored

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include', action='append', required=True, help='Repository-relative glob; repeat for more selections. Quote * in the shell.')
    parser.add_argument('--list', action='store_true', help='List matching entries without reading archives or writing files.')
    parser.add_argument('--backup-dir', type=Path)
    parser.add_argument('--destination', type=Path, help='Restore root, corresponding to the original /data/cjk.')
    parser.add_argument('--verify', action='store_true', help='Verify complete compressed-archive SHA-256 values before extraction.')
    args = parser.parse_args()
    with (ROOT / 'assets' / 'backup_manifest.tsv').open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source, delimiter='\t'))
    selected = [row for row in rows if any(fnmatch.fnmatchcase(row['path'], pattern) for pattern in args.include)]
    if not selected:
        parser.error('No entries match the requested pattern(s).')
    for row in selected:
        safe_relative(row['path'])
    size = sum(int(row['size_bytes']) for row in selected)
    print(f'Selected {len(selected)} entries, {size / 1024**3:.3f} GiB.')
    if args.list:
        for row in selected:
            print('\t'.join(row[key] for key in ['kind', 'size_bytes', 'archive', 'path', 'link_target']))
        return
    if not args.backup_dir or not args.destination:
        parser.error('--backup-dir and --destination are required when restoring.')
    if any(row['kind'] != 'file' for row in selected):
        parser.error('Selection contains links. Narrow the selection to regular files, or use the full Linux restore in docs/storage.md.')
    if not shutil.which('zstd'):
        parser.error('zstd is required on PATH.')
    destination = args.destination.resolve()
    archive_config = json.loads((ROOT / 'assets' / 'archives.json').read_text(encoding='utf-8'))
    grouped = {}
    for row in selected:
        target = target_path(destination, row['path'])
        if target.exists() or target.with_name(target.name + '.partial').exists():
            raise FileExistsError('Refusing to overwrite: ' + str(target))
        grouped.setdefault(row['archive'], {})[row['path']] = row
    known = {item['file']: item for item in archive_config['archives']}
    for archive_name in grouped:
        if archive_name not in known:
            raise ValueError('Unknown archive in manifest: ' + archive_name)
        archive = args.backup_dir / archive_name
        if archive.stat().st_size != known[archive_name]['size_bytes']:
            raise ValueError('Archive size mismatch: ' + archive_name)
        if args.verify:
            print('Verifying ' + archive_name, flush=True)
            if sha256_file(archive) != known[archive_name]['sha256']:
                raise ValueError('Archive SHA-256 mismatch: ' + archive_name)
    report = []
    for archive_name in archive_config['restore_order']:
        if archive_name in grouped:
            report.extend(restore_archive(args.backup_dir / archive_name, grouped[archive_name], destination))
    destination.mkdir(parents=True, exist_ok=True)
    report_path = target_path(destination, 'asset_restore_report.json')
    # Append reports without overwriting an earlier record of restored files.
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8')) + report
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Per-file checksums: ' + str(report_path))

if __name__ == '__main__':
    main()
