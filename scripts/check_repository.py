"""Check staged files for accidentally included experiment data or credentials."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {'.git', '.codex', '.ssh', '.cache', '.venv', '__pycache__', 'data', 'dataset', 'datasets', 'raw', 'artifacts', 'runs', 'output', 'outputs', 'logs', 'checkpoints', 'wandb'}
FORBIDDEN_SUFFIXES = {'.safetensors', '.pt', '.pth', '.bin', '.ckpt', '.onnx', '.gguf', '.pkl', '.pickle', '.npy', '.npz', '.parquet', '.arrow', '.jsonl', '.log', '.sqlite', '.db', '.zip', '.gz', '.zst', '.pem', '.key', '.p12'}
PATTERNS = {
    'private key': re.compile(rb'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----'),
    'GitHub token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'Hugging Face token': re.compile(rb'hf_[A-Za-z0-9]{30,}'),
    'OpenAI-style API key': re.compile(rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}'),
    'AWS access key': re.compile(rb'(?:AKIA|ASIA)[A-Z0-9]{16}'),
    'credential in URL': re.compile(rb'https?://[^\s/:@]+:[^\s/@]+@'),
    'assigned secret': re.compile(rb'''(?im)^\s*["']?(?:api_key|api_secret|access_token|auth_token|password|aws_secret_access_key)["']?\s*[:=]\s*["']([^"'\r\n]{12,})["']'''),
}

def main() -> int:
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT).split(b'\0')
    entries = [entry for entry in entries if entry]
    oids = [entry.split(b'\t', 1)[0].split()[1] for entry in entries]
    blobs = subprocess.check_output(['git', 'cat-file', '--batch'], input=b'\n'.join(oids) + b'\n', cwd=ROOT)
    offset = 0
    issues = []
    count = total = 0
    for entry in entries:
        metadata, raw_name = entry.split(b'\t', 1)
        mode, oid, stage = metadata.decode('ascii').split()
        name = raw_name.decode('utf-8')
        path = Path(name)
        end = blobs.index(b'\n', offset)
        _, object_type, length = blobs[offset:end].split()
        length = int(length)
        content = blobs[end + 1:end + 1 + length]
        offset = end + 2 + length
        if mode not in {'100644', '100755'} or stage != '0':
            issues.append((name, 'unexpected Git entry mode/stage'))
            continue
        count += 1
        total += len(content)
        if any(part in FORBIDDEN_DIRS for part in path.parts[:-1]) or path.suffix.lower() in FORBIDDEN_SUFFIXES or (path.name.startswith('.env') and path.name != '.env.example'):
            issues.append((name, 'runtime asset or sensitive file name'))
        if len(content) > 10 * 1024**2:
            issues.append((name, 'larger than the repository 10 MiB source-file budget'))
        if b'\0' in content:
            issues.append((name, 'binary content'))
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                value = match.group(1) if label == 'assigned secret' else b''
                if value and any(s in value.lower() for s in [b'example', b'placeholder', b'your_', b'os.environ', b'getenv', b'${', b'<', b'...']):
                    continue
                line = content.count(b'\n', 0, match.start()) + 1
                issues.append((name + ':' + str(line), label + ' (value redacted)'))
    for name, problem in issues:
        print(name + ': ' + problem)
    print(f'Checked {count} staged/tracked files, {total / 1024**2:.2f} MiB; {len(issues)} issue(s).')
    return bool(issues)

if __name__ == '__main__':
    raise SystemExit(main())
