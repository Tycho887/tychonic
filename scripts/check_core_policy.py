#!/usr/bin/env python3
"""Check Tychonic-owned production source; dependencies are audited separately."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
# Satkit requires owned settings and mutable TLEs. These functions are the only
# accepted source-level clone sites; see docs/safety.md for their contracts.
DEPENDENCY_COPIES = {
    ('src/kernels/stm.rs', 'sample_arc'),
    ('src/kernels/stm.rs', 'initial_state_from_tle'),
    ('src/estimate/tca_ukf.rs', 'shifted_tle'),
    ('src/measurements/rf.rs', 'build'),
    ('src/measurements/rf.rs', 'filter'),
}
TEXT = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"')
BANNED = re.compile(
    r'\bunsafe\b|\b(?:Arc|Rc|RefCell|Mutex|RwLock)\b'
    r'|\.\s*(?:unwrap|expect|unwrap_unchecked|cloned)\s*\('
    r'|\b(?:panic|unreachable|todo|unimplemented|assert|assert_eq|debug_assert|debug_assert_eq)!'
    r'|\bthread\s*::'
)


def mask_text(source):
    return TEXT.sub(lambda match: ''.join('\n' if c == '\n' else ' ' for c in match[0]), source)


def function_ranges(source):
    ranges = []
    for match in re.finditer(r'\bfn\s+(\w+)', source):
        start = source.find('{', match.end())
        if start < 0:
            continue
        depth, end = 1, start + 1
        while depth and end < len(source):
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        ranges.append((start, end, match[1]))
    return ranges


def main():
    failures = []
    for path in sorted((ROOT / 'src').rglob('*.rs')):
        relative = path.relative_to(ROOT).as_posix()
        # Clippy also requires test modules to follow all production items.
        source = mask_text(path.read_text().split('#[cfg(test)]', 1)[0])
        for match in BANNED.finditer(source):
            failures.append(f'{relative}:{source.count(chr(10), 0, match.start()) + 1}: {match[0]}')
        ranges = function_ranges(source)
        for match in re.finditer(r'\.\s*clone\s*\(', source):
            owners = [(end - start, name) for start, end, name in ranges if start <= match.start() < end]
            function = min(owners)[1] if owners else ''
            if (relative, function) not in DEPENDENCY_COPIES:
                failures.append(f'{relative}:{source.count(chr(10), 0, match.start()) + 1}: undocumented clone')
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print('Core policy passed: safe Rust, sequential execution, checked failures, documented dependency copies.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
