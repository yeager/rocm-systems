# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Retain validated selections when their combined probe branches cannot reach."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

ENTRIES = ('cap_a', 'cap_b')
RANGE_REASON = 'unsupported_control_transfer_range'


def symbols(image):
    """Read this fixture's ELF64 symbol bytes independently of preparation facts."""
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', image)
    assert header[0][:6] == b'\x7fELF\x02\x01'
    assert header[11] == 64
    sections = [
        struct.unpack_from('<IIQQQQIIQQ', image, header[6] + index * header[11])
        for index in range(header[12])
    ]
    result = {}
    for section in sections:
        if section[1] not in (2, 11):
            continue
        assert section[9] == 24 and section[5] % section[9] == 0
        strings = sections[section[6]]
        names = image[strings[4] : strings[4] + strings[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, _info, _other, owner, value, size = struct.unpack_from(
                '<IBBHQQ', image, offset
            )
            if owner == 0 or owner >= len(sections):
                continue
            end = names.index(b'\0', name)
            label = names[name:end].decode()
            if label not in (*ENTRIES, *(entry + '.kd' for entry in ENTRIES)):
                continue
            address = sections[owner]
            assert value >= address[3] and value - address[3] + size <= address[5]
            begin = address[4] + value - address[3]
            entry = {'offset': begin, 'bytes': image[begin : begin + size]}
            assert entry['bytes'] and len(entry['bytes']) == size
            assert label not in result or result[label] == entry
            result[label] = entry
    assert set(result) == {*ENTRIES, *(entry + '.kd' for entry in ENTRIES)}
    return result


def descriptor_resources(descriptor):
    assert len(descriptor) == 64
    # The entry address is relocated; all native ABI/resource bytes must match.
    return descriptor[:16] + bytes(8) + descriptor[24:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepare', 'fixture', 'root'):
        parser.add_argument(
            '--' + name, type=lambda value: Path(value).resolve(), required=True
        )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='prepare-composition-', dir=args.root))
    print(f'evidence: {directory}', flush=True)
    original = args.fixture.read_bytes()
    original_symbols = symbols(original)
    original_digest = hashlib.sha256(original).hexdigest()
    commands = []

    def run(name, entries=(), *, require_all=False, expected=0):
        destination = directory / name
        command = [str(args.prepare), str(args.fixture), 'gfx1250', str(destination)]
        for entry in entries:
            command.extend(('--entry', entry))
        if require_all:
            command.append('--require-all')
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        commands.append(
            {
                'stage': name,
                'command': command,
                'exit': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
        )
        (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        assert result.returncode == expected, commands[-1]
        if expected:
            assert not destination.exists(), commands[-1]
            return None
        report = json.loads((destination / 'preparation.json').read_text())
        assert report['schema_version'] == 1 and report['target'] == 1250, report
        assert report['original_sha256'] == original_digest, report
        assert report['original_bytes'] == len(original), report
        assert report['outcome'] == 'instrumented', report
        assert (destination / 'ready').is_file()
        assert (destination / 'facts.bin').is_file()
        assert (destination / 'prepared.hsaco').is_file()
        return report

    def check_image(name, selected, native):
        prepared = symbols((directory / name / 'prepared.hsaco').read_bytes())
        native_body = original_symbols[native]['bytes']
        # Appended caves may extend the final function's ELF symbol. Its owning
        # original body still ends at the fixture's explicit s_endpgm boundary.
        assert len(prepared[native]['bytes']) >= len(native_body)
        assert prepared[native]['bytes'][: len(native_body)] == native_body
        assert descriptor_resources(
            prepared[native + '.kd']['bytes']
        ) == descriptor_resources(original_symbols[native + '.kd']['bytes'])
        assert prepared[selected]['bytes'] != original_symbols[selected]['bytes']
        assert struct.unpack_from('<I', prepared[selected + '.kd']['bytes'], 8)[0] > 16

    for selected, native in (ENTRIES, tuple(reversed(ENTRIES))):
        name = 'single-' + selected
        report = run(name, (selected,))
        assert (report['requested'], report['instrumented'], report['skipped']) == (
            1,
            1,
            0,
        )
        assert report['requested_coverage'] == 'complete', report
        check_image(name, selected, native)

    # Automatic selection follows descriptor order. Reverse explicit order also
    # proves that either candidate can survive as the independently valid subset.
    for name, order in (('automatic', ENTRIES), ('reverse', tuple(reversed(ENTRIES)))):
        report = run(name, () if name == 'automatic' else order)
        assert (report['requested'], report['instrumented'], report['skipped']) == (
            2,
            1,
            1,
        )
        assert report['requested_coverage'] == 'partial', report
        assert [entry['entry'] for entry in report['entries']] == list(order), report
        selected, skipped = report['entries']
        assert selected['disposition'] == 'instrumented', report
        assert skipped['disposition'] == 'native_uncovered', report
        assert skipped['reason_code'] == RANGE_REASON, skipped
        assert isinstance(skipped['reason'], str) and skipped['reason'], skipped
        for entry in report['entries']:
            assert (
                entry['entry_key'] == original_symbols[entry['entry'] + '.kd']['offset']
            )
        check_image(name, *order)
        for artifact in ('prepared.hsaco', 'facts.bin', 'ready'):
            assert (directory / name / artifact).read_bytes() == (
                directory / ('single-' + order[0]) / artifact
            ).read_bytes(), artifact

    run('strict', require_all=True, expected=2)
    assert args.fixture.read_bytes() == original
    (directory / 'result.json').write_text(
        json.dumps(
            {
                'passed': True,
                'original_sha256': original_digest,
                'typed_skip_reason': RANGE_REASON,
                'selected_and_native_orders': [list(ENTRIES), list(reversed(ENTRIES))],
                'strict_published': False,
            },
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
