#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Analyze an owning application index without original/prepared code images."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze_application(index_path, analyzer):
    index = json.loads(index_path.read_text())
    require(
        index['schema_version'] == 1
        and index['scope'] == 'provided_application_images',
        'unsupported application index',
    )
    require(
        isinstance(index['images'], list) and 0 < len(index['images']) <= 1024,
        'empty or oversized application image list',
    )
    output = {
        'scope': 'provided_application_images',
        'whole_application_coverage': 'not_established',
        'lds_analysis': 'not_performed',
        'lds_observations_present': False,
        'coverage_complete': True,
        'cross_wave_pairs': 0,
        'collision_groups': 0,
        'images': [],
    }
    identities = set()
    captures = set()
    run_identity = None
    for image in index['images']:
        report = image['preparation']
        identity = report['original_sha256']
        require(
            report['schema_version'] == 1
            and isinstance(identity, str)
            and len(identity) == 64
            and len(bytes.fromhex(identity)) == 32
            and identity not in identities,
            'invalid or duplicate original image identity',
        )
        identities.add(identity)
        entries = report['entries']
        require(
            isinstance(entries, list)
            and 0 < len(entries) <= 1024
            and report['requested'] == len(entries),
            'invalid preparation entry inventory',
        )
        names = set()
        keys = set()
        selected = {}
        for entry in entries:
            require(
                entry['entry'] not in names and entry['entry_key'] not in keys,
                'duplicate preparation entry identity',
            )
            names.add(entry['entry'])
            keys.add(entry['entry_key'])
            require(
                entry['disposition'] in ('instrumented', 'native_uncovered'),
                'unknown preparation entry disposition',
            )
            if entry['disposition'] == 'instrumented':
                selected[entry['entry_key']] = entry['entry']
        require(
            report['instrumented'] == len(selected)
            and report['skipped'] == len(entries) - len(selected),
            'preparation counts do not match owning entries',
        )
        row = {
            'original_sha256': identity,
            'preparation': report,
            'analysis': None,
            'issues': [],
            'selected_entries_not_observed': sorted(selected.values()),
        }
        output['images'].append(row)
        if not selected:
            require(
                report['outcome'] == 'native_only'
                and image['capture'] is None
                and report['requested_coverage'] == 'not_performed',
                'native-only image claims selected coverage',
            )
            row['issues'].append('native_uncovered')
            output['coverage_complete'] = False
            continue
        require(
            report['outcome'] == 'instrumented' and isinstance(image['capture'], str),
            'selected image has no capture declaration',
        )
        relative = Path(image['capture'])
        require(
            not relative.is_absolute()
            and '..' not in relative.parts
            and relative.as_posix() == image['capture']
            and relative not in captures,
            'invalid or duplicate relative capture path',
        )
        captures.add(relative)
        capture = index_path.parent / relative
        if not capture.exists():
            row['issues'].append('selected_image_has_no_closed_capture')
            output['coverage_complete'] = False
            continue
        result = subprocess.run(
            [str(analyzer), str(capture)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        require(
            result.returncode in (0, 1),
            f'invalid capture {relative}: {result.stderr.strip()}',
        )
        analysis = json.loads(result.stdout)
        require(
            analysis['original_sha256'] == identity
            and analysis['selection_profile'] == report['selection_profile']
            and analysis['target'] == report['target'],
            'capture and preparation identities differ',
        )
        if run_identity is None:
            run_identity = analysis['run_identity']
        require(
            analysis['run_identity'] == run_identity,
            'captures belong to different application runs',
        )
        captured_entries = {entry['entry_key']: entry for entry in analysis['entries']}
        require(
            len(captured_entries) == len(entries), 'capture entry inventory differs'
        )
        for entry in entries:
            captured = captured_entries.get(entry['entry_key'])
            require(
                captured is not None
                and captured['entry'] == entry['entry']
                and (captured['disposition'] == 'selected')
                == (entry['entry_key'] in selected),
                'capture entry selection differs from preparation report',
            )
        executions = analysis['executions']
        observed = {execution['entry_key'] for execution in executions}
        require(observed.issubset(selected), 'execution references an unselected entry')
        row['selected_entries_not_observed'] = sorted(
            selected[key] for key in selected.keys() - observed
        )
        row['analysis'] = analysis
        trace = analysis['scope'] == 'execution_trace'
        complete = (
            analysis['transport_complete'] if trace else analysis['coverage_complete']
        )
        if not trace:
            observations = sum(
                execution['valid_observations'] for execution in executions
            )
            if executions:
                output['lds_analysis'] = 'performed'
            output['lds_observations_present'] |= observations > 0
            output['cross_wave_pairs'] += analysis['cross_wave_pairs']
            output['collision_groups'] += analysis['collision_groups']
        if row['selected_entries_not_observed']:
            row['issues'].append('selected_entries_not_executed')
        output['coverage_complete'] &= (
            bool(complete) and not row['issues'] and report['skipped'] == 0
        )
    output['run_identity'] = run_identity
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=lambda value: Path(value).resolve())
    parser.add_argument(
        '--analyzer', type=lambda value: Path(value).resolve(), required=True
    )
    args = parser.parse_args()
    result = analyze_application(args.index, args.analyzer)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(
        not result['coverage_complete']
        or bool(result['cross_wave_pairs'] or result['collision_groups'])
    )


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f'ConSan application analysis: {error}', file=sys.stderr)
        sys.exit(2)
