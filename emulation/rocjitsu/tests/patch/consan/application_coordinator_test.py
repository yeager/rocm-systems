# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise automatic whole-application preparation and owning index checks."""

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest.mock


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path, required=True)
    parser.add_argument('--coordinator', type=Path, required=True)
    parser.add_argument('--analyze-coordinator', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix='all-images-', dir=args.artifacts)
    ).resolve()
    print(f'evidence: {directory}', flush=True)
    inputs = directory / 'inputs'
    inputs.mkdir()
    for name in ('mixed', 'unsupported', 'unsupported-only'):
        shutil.copyfile(
            args.fixtures / f'multi-entry-{name}.hsaco', inputs / f'{name}.hsaco'
        )
    # Deduplicate byte-identical inputs while preserving every distinct image.
    shutil.copyfile(inputs / 'mixed.hsaco', inputs / 'duplicate.hsaco')
    output = directory / 'application'
    command = [
        str(args.coordinator),
        '--prepare',
        str(args.prepare),
        '--target',
        'gfx1250',
        '--images-dir',
        str(inputs),
        '--output',
        str(output),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=False
    )
    (directory / 'prepare.json').write_text(
        json.dumps(
            {
                'command': command,
                'exit': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            },
            indent=2,
        )
    )
    assert result.returncode == 0, result.stderr
    index_path = output / 'application.json'
    index = json.loads(index_path.read_text())
    assert len(index['images']) == 3, index
    outcomes = sorted(
        (
            row['preparation']['requested'],
            row['preparation']['instrumented'],
            row['preparation']['skipped'],
        )
        for row in index['images']
    )
    assert outcomes == [(1, 0, 1), (3, 2, 1), (3, 3, 0)], outcomes
    for row in index['images']:
        report = row['preparation']
        names = [entry['entry'] for entry in report['entries']]
        assert names == (
            ['mk_c'] if report['requested'] == 1 else ['mk_a', 'mk_b', 'mk_c']
        ), report
        artifact = output / 'images' / report['original_sha256']
        assert (artifact / 'original.hsaco').is_file()
        assert (artifact / 'ready').exists() == (report['instrumented'] != 0)
    catalog = output / 'application.bin'
    check = subprocess.run(
        [str(args.prepare), '--validate-application-catalog', str(catalog)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert check.returncode == 0 and check.stdout.strip() == 'images=3', check
    strict = subprocess.run(
        [
            str(args.coordinator),
            '--prepare',
            str(args.prepare),
            '--target',
            'gfx1250',
            '--images-dir',
            str(inputs),
            '--output',
            str(directory / 'strict'),
            '--require-all',
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert (
        strict.returncode == 2 and not (directory / 'strict/application.bin').exists()
    ), strict.stderr
    assert not (directory / 'strict/application.json').exists()
    # Index replay consumes owning reports, so absent captures remain explicit
    # coverage gaps even after all original/prepared input paths disappear.
    inputs.rename(directory / 'inputs.hidden')
    (output / 'images').rename(output / 'images.hidden')
    analyzer = load_module('consan_analyze_application', args.analyze_coordinator)
    missing = analyzer.analyze_application(index_path, Path('/unused-analyzer'))
    assert missing['scope'] == 'provided_application_images', missing
    assert missing['whole_application_coverage'] == 'not_established', missing
    assert not missing['lds_observations_present'], missing
    assert (
        missing['lds_analysis'] == 'not_performed' and not missing['coverage_complete']
    ), missing
    assert (
        sum('native_uncovered' in row['issues'] for row in missing['images']) == 1
    ), missing
    assert (
        sum(
            'selected_image_has_no_closed_capture' in row['issues']
            for row in missing['images']
        )
        == 2
    ), missing

    # The following tests isolate index aggregation from the native capture
    # decoder. The runtime/simulator tests exercise actual binary captures.
    complete = next(
        row for row in index['images'] if row['preparation']['skipped'] == 0
    )
    selected = complete['preparation']
    capture = output / complete['capture']
    capture.touch()
    fixture = {
        'original_sha256': selected['original_sha256'],
        'selection_profile': selected['selection_profile'],
        'target': selected['target'],
        'run_identity': '01' * 16,
        'scope': 'native_lds_single_range_mixed_scopes',
        'coverage_complete': True,
        'cross_wave_pairs': 0,
        'collision_groups': 0,
        'entries': [
            {
                'entry_key': entry['entry_key'],
                'entry': entry['entry'],
                'disposition': 'selected',
            }
            for entry in selected['entries']
        ],
        'executions': [
            {'entry_key': entry['entry_key'], 'valid_observations': 2}
            for entry in selected['entries']
        ],
    }
    single = {'schema_version': 1, 'scope': index['scope'], 'images': [complete]}
    single_path = output / 'single.json'
    single_path.write_text(json.dumps(single))

    def analyze(native):
        response = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(native), stderr=''
        )
        with unittest.mock.patch.object(
            analyzer.subprocess, 'run', return_value=response
        ):
            return analyzer.analyze_application(single_path, Path('/native-analyzer'))

    combined = analyze(fixture)
    assert (
        combined['lds_analysis'] == 'performed' and combined['coverage_complete']
    ), combined
    assert combined['lds_observations_present'], combined
    empty_memory = deepcopy(fixture)
    for execution in empty_memory['executions']:
        execution['valid_observations'] = 0
    combined = analyze(empty_memory)
    assert combined['lds_analysis'] == 'performed', combined
    assert not combined['lds_observations_present'], combined
    assert combined['coverage_complete'], combined
    assert combined['images'][0]['analysis']['coverage_complete'], combined
    assert not combined['images'][0]['issues'], combined
    absent_entry = deepcopy(fixture)
    absent_entry['executions'].pop()
    combined = analyze(absent_entry)
    assert not combined['coverage_complete'], combined
    assert combined['images'][0]['selected_entries_not_observed'] == [
        selected['entries'][-1]['entry']
    ], combined
    invalid = deepcopy(fixture)
    invalid['original_sha256'] = 'ff' * 32
    try:
        analyze(invalid)
    except ValueError as error:
        assert 'identities differ' in str(error)
    else:
        raise AssertionError('capture with a different original identity was accepted')
    single['images'].append(deepcopy(complete))
    single_path.write_text(json.dumps(single))
    try:
        analyze(fixture)
    except ValueError as error:
        assert 'duplicate original' in str(error)
    else:
        raise AssertionError('duplicate original identity was accepted')


if __name__ == '__main__':
    main()
