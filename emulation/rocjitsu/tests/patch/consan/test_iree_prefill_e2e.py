# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check wrapper evidence and replay cleanup without IREE or a GPU."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import consan_iree_prefill_e2e as prefill


def complete_analysis():
    keys = {prefill.SCATTER: 2048, prefill.ELEMENTWISE: 2112}
    entries = [
        {'entry': name, 'entry_key': key, 'disposition': 'selected'}
        for name, key in keys.items()
    ]
    entries.extend(
        {
            'entry': f'native_{index}',
            'entry_key': 2176 + index * 64,
            'disposition': 'native_uncovered',
        }
        for index in range(53)
    )
    executions = []
    for serial, name in enumerate(
        (prefill.SCATTER, prefill.ELEMENTWISE, prefill.SCATTER) * 3, 1
    ):
        groups = 16 if name == prefill.SCATTER else 1
        executions.append(
            {
                'serial': serial,
                'entry': name,
                'entry_key': keys[name],
                'grid': [groups * 32, 1, 1],
                'workgroup': [32, 1, 1],
                'expected_rows': groups,
                'completed_rows': groups,
                'entry_events': groups,
                'exit_events': groups,
                'integrity_events': 0,
                'issues': [],
            }
        )
    return {
        'scope': 'execution_trace',
        'lds_analysis': 'not_performed',
        'transport_complete': True,
        'entries': entries,
        'executions': executions,
        'expected_rows': 99,
        'completed_rows': 99,
        'entry_events': 99,
        'exit_events': 99,
    }


def lds_analysis():
    executions = [
        {
            'serial': serial,
            'entry': 'prefill_bs1$async_dispatch_3_matmul_Dx128x256_f16xf16xf32',
            'entry_key': 2048,
            'workgroup_count': 2,
            'waves_per_workgroup': 4,
            'valid_observations': 4608,
            'rejected_records': 0,
            'integrity_events': 0,
            'lost_occurrences': 576,
            'synchronization_loss': 0,
            'address_group_exhaustion': 576,
            'counter_exhausted_waves': 0,
            'coverage_complete': False,
            'issues': [{'reason': 'range_reservation_exhausted'}],
        }
        for serial in range(1, 7)
    ]
    return {
        'verdict': 'no_observed_findings',
        'scope': 'native_lds_multi_range_epoch_workgroup_barriers',
        'lds_analysis': 'performed',
        'execution_conditions': 8,
        'coverage_complete': False,
        'entries': [
            {
                'entry': executions[0]['entry'],
                'entry_key': 2048,
                'disposition': 'selected',
            }
        ]
        + [
            {
                'entry': f'native_{index}',
                'entry_key': 2112 + index * 64,
                'disposition': 'native_uncovered',
            }
            for index in range(54)
        ],
        'executions': executions,
    }


class PrefillEvidence(unittest.TestCase):
    def test_all_image_evidence_requires_every_attempt_and_actual_lds_observations(
        self,
    ):
        native = lds_analysis()
        native['entries'][0]['entry'] = prefill.LDS_FIRST_MATMUL
        native['entries'][1]['entry'] = prefill.LDS_MATMUL
        native['executions'] = native['executions'][:4]
        for execution in native['executions']:
            execution.update(
                entry=prefill.LDS_FIRST_MATMUL,
                workgroup_count=4,
                valid_observations=9216,
                lost_occurrences=1152,
                address_group_exhaustion=1152,
            )
        model_entries = [
            dict(
                entry,
                disposition=(
                    'instrumented'
                    if entry['disposition'] == 'selected'
                    else 'native_uncovered'
                ),
                reason_code=(
                    'none'
                    if entry['disposition'] == 'selected'
                    else 'unsupported_instruction'
                ),
                reason='' if entry['disposition'] == 'selected' else 'outside subset',
            )
            for entry in native['entries']
        ]
        model_entries[1]['reason_code'] = 'unsupported_control_transfer_range'
        index = {
            'scope': 'provided_application_images',
            'images': [
                {
                    'preparation': {
                        'original_sha256': '11' * 32,
                        'requested': 55,
                        'entries': model_entries,
                    }
                },
                {
                    'preparation': {
                        'original_sha256': '22' * 32,
                        'requested': 13,
                        'entries': [
                            {
                                'entry': f'helper_{index}',
                                'disposition': 'native_uncovered',
                                'reason_code': 'unsupported_entry',
                                'reason': 'no LDS sites',
                            }
                            for index in range(13)
                        ],
                    }
                },
            ],
        }
        analysis = {
            'scope': 'provided_application_images',
            'whole_application_coverage': 'not_established',
            'lds_analysis': 'performed',
            'lds_observations_present': True,
            'coverage_complete': False,
            'images': [
                {'original_sha256': '11' * 32, 'analysis': native},
                {'original_sha256': '22' * 32, 'analysis': None},
            ],
        }
        prefill.validate_all_images(index, analysis)
        for field, value in (
            ('scope', 'application'),
            ('lds_analysis', 'not_performed'),
            ('lds_observations_present', False),
            ('coverage_complete', True),
        ):
            changed = deepcopy(analysis)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_all_images(index, changed)
        changed = deepcopy(index)
        changed['images'][0]['preparation']['requested'] = 1
        with self.assertRaisesRegex(ValueError, 'every model and helper'):
            prefill.validate_all_images(changed, analysis)
        changed = deepcopy(analysis)
        changed['images'].pop()
        with self.assertRaisesRegex(ValueError, 'dropped a provided image'):
            prefill.validate_all_images(index, changed)
        changed = deepcopy(analysis)
        changed['images'][0]['analysis']['executions'][0]['valid_observations'] = 0
        with self.assertRaisesRegex(ValueError, 'observation/loss'):
            prefill.validate_all_images(index, changed)

    def test_lds_analysis_requires_observations_for_every_selected_execution(self):
        prefill.validate_lds(lds_analysis())
        for field, value in (
            ('lds_analysis', 'not_performed'),
            ('execution_conditions', 15),
            ('executions', []),
            ('scope', 'execution_trace'),
        ):
            analysis = lds_analysis()
            analysis[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_lds(analysis)
        for field, value in (
            ('serial', 9),
            ('entry', prefill.SCATTER),
            ('entry_key', 2112),
            ('workgroup_count', 1),
            ('waves_per_workgroup', 1),
            ('valid_observations', 0),
            ('valid_observations', 4607),
            ('lost_occurrences', 575),
            ('address_group_exhaustion', 575),
            ('synchronization_loss', 1),
            ('integrity_events', 1),
            ('rejected_records', 1),
            ('counter_exhausted_waves', 1),
            ('issues', [{'reason': 'invalid_record_commit_identity_mask_or_sequence'}]),
        ):
            analysis = lds_analysis()
            analysis['executions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_lds(analysis)

    def test_findings_are_a_completed_analysis_and_input_errors_are_rejected(self):
        analysis = lds_analysis()
        analysis['verdict'] = 'findings'
        prefill.validate_lds(analysis, 1)
        with self.assertRaisesRegex(ValueError, 'status contradicts'):
            prefill.validate_lds(analysis, 0)
        for code in (0, 1, 2, 3):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                commands = prefill.Commands(root, root, {}, 10)
                with mock.patch.object(
                    prefill.subprocess, 'run', return_value=mock.Mock(returncode=code)
                ):
                    if code in (0, 1):
                        commands.run('analysis', ['analyzer'], allowed_exits=(0, 1))
                    else:
                        with self.assertRaisesRegex(ValueError, f'exit {code}'):
                            commands.run('analysis', ['analyzer'], allowed_exits=(0, 1))
                self.assertEqual(commands.records[-1]['exit'], code)

    def test_findings_replay_requires_the_same_analyzer_exit_status(self):
        for code in (0, 1):
            commands = mock.Mock()
            commands.run.return_value = 'analysis'
            commands.records = [{'exit': code}]
            if code == 1:
                prefill.offline_replay(
                    commands,
                    'analyzer',
                    'capture',
                    (),
                    'analysis',
                    allowed_exits=(0, 1),
                    expected_exit=1,
                )
            else:
                with self.assertRaisesRegex(ValueError, 'status differs'):
                    prefill.offline_replay(
                        commands,
                        'analyzer',
                        'capture',
                        (),
                        'analysis',
                        allowed_exits=(0, 1),
                        expected_exit=1,
                    )
            self.assertEqual(commands.run.call_args.kwargs['allowed_exits'], (0, 1))

    def test_bounded_lds_loss_is_reported_without_claiming_complete_coverage(self):
        analysis = lds_analysis()
        execution = analysis['executions'][0]
        prefill.validate_lds(analysis)
        execution['coverage_complete'] = True
        with self.assertRaises(ValueError):
            prefill.validate_lds(analysis)
        execution['coverage_complete'] = False
        analysis['coverage_complete'] = True
        with self.assertRaises(ValueError):
            prefill.validate_lds(analysis)

    def test_complete_trace_and_rejection_of_scope_or_inventory_gaps(self):
        self.assertEqual(
            prefill.validate_trace(complete_analysis()),
            {prefill.SCATTER: 2048, prefill.ELEMENTWISE: 2112},
        )
        for field, value in (
            ('scope', 'native_lds'),
            ('lds_analysis', 'performed'),
            ('transport_complete', False),
            ('executions', []),
            ('completed_rows', 98),
        ):
            analysis = complete_analysis()
            analysis[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_trace(analysis)
        analysis = complete_analysis()
        analysis['entries'] = analysis['entries'][1:]
        with self.assertRaisesRegex(ValueError, 'inventory'):
            prefill.validate_trace(analysis)

    def test_each_execution_identity_geometry_and_boundary_is_checked(self):
        for field, value in (
            ('serial', 2),
            ('entry', prefill.ELEMENTWISE),
            ('entry_key', 2112),
            ('grid', [32, 1, 1]),
            ('workgroup', [64, 1, 1]),
            ('completed_rows', 15),
            ('entry_events', 0),
            ('exit_events', 2),
            ('integrity_events', 1),
            ('issues', ['invalid_entry_exit_row_0']),
        ):
            analysis = complete_analysis()
            analysis['executions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_trace(analysis)

    def test_clean_environment_preserves_sdk_paths_and_removes_old_tool_settings(self):
        removed = {
            'RJ_CONSAN_ARTIFACT_DIR': '/stale/artifact',
            'RJ_CONSAN_CAPTURE_PATH': '/shared/capture',
            'HSA_TOOLS_LIB': '/stale/plugin.so',
            'HSA_TOOLS_DISABLE_REGISTER': '1',
            'LD_PRELOAD': '/other/interposer.so',
            'LOADER_OPTIONS_APPEND': '-dump-code',
            'GPU_DUMP_CODE_OBJECT': '1',
        }
        sdk = {'PATH': '/own/venv/bin:/own/sdk/bin', 'LD_LIBRARY_PATH': '/own/sdk/lib'}
        inherited = (
            removed | sdk | {'HSA_HOTSWAP_DISABLE': '0', 'HSA_HOTSWAP_ENABLE': '1'}
        )
        cleaned = prefill.clean_environment(inherited)
        for key in removed:
            self.assertNotIn(key, cleaned)
            self.assertIn(key, inherited)
        for key, value in sdk.items():
            self.assertEqual(cleaned[key], value)
        self.assertEqual(cleaned['HSA_HOTSWAP_DISABLE'], '1')
        self.assertEqual(cleaned['HSA_HOTSWAP_ENABLE'], '0')

    def test_replay_hides_inputs_and_restores_them_on_success_or_failure(self):
        for outcome in ('same', 'changed', 'process-failure'):
            with self.subTest(
                outcome=outcome
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = (root / 'original', root / 'artifact')
                for path in paths:
                    path.mkdir()
                    (path / 'witness').write_text(path.name)
                analyzer, capture = root / 'analyzer', root / 'capture.bin'
                commands = mock.Mock()

                def analyze(_stage, argv, *, allowed_exits):
                    self.assertEqual(allowed_exits, (0,))
                    self.assertEqual(argv, [analyzer, capture])
                    self.assertTrue(all(not path.exists() for path in paths))
                    if outcome == 'process-failure':
                        raise RuntimeError('analyzer failed')
                    return 'analysis' if outcome == 'same' else 'different analysis'

                commands.run.side_effect = analyze
                if outcome == 'same':
                    prefill.offline_replay(
                        commands, analyzer, capture, paths, 'analysis'
                    )
                else:
                    error = RuntimeError if outcome == 'process-failure' else ValueError
                    with self.assertRaises(error):
                        prefill.offline_replay(
                            commands, analyzer, capture, paths, 'analysis'
                        )
                commands.run.assert_called_once()
                for path in paths:
                    self.assertEqual((path / 'witness').read_text(), path.name)
                    self.assertFalse(path.with_name(path.name + '.hidden').exists())

    def test_replay_never_clobbers_an_existing_hidden_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'original'
            original.mkdir()
            hidden = root / 'original.hidden'
            hidden.mkdir()
            (hidden / 'unrelated').write_text('preserve')
            commands = mock.Mock()
            with self.assertRaisesRegex(ValueError, 'scratch paths already exist'):
                prefill.offline_replay(
                    commands,
                    root / 'analyzer',
                    root / 'capture.bin',
                    (original,),
                    'analysis',
                )
            commands.run.assert_not_called()
            self.assertTrue(original.is_dir())
            self.assertEqual((hidden / 'unrelated').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
