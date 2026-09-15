"""JAPAN-Irregular reporting distributions and continuous-state protocol."""

PROFILES = {'L0': {'gaps': [1, 2, 3, 4, 5, 6, 7],
        'probabilities': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
 'L1': {'gaps': [1, 2, 3, 4, 5, 6, 7],
        'probabilities': [0.94, 0.015, 0.03, 0.005, 0.0025, 0.0025, 0.005]},
 'L2': {'gaps': [1, 2, 3, 4, 5, 6, 7],
        'probabilities': [0.88, 0.03, 0.06, 0.01, 0.005, 0.005, 0.01]},
 'L3': {'gaps': [1, 2, 3, 4, 5, 6, 7],
        'probabilities': [0.76, 0.06, 0.12, 0.02, 0.01, 0.01, 0.02]},
 'L4': {'gaps': [1, 2, 3, 4, 5, 6, 7],
        'probabilities': [0.52, 0.12, 0.24, 0.04, 0.02, 0.02, 0.04]}}

PROTOCOL = {'time_resolution': 'D',
 'date_start': '2020-04-01',
 'date_stop_inclusive': '2021-09-21',
 'raw_length': 539,
 'node_count': 47,
 'observation_length': 14,
 'forecast_horizon': 14,
 'split_ratio': [6, 1, 1],
 'reporting_ranges': {'train': [0, 407],
                      'validate': [407, 472],
                      'test': [472, 539]},
 'forecast_origin_ranges': {'train': [14, 394],
                            'validate': [421, 459],
                            'test': [486, 526]},
 'seed_derivation': {'algorithm': 'numpy.random.SeedSequence',
                     'split_stream_ids': {'train': 0, 'validate': 1, 'test': 2},
                     'node_stream': 'SeedSequence([derived_split_seed, '
                                    'node_index])'},
 'node_views': {'data_node_case_channel': 'log1p(observed_report_count)',
                'interval_node_case_channel': 'report_mask * '
                                              'log1p(report_rate)',
                'interval_node_appended_channels': ['report_mask',
                                                    'log1p(report_interval)']},
 'schema': 'costep.japan_irregular.v2',
 'state_construction_version': 2,
 'sir_reconstruction': {'method': 'synchronized_observed_exits_continuous_stocks',
                        'incidence': 'observed_report_count',
                        'exits': 'signed increments of reference R, '
                                 'accumulated with the same reporting mask as '
                                 'cases',
                        'reset_at_split_boundaries': False,
                        'initial_state_source': 'reference SIR at series '
                                                'start'},
 'first_report_gap': 'report date minus current split start plus one'}
