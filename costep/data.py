import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from .intervals import build_future_reporting_payload, build_history_reporting_payload, build_reporting_audit, build_reporting_metadata, build_window_reporting_payload, is_report_mae_selection_metric, write_reporting_audit
from .irregular import build_japan_irregular_view, write_japan_irregular_manifest, load_japan_irregular_profile

def build_model_reporting_view(view_data: dict, reporting_metadata: dict | None, *, reporting_mode: str, preserve_regular_node_topology: bool=False):
    if reporting_mode != 'interval' or preserve_regular_node_topology:
        return view_data
    if reporting_metadata is None:
        raise ValueError('interval model features require reporting metadata')
    feature_view = dict(view_data)
    interval_node = np.array(view_data['node'], copy=True)
    interval_case = np.where(reporting_metadata['report_mask'], np.log1p(reporting_metadata['report_rate']), 0.0)
    interval_node[..., 0] = interval_case[..., 0]
    interval_node = np.concatenate([interval_node, reporting_metadata['report_mask'].astype(interval_node.dtype), np.log1p(reporting_metadata['report_interval']).astype(interval_node.dtype)], axis=-1)
    feature_view['node'] = interval_node
    feature_view['sir_documented_y'] = reporting_metadata['causal_report_rate']
    return feature_view

class StandardScaler:

    def __init__(self, mean, std):
        self.mean = mean
        raw_std = np.asarray(std)
        if not np.all(np.isfinite(raw_std)):
            raise ValueError('StandardScaler std must be finite')
        if np.any(raw_std < 0):
            raise ValueError('StandardScaler std must be nonnegative')
        dtype = raw_std.dtype if np.issubdtype(raw_std.dtype, np.floating) else np.dtype('float64')
        self._roundoff_tolerance = 10 * np.finfo(dtype).eps * np.maximum(1.0, np.abs(mean))
        self._near_constant = raw_std <= self._roundoff_tolerance
        safe_std = np.where(self._near_constant, 1.0, raw_std)
        self.std = safe_std.item() if safe_std.ndim == 0 else safe_std

    def transform(self, data):
        if not hasattr(self, '_near_constant'):
            self.__init__(self.mean, self.std)
        centered = data - self.mean
        if torch.is_tensor(centered):
            near = torch.as_tensor(np.asarray(self._near_constant), device=centered.device)
            tol = torch.as_tensor(self._roundoff_tolerance, device=centered.device)
            centered = torch.where(near & (centered.abs() <= tol), 0.0, centered)
        else:
            centered = np.where(self._near_constant & (np.abs(centered) <= self._roundoff_tolerance), 0.0, centered)
        return centered / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

class ForecastDataset(Dataset):

    def __init__(self, inputs: dict, output, mode: str, mode_len: dict, mode_indices: dict | None=None):
        self.mode = mode
        self.mode_len = mode_len
        self.mode_indices = mode_indices
        (self.inputs, self.output) = self.prepare_xy(inputs, output)

    def __len__(self):
        if self.mode_indices is not None:
            return len(self.mode_indices[self.mode])
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        output_item = self._get_item(self.output, item)
        return (self.inputs['x_od'][item], self.inputs['x_node'][item], self.inputs['x_SIR'][item], output_item)

    def prepare_xy(self, inputs: dict, output):
        x = dict()
        if self.mode_indices is not None:
            indices = np.asarray(self.mode_indices[self.mode], dtype=np.int64)
            x['x_od'] = self._select_by_indices(inputs['x_od'], indices)
            x['x_SIR'] = self._select_by_indices(inputs['x_SIR'], indices)
            x['x_node'] = self._select_by_indices(inputs['x_node'], indices)
            y = self._select_by_indices(output, indices)
            return (x, y)
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:
            start_idx = self.mode_len['train'] + self.mode_len['validate']
        x['x_od'] = inputs['x_od'][start_idx:start_idx + self.mode_len[self.mode]]
        x['x_SIR'] = inputs['x_SIR'][start_idx:start_idx + self.mode_len[self.mode]]
        x['x_node'] = inputs['x_node'][start_idx:start_idx + self.mode_len[self.mode]]
        y = self._slice_values(output, start_idx, start_idx + self.mode_len[self.mode])
        return (x, y)

    @classmethod
    def _select_by_indices(cls, values, indices: np.ndarray):
        if isinstance(values, dict):
            return {key: cls._select_by_indices(value, indices) for (key, value) in values.items()}
        if isinstance(values, list):
            return [values[int(idx)] for idx in indices.tolist()]
        return values[indices]

    @classmethod
    def _slice_values(cls, values, start: int, stop: int):
        if isinstance(values, dict):
            return {key: cls._slice_values(value, start, stop) for (key, value) in values.items()}
        return values[start:stop]

    @classmethod
    def _get_item(cls, values, item):
        if isinstance(values, dict):
            return {key: cls._get_item(value, item) for (key, value) in values.items()}
        return values[item]

class AuditedEpochPermutationSampler(Sampler):
    """Deterministic per-epoch sampler with an append-only origin-order audit."""

    def __init__(self, origins: torch.Tensor, seed: int, audit_path: str | Path):
        self.origins = torch.as_tensor(origins, dtype=torch.int64).cpu()
        self.seed = int(seed)
        self.audit_path = Path(audit_path)
        self.epoch = 0
        if self.audit_path.exists():
            raise FileExistsError(f'refusing to overwrite sampler audit: {self.audit_path}')
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return int(self.origins.numel())

    def __iter__(self):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self), generator=generator).tolist()
        ordered_origins = [int(self.origins[index]) for index in order]
        payload = {'epoch': self.epoch + 1, 'permutation_seed': self.seed + self.epoch, 'dataset_indices': order, 'absolute_origins': ordered_origins}
        with self.audit_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, sort_keys=True) + '\n')
        self.epoch += 1
        return iter(order)

class WindowBuilder(object):

    def __init__(self, obs_len: int, pred_len, data_split_ratio: tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len: int):
        mode_len = dict()
        mode_len['train'] = int(self.data_split_ratio[0] / sum(self.data_split_ratio) * data_len)
        mode_len['validate'] = int(self.data_split_ratio[1] / sum(self.data_split_ratio) * data_len)
        mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
        return mode_len

    @staticmethod
    def _split_indices_by_ratio(start: int, stop: int, ratio: tuple[int, int], left_name: str, right_name: str):
        total_len = int(stop) - int(start)
        if total_len <= 1:
            raise ValueError(f'Cannot split [{start}, {stop}) into {left_name}/{right_name}: need at least 2 windows, got {total_len}.')
        left_len = int(float(ratio[0]) / float(sum(ratio)) * total_len)
        right_len = total_len - left_len
        if left_len <= 0 or right_len <= 0:
            raise ValueError(f'Cannot split [{start}, {stop}) into non-empty {left_name}/{right_name}: {left_name}={left_len}, {right_name}={right_len}.')
        split_at = int(start) + left_len
        return {left_name: np.arange(int(start), split_at, dtype=np.int64), right_name: np.arange(split_at, int(stop), dtype=np.int64)}

    def build_mode_indices(self, data_len: int, params: dict | None=None):
        params = params or {}
        split_mode = params.get('split_mode', 'ratio')
        if split_mode == 'ratio':
            legacy_mode_len = self.split2len(data_len=data_len)
            if not params.get('non_test_overlap', False):
                return (legacy_mode_len, None)
            legacy_test_start = int(legacy_mode_len['train']) + int(legacy_mode_len['validate'])
            discarded_count = int(self.pred_len) - 1
            pretest_stop = legacy_test_start - discarded_count
            if pretest_stop <= 1:
                raise ValueError(f'non_test_overlap leaves too few pre-test origins after removing {discarded_count} boundary windows: stop={pretest_stop}')
            pretest = self._split_indices_by_ratio(start=0, stop=pretest_stop, ratio=(self.data_split_ratio[0], self.data_split_ratio[1]), left_name='train', right_name='validate')
            mode_indices = {**pretest, 'test': np.arange(legacy_test_start, int(data_len), dtype=np.int64)}
            mode_len = {mode: int(len(indices)) for (mode, indices) in mode_indices.items()}
            params['non_test_overlap_contract'] = {'legacy_mode_len': {mode: int(length) for (mode, length) in legacy_mode_len.items()}, 'legacy_test_start_sample': int(legacy_test_start), 'test_stop_sample': int(data_len), 'legal_information_cutoff_raw': int(legacy_test_start + self.obs_len), 'boundary_start_sample': int(legacy_test_start - discarded_count), 'boundary_stop_sample': int(legacy_test_start), 'boundary_window_count': int(discarded_count), 'partial_labels': False, 'post_train_full_boundary_dual_use': False, 'boundary_dual_use_absolute_indices': [], 'pretest_stop_sample': int(pretest_stop), 'mode_len': dict(mode_len)}
            return (mode_len, mode_indices)
        if split_mode == 'calendar_disjoint':
            raw_data_len = int(params.get('raw_data_len', data_len + self.obs_len + self.pred_len - 1))
            target_days = raw_data_len - int(self.obs_len)
            if target_days < 3 * int(self.pred_len):
                raise ValueError('calendar_disjoint requires at least pred_len target dates in each split')
            ratio = tuple((float(value) for value in self.data_split_ratio))
            total = float(sum(ratio))
            train_days = int(ratio[0] / total * target_days)
            validate_days = int(ratio[1] / total * target_days)
            test_days = target_days - train_days - validate_days
            if min(train_days, validate_days, test_days) < int(self.pred_len):
                raise ValueError(f'calendar_disjoint produced a split shorter than pred_len: train={train_days}, validate={validate_days}, test={test_days}')
            train_start = int(self.obs_len)
            train_stop = train_start + train_days
            validate_start = train_stop
            validate_stop = validate_start + validate_days
            test_start = validate_stop
            test_stop = raw_data_len

            def origins_to_samples(start, stop):
                last_origin = int(stop) - int(self.pred_len)
                if last_origin < int(start):
                    return np.asarray([], dtype=np.int64)
                origins = np.arange(int(start), last_origin + 1, dtype=np.int64)
                return origins - int(self.obs_len)
            split_contained_context = bool(params.get('japan_irregular', False))
            validate_origin_start = validate_start + int(self.obs_len) if split_contained_context else validate_start
            test_origin_start = test_start + int(self.obs_len) if split_contained_context else test_start
            mode_indices = {'train': origins_to_samples(train_start, train_stop), 'validate': origins_to_samples(validate_origin_start, validate_stop), 'test': origins_to_samples(test_origin_start, test_stop)}
            mode_len = {mode: int(len(indices)) for (mode, indices) in mode_indices.items()}
            params['calendar_split_boundaries'] = {'train': (0, train_stop), 'validate': (validate_start, validate_stop), 'test': (test_start, test_stop)}
            return (mode_len, mode_indices)
        raise ValueError('Only ratio and calendar_disjoint splits are supported')

    @staticmethod
    def _build_modify_amount_features(od_window: np.ndarray, sir_window: np.ndarray):
        od_core = od_window[..., 0]
        outbound_total = od_core.sum(axis=2)
        population = sir_window[[0], :, :3].sum(axis=-1)
        population = np.clip(population, a_min=1e-06, a_max=None)
        outbound_safe = np.clip(outbound_total, a_min=1e-06, a_max=None)
        amount_per_pop = outbound_total / population
        stay_ratio = np.diagonal(od_core, axis1=1, axis2=2) / outbound_safe
        return np.stack([amount_per_pop, stay_ratio], axis=-1)

    def _append_modify_amount_features(self, node_window: np.ndarray, od_window: np.ndarray, sir_window: np.ndarray):
        aligned_od = od_window[:node_window.shape[0], ...]
        amount_feats = self._build_modify_amount_features(aligned_od, sir_window)
        return np.concatenate([node_window, amount_feats], axis=-1)

    @staticmethod
    def _normalize_modify_amount_mode(modify_amount):
        if modify_amount in [None, False, 'origin']:
            return 'norm'
        return modify_amount

    @staticmethod
    def _slice_or_pad(values, start: int, stop: int):
        """Return an exact-length time slice, padding only unavailable tail dates."""
        target_len = int(stop) - int(start)
        window = values[int(start):min(int(stop), len(values))]
        if len(window) == target_len:
            return window
        pad_shape = (target_len - len(window),) + tuple(window.shape[1:])
        return np.concatenate([window, np.zeros(pad_shape, dtype=window.dtype)], axis=0)

    def _build_stitched_masks(self, sample_count: int, raw_data_len: int, mode_len: dict, mode_indices: dict | None, extra_len: int):
        """Build calendar-safe context and auxiliary-label masks.

        The ordinary split is defined by 14-day targets and remains unchanged.  A
        training origin may use the extra seven days only if the *whole* extra
        block ends no later than the last ordinary target in that training
        partition.  This masks the last ``extra_len`` training origins instead
        of moving the train/validation boundary.
        """
        context_len = self.obs_len + self.pred_len + extra_len
        sample_indices = np.arange(sample_count, dtype=np.int64)
        raw_available = np.clip(raw_data_len - sample_indices, 0, context_len)
        context_mask = np.arange(context_len)[None, :] < raw_available[:, None]
        extra_mask = np.zeros((sample_count, extra_len, 1, 1), dtype=bool)
        if mode_indices is None:
            starts = {'train': 0, 'validate': mode_len['train'], 'test': mode_len['train'] + mode_len['validate']}
            groups = {mode: np.arange(starts[mode], starts[mode] + mode_len[mode], dtype=np.int64) for mode in ['train', 'validate', 'test'] if mode in mode_len}
        else:
            groups = {mode: np.asarray(indices, dtype=np.int64) for (mode, indices) in mode_indices.items()}
        for mode in ['train', 'adapt_train']:
            indices = groups.get(mode, np.asarray([], dtype=np.int64))
            if len(indices) == 0:
                continue
            ordinary_target_boundary = self.obs_len + int(indices.max()) + self.pred_len
            allowed_context_len = np.clip(ordinary_target_boundary - indices, 0, context_len)
            context_mask[indices] &= np.arange(context_len)[None, :] < allowed_context_len[:, None]
            eligible = indices + extra_len <= int(indices.max())
            extra_mask[indices[eligible], :, :, :] = True
        return (context_mask, extra_mask)

    @staticmethod
    def _apply_time_mask(values, mask: np.ndarray):
        """Zero forbidden calendar positions without materializing lazy OD views."""
        if isinstance(values, list):
            masked = list(values)
            for idx in np.flatnonzero(~mask.all(axis=1)):
                window = np.array(masked[int(idx)], copy=True)
                window[~mask[int(idx)]] = 0
                masked[int(idx)] = window
            return masked
        masked = np.array(values, copy=True)
        masked[~mask] = 0
        return masked

    @staticmethod
    def _standardize_node_features(x_node: np.ndarray, train_indices: np.ndarray, fit_time_len: int | None=None):
        """Fit node scalers on a fixed context prefix, then transform all dates."""
        node_scalers = []
        fit_time_slice = slice(None, fit_time_len)
        for feature_idx in range(x_node.shape[-1]):
            train_feature = x_node[train_indices, fit_time_slice, ..., feature_idx]
            scaler = StandardScaler(mean=train_feature.mean(), std=train_feature.std())
            node_scalers.append(scaler)
            x_node[..., feature_idx] = scaler.transform(x_node[..., feature_idx])
        return (node_scalers, x_node)

    @staticmethod
    def _fit_max_od(values, train_indices: np.ndarray, fit_time_len: int | None=None):
        fit_time_slice = slice(None, fit_time_len)
        if isinstance(values, list):
            if fit_time_len is None:
                return max((values[int(idx)].max() for idx in train_indices.tolist()))
            return max((values[int(idx)][fit_time_slice, ..., 0].max() for idx in train_indices.tolist()))
        return values[train_indices, fit_time_slice, ..., 0].max()

    def get_data_loader(self, data: dict, params: dict):
        modify_amount = self._normalize_modify_amount_mode('align')
        reporting_mode = params.get('reporting_mode', 'daily')
        selection_metric = params.get('selection_metric', 'legacy_mae')
        needs_reporting_metadata = reporting_mode == 'interval' or params.get('loss') == 'report_mae' or is_report_mae_selection_metric(selection_metric) or bool(params.get('emit_reporting_metrics', False)) or bool(params.get('japan_irregular', False))
        stitched_extra_len = int(params.get('stitched_extra_len', 7))
        if stitched_extra_len != 7:
            raise ValueError('stitched_extra_len must equal cross_origin_shift so shifted context, targets, and reporting metadata share one origin')
        raw_data_len = int(np.asarray(data['y']).shape[0])
        sample_count = raw_data_len - int(self.obs_len) - int(self.pred_len) + 1
        split_params = dict(params)
        split_params['raw_data_len'] = raw_data_len
        (mode_len, mode_indices) = self.build_mode_indices(data_len=sample_count, params=split_params)
        if 'calendar_split_boundaries' in split_params:
            params['calendar_split_boundaries'] = split_params['calendar_split_boundaries']
        if 'non_test_overlap_contract' in split_params:
            params['non_test_overlap_contract'] = split_params['non_test_overlap_contract']
        params['_materialized_mode_lengths'] = {mode: int(length) for (mode, length) in mode_len.items()}

        def build_view(view_data):
            metadata = None
            windows = None
            if needs_reporting_metadata:
                metadata = view_data.get('reporting_metadata')
                if metadata is None:
                    metadata = build_reporting_metadata(view_data['y'], report_mask=view_data.get('report_mask'), reset_indices=tuple(view_data.get('reporting_reset_indices', ())))
                windows = build_window_reporting_payload(metadata, obs_len=self.obs_len, pred_len=self.pred_len)
            feature_view = build_model_reporting_view(view_data, metadata, reporting_mode=reporting_mode, preserve_regular_node_topology=False)
            view_feats = self.get_feats(feature_view, stitched_extra_len=stitched_extra_len)
            return (metadata, windows, view_feats)
        augmentation_mode = params.get('reporting_augmentation', 'native')
        japan_irregular = bool(params.get('japan_irregular', False))
        active_data = data
        profile = None
        if augmentation_mode == 'coarsened' or japan_irregular:
            profile = load_japan_irregular_profile(params['japan_irregular_level'])
        if japan_irregular:
            if params.get('dataset') != 'JAPAN':
                raise ValueError('--japan_irregular requires --dataset JAPAN')
            if params.get('split_mode') != 'calendar_disjoint':
                raise ValueError('--japan_irregular requires calendar_disjoint')
            data_seed = params.get('reporting_data_seed')
            if data_seed is None:
                raise ValueError('--japan_irregular requires --reporting_data_seed')
            (active_data, irregular_manifest) = build_japan_irregular_view(data, split_ranges=params['calendar_split_boundaries'], profile=profile, data_seed=int(data_seed), coarsen_train=augmentation_mode == 'coarsened', observation_length=self.obs_len, forecast_horizon=self.pred_len)
            write_japan_irregular_manifest(Path(params['output_dir']) / 'japan_irregular_manifest.json', irregular_manifest)
        (reporting_metadata, reporting_windows, feats) = build_view(active_data)
        stitched_reporting_windows = None
        shifted_reporting_windows = None
        consistency_history_report_mask = None
        if reporting_windows is not None:
            stitched_reporting_windows = {key: np.array(value, copy=True) for (key, value) in reporting_windows.items()}
            base_origins = np.arange(sample_count, dtype=np.int64) + self.obs_len
            max_consistency_shift = max(7, self.pred_len // 2)
            consistency_origins = base_origins[:, None] + np.arange(1, max_consistency_shift + 1, dtype=np.int64)[None, :]
            consistency_history = build_history_reporting_payload(reporting_metadata, origin_indices=consistency_origins.reshape(-1), obs_len=self.obs_len)['history_report_mask']
            consistency_history_report_mask = consistency_history.reshape((sample_count, max_consistency_shift) + consistency_history.shape[1:])
            shifted_origins = base_origins + 7
            shifted_reporting_windows = build_future_reporting_payload(reporting_metadata, origin_indices=shifted_origins, pred_len=self.pred_len)
        (x_od, x_node, x_SIR, y) = feats
        if np.prod(x_od[0].shape) <= 100000:
            x_od = np.asarray(x_od)
        x_node = np.asarray(x_node)
        x_SIR = np.asarray(x_SIR)
        y = np.asarray(y)
        if reporting_metadata is not None:
            (audit_summary, audit_rows) = build_reporting_audit(active_data, reporting_metadata, protocol=params.get('us_protocol'), mode_indices=mode_indices, mode_len=mode_len, obs_len=self.obs_len, pred_len=self.pred_len)
            write_reporting_audit(params['output_dir'], audit_summary, audit_rows)
        overlap_contract = params.get('non_test_overlap_contract')
        overlap_target_valid_mask = None
        pretest_indices = None
        if overlap_contract is not None:
            pretest_indices = np.concatenate([np.asarray(mode_indices[mode], dtype=np.int64) for mode in ('train', 'validate', 'adapt_train', 'adapt_val') if mode in mode_indices and len(mode_indices[mode]) > 0])
            cutoff = int(overlap_contract['legal_information_cutoff_raw'])
            target_calendar = np.arange(sample_count, dtype=np.int64)[:, None] + int(self.obs_len) + np.arange(y.shape[1], dtype=np.int64)[None, :]
            overlap_target_valid_mask = np.ones((sample_count, y.shape[1], 1, 1), dtype=bool)
            overlap_target_valid_mask[pretest_indices] = target_calendar[pretest_indices, :, None, None] < cutoff
            main_valid = overlap_target_valid_mask[pretest_indices, :self.pred_len, ...]
            if not bool(main_valid.all()):
                raise RuntimeError('non_test_overlap retained a pre-test origin whose ordinary target crosses the legal information cutoff')
            invalid_target = ~overlap_target_valid_mask[pretest_indices]
            selected_targets = y[pretest_indices]
            y[pretest_indices] = np.where(invalid_target, np.nan, selected_targets)
        (stitched_context_mask, stitched_extra_mask) = self._build_stitched_masks(sample_count=y.shape[0], raw_data_len=data['y'].shape[0], mode_len=mode_len, mode_indices=mode_indices, extra_len=stitched_extra_len)
        if overlap_contract is not None:
            cutoff = int(overlap_contract['legal_information_cutoff_raw'])
            context_calendar = np.arange(sample_count, dtype=np.int64)[:, None] + np.arange(stitched_context_mask.shape[1], dtype=np.int64)[None, :]
            stitched_context_mask[pretest_indices] &= context_calendar[pretest_indices] < cutoff
            extra_available = overlap_target_valid_mask[:, self.pred_len:self.pred_len + stitched_extra_len, ...]
            stitched_extra_mask &= extra_available
        x_od = self._apply_time_mask(x_od, stitched_context_mask)
        x_node = self._apply_time_mask(x_node, stitched_context_mask)
        x_SIR = self._apply_time_mask(x_SIR, stitched_context_mask)
        y[:, self.pred_len:self.pred_len + stitched_extra_len, ...] = np.where(stitched_extra_mask, y[:, self.pred_len:self.pred_len + stitched_extra_len, ...], 0)
        train_indices = np.asarray(mode_indices['train'], dtype=np.int64) if mode_indices is not None else np.arange(mode_len['train'], dtype=np.int64)
        if len(train_indices):
            params['_stitched_train_max_origin'] = int(train_indices.max()) + self.obs_len
            params['_stitched_common_support_origin_count'] = int(len(train_indices))
        scaler_split_ratio = params.get('scaler_split_ratio')
        if mode_indices is not None:
            scaler_mode_len = {mode: int(len(indices)) for (mode, indices) in mode_indices.items()}
        elif scaler_split_ratio is None:
            scaler_mode_len = mode_len
        else:
            scaler_total = float(sum(scaler_split_ratio))
            scaler_mode_len = {'train': int(float(scaler_split_ratio[0]) / scaler_total * y.shape[0]), 'validate': int(float(scaler_split_ratio[1]) / scaler_total * y.shape[0])}
            scaler_mode_len['test'] = y.shape[0] - scaler_mode_len['train'] - scaler_mode_len['validate']
        if mode_indices is None:
            print('train: [{}-{}), val: [{}-{}), test: [{}-{})'.format(self.obs_len, mode_len['train'] + self.obs_len, mode_len['train'] + self.obs_len, mode_len['train'] + mode_len['validate'] + self.obs_len, mode_len['train'] + mode_len['validate'] + self.obs_len, mode_len['train'] + mode_len['validate'] + mode_len['test'] + self.obs_len))
        else:
            split_ranges = []
            for (mode_name, indices) in mode_indices.items():
                if len(indices) == 0:
                    split_ranges.append(f'{mode_name}: empty')
                elif len(indices) == 1:
                    split_ranges.append(f'{mode_name}: [{int(indices[0])}]')
                else:
                    split_ranges.append(f'{mode_name}: [{int(indices[0])}-{int(indices[-1])}]')
            print(f"{params.get('split_mode', 'ratio')} split | " + ', '.join(split_ranges))
        print('scaler fit split:', scaler_mode_len)
        scaler_fit_time_len = self.obs_len + self.pred_len
        max_od = self._fit_max_od(x_od, train_indices, scaler_fit_time_len)
        if isinstance(x_od, np.ndarray) or isinstance(x_od, torch.Tensor):
            x_od = x_od / max_od
        elif isinstance(x_od, list):
            data['od'] /= max_od
            for od_window in x_od:
                if not np.shares_memory(od_window, data['od']):
                    od_window /= max_od
        (node_scalers, x_node) = self._standardize_node_features(x_node, train_indices, fit_time_len=scaler_fit_time_len)
        feat_dict = dict()
        if isinstance(x_od, np.ndarray) or isinstance(x_od, torch.Tensor):
            feat_dict['x_od'] = torch.from_numpy(x_od).float().to(params['GPU'])
        elif isinstance(x_od, list):
            feat_dict['x_od'] = x_od
        feat_dict['x_node'] = torch.from_numpy(x_node).float().to(params['GPU'])
        feat_dict['x_SIR'] = torch.from_numpy(x_SIR).float().to(params['GPU'])
        y = torch.from_numpy(y).float().to(params['GPU'])
        main_y = y[:, :self.pred_len, ...]
        reporting_output = None
        if reporting_windows is not None:
            reporting_output = {'target': main_y, 'origin_index': torch.arange(y.shape[0], device=params['GPU'], dtype=torch.int64) + self.obs_len, **{key: torch.from_numpy(value).to(params['GPU']) for (key, value) in reporting_windows.items()}}
            if overlap_target_valid_mask is not None:
                reporting_output['target_valid_mask'] = torch.from_numpy(overlap_target_valid_mask[:, :self.pred_len, ...]).to(params['GPU'])
        stitched_output = {'target': y, 'extra_mask': torch.from_numpy(stitched_extra_mask).to(params['GPU']), 'origin_index': torch.arange(y.shape[0], device=params['GPU']) + self.obs_len}
        if stitched_reporting_windows is not None:
            stitched_output.update({key: torch.from_numpy(value).to(params['GPU']) for (key, value) in stitched_reporting_windows.items() if key != 'origin_index'})
            stitched_output.update({f'shifted_{key}': torch.from_numpy(value).to(params['GPU']) for (key, value) in shifted_reporting_windows.items() if key != 'origin_index'})
            stitched_output['consistency_history_report_mask'] = torch.from_numpy(consistency_history_report_mask).to(params['GPU'])
            prefix_len = self.pred_len - 7
            shifted_target_valid_mask = np.concatenate([np.ones((y.shape[0], prefix_len, 1, 1), dtype=bool), stitched_extra_mask], axis=1)
            if overlap_target_valid_mask is not None:
                shifted_target_valid_mask &= overlap_target_valid_mask[:, 7:7 + self.pred_len, ...]
            stitched_output['shifted_target_valid_mask'] = torch.from_numpy(shifted_target_valid_mask).to(params['GPU'])
        if overlap_target_valid_mask is not None:
            stitched_output['target_valid_mask'] = torch.from_numpy(overlap_target_valid_mask).to(params['GPU'])
        print('Data split:', mode_len)
        data_loader = dict()
        data_loader['max_od'] = max_od
        data_loader['node_scalers'] = node_scalers
        data_loader['_active_calendar_data'] = active_data
        if japan_irregular:
            data_loader['_japan_irregular_manifest'] = irregular_manifest
        data_loader['finite_target_counts'] = {}
        data_modes = [mode for mode in ['train', 'validate', 'adapt_train', 'adapt_val', 'test'] if mode in mode_len]
        requested_data_loader_modes = params.get('_requested_data_loader_modes')
        if requested_data_loader_modes is not None:
            requested_data_loader_modes = set(requested_data_loader_modes)
            data_modes = [mode for mode in data_modes if mode in requested_data_loader_modes]
        for mode in data_modes:
            mode_output = stitched_output if mode in ['train', 'adapt_train'] else reporting_output if reporting_output is not None else main_y
            dataset = ForecastDataset(inputs=feat_dict, output=mode_output, mode=mode, mode_len=mode_len, mode_indices=mode_indices)
            output_shape = dataset.output['target'].shape if isinstance(dataset.output, dict) else dataset.output.shape
            print('Data loader', '|', mode, '|', 'input node features:', dataset.inputs['x_node'].shape, '|output:', output_shape)
            if mode in ['train', 'adapt_train']:
                if mode == 'train':
                    sampler = AuditedEpochPermutationSampler(origins=dataset.output['origin_index'], seed=int(params['random_seed']), audit_path=Path(params['output_dir']) / 'stitched_sampler_permutations.jsonl')
                    data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], sampler=sampler)
                else:
                    data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True)
            else:
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False)
            target_values = dataset.output['target'] if isinstance(dataset.output, dict) else dataset.output
            data_loader['finite_target_counts'][mode] = int(torch.isfinite(target_values[:, :self.pred_len, ...]).sum().item())
        if overlap_contract is not None:
            contract_payload = {**overlap_contract, 'obs_len': int(self.obs_len), 'pred_len': int(self.pred_len), 'finite_target_counts': dict(data_loader['finite_target_counts'])}
            contract_path = Path(params['output_dir']) / 'non_test_overlap_contract.json'
            if contract_path.exists():
                existing = json.loads(contract_path.read_text(encoding='utf-8'))
                if existing != contract_payload:
                    raise RuntimeError(f'non-test-overlap contract changed within one run: {contract_path}')
            else:
                contract_path.write_text(json.dumps(contract_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        return data_loader

    def get_feats(self, data, *, stitched_extra_len=7):
        """Construct FullDynamic windows, including the SRC context."""
        (x_od, x_node, x_SIR, y) = ([], [], [], [])
        context_extra_len = int(stitched_extra_len)
        context_len = self.obs_len + self.pred_len + context_extra_len
        target_len = self.pred_len + context_extra_len
        for i in range(self.obs_len, data['od'].shape[0] - self.pred_len + 1):
            window_start = i - self.obs_len
            window_stop = window_start + context_len
            od_window = self._slice_or_pad(data['od'], window_start, window_stop)
            node_window = self._slice_or_pad(data['node'], window_start, window_stop)
            sir_window = np.concatenate([self._slice_or_pad(data['SIR'], window_start, window_stop), self._slice_or_pad(data.get('sir_documented_y', data['y']), window_start, window_stop)], axis=-1)
            node_window = self._append_modify_amount_features(node_window, od_window, sir_window)
            x_od.append(od_window)
            x_node.append(node_window)
            x_SIR.append(sir_window)
            y.append(self._slice_or_pad(data['y'], i, i + target_len))
        return (x_od, x_node, x_SIR, y)
