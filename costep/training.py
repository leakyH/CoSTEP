import json
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from . import adaptation as adaptor, model as mechanistic_model, data as windowing
from . import training_log as epoch_reporting
from .intervals import interval_event_tensors, interval_normalized_mae_loss, summarize_interval_events, complete_horizon_cumulative_mae, is_report_mae_selection_metric
from .losses import build_elementwise_loss_tensor

def advance_curriculum_task_level(epoch: int, task_level: int, pred_len: int) -> int:
    """Advance curriculum exactly once at an epoch boundary.

    The invalidated implementation called this update from inside the training
    batch loop, making the horizon trajectory depend on the number of batches.
    """
    epoch = int(epoch)
    task_level = int(task_level)
    pred_len = int(pred_len)
    if pred_len < 1:
        raise ValueError(f'pred_len must be positive, got {pred_len}')
    if task_level < 1:
        raise ValueError(f'task_level must be positive, got {task_level}')
    if epoch % 2 == 0 and task_level < pred_len:
        task_level += 1
    return min(task_level, pred_len)

class Trainer(object):

    def __init__(self, params, data, data_container=None, max_od=None):
        self.params = params
        self.raw_data = self.data = data
        self.data_container = data_container
        self.max_od = max_od
        self.training_stage = 'base'
        self.consistency_loss_lambda = float(params['consistency_loss_lambda'])
        self.current_consistency_lambda = self.consistency_loss_lambda
        self.model = self.get_model().to(params['GPU'])
        self.train_criterion = self.get_loss()
        self.test_criterion = nn.L1Loss()
        self.optimizer = self.get_optimizer()
        self.clip = 5
        self._initialize_base_curriculum()
        self.logger = logging.getLogger('costep.' + str(params['output_dir']))
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(Path(params['output_dir']) / 'train.log')
            self.logger.addHandler(handler)
        self.curriculum_horizon_trace = []
        self.curriculum_horizon_ready_epoch = None
        self.curriculum_checkpoint_eligible_epoch = None
        self.active_early_stop_patience = None

    def _initialize_base_curriculum(self) -> None:
        self.cl = not self.params.get('base_full_horizon_training', False)
        self.task_level = 1 if self.cl else int(self.params['pred_len'])

    def _effective_consistency_method(self, mode):
        if not self.params['consistency_enabled']:
            return 'none'
        if self.training_stage == 'post' and mode != 'train':
            return 'none'
        return 'capc_src'

    def _resolve_batch_consistency_lambda(self, main_loss, raw_loss, epoch):
        return self.consistency_loss_lambda

    @staticmethod
    def _unpack_batch(batch):
        (x_od, x_node, x_SIR, y_true) = batch
        if isinstance(y_true, dict):
            y_true = y_true['target']
        return (x_od, x_node, x_SIR, y_true)

    @staticmethod
    def _cross_origin_target_metadata(batch):
        """Return optional metadata carried by the stitched-rollout dataset."""
        target_payload = batch[-1]
        return target_payload if isinstance(target_payload, dict) else {}

    @staticmethod
    def _interval_metadata(target_metadata: dict, prefix: str='') -> dict:
        required = {'future_event_valid', 'future_interval_start', 'future_report_interval'}
        source_keys = {f'{prefix}{key}' for key in required}
        if not source_keys.issubset(target_metadata):
            missing = sorted(source_keys.difference(target_metadata))
            raise ValueError(f'interval target metadata is missing {missing}')
        metadata = {key: target_metadata[f'{prefix}{key}'] for key in required}
        target_valid_mask = target_metadata.get(f'{prefix}target_valid_mask')
        if target_valid_mask is not None:
            event_valid = metadata['future_event_valid']
            if target_valid_mask.shape[1] < event_valid.shape[1]:
                raise ValueError(f'target_valid_mask horizon is shorter than interval metadata: {target_valid_mask.shape} vs {event_valid.shape}')
            target_valid_mask = target_valid_mask[:, :event_valid.shape[1], ...]
            try:
                target_valid_mask = torch.broadcast_to(target_valid_mask, event_valid.shape)
            except RuntimeError as error:
                raise ValueError(f'target_valid_mask cannot broadcast to interval metadata: {target_valid_mask.shape} vs {event_valid.shape}') from error
            metadata['target_valid_mask'] = target_valid_mask
            metadata['future_event_valid'] = event_valid.bool() & target_valid_mask.bool()
        return metadata

    def _interval_stitched_rollout_loss(self, prediction: torch.Tensor, target: torch.Tensor, target_metadata: dict, *, prefix: str='', origin_mask: torch.Tensor | None=None) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute loss for one shifted window over complete reporting intervals."""
        metadata = self._interval_metadata(target_metadata, prefix=prefix)
        if origin_mask is not None:
            origin_mask = origin_mask.to(device=prediction.device, dtype=torch.bool)
            target_valid_mask = metadata.get('target_valid_mask')
            metadata['target_valid_mask'] = origin_mask if target_valid_mask is None else target_valid_mask.to(device=prediction.device, dtype=torch.bool) & origin_mask
            metadata['future_event_valid'] = metadata['future_event_valid'].to(device=prediction.device, dtype=torch.bool) & origin_mask
        (loss, events) = interval_normalized_mae_loss(prediction, target, metadata)
        if loss is None:
            loss = prediction.sum() * 0.0
        return (loss, events, metadata)

    def _curriculum_is_active(self) -> bool:
        return bool(self.cl)

    def _advance_curriculum_epoch(self, epoch: int) -> int:
        """Advance only at epoch boundaries; disabled CL uses the full horizon."""
        if self._curriculum_is_active():
            self.task_level = advance_curriculum_task_level(epoch, self.task_level, self.params['pred_len'])
        return min(int(self.task_level), int(self.params['pred_len']))

    def _curriculum_horizon_ready(self) -> bool:
        return bool(not self._curriculum_is_active() or int(self.task_level) >= int(self.params['pred_len']))

    @staticmethod
    def _build_cross_origin_stitched_params(base_beta: torch.Tensor, base_gamma: torch.Tensor, shifted_beta: torch.Tensor, shifted_gamma: torch.Tensor, shift: int, *, detach_boundary_anchors: bool=False):
        """Splice two parameter trajectories whose forecast origins differ by ``shift``.

        For shift=7 and 28-day trajectories the original method constructs
        ``left=[A0|B7|C7|D7]`` and ``right=[B0|C0|D0|E7]``.  The detached-anchor
        variant instead uses ``stopgrad(A0)`` and ``stopgrad(E7)``.  Overlap
        tensors returned below always remain attached so the direct L1 term
        still updates both encoder evaluations.
        """
        trajectories = (base_beta, base_gamma, shifted_beta, shifted_gamma)
        if any((tensor.dim() < 2 for tensor in trajectories)):
            raise ValueError('cross-origin parameters must include a time dimension')
        trajectory_length = base_beta.shape[1]
        if any((tensor.shape != base_beta.shape for tensor in trajectories[1:])):
            raise ValueError('base and shifted beta/gamma trajectories must have identical shapes')
        if shift < 1 or shift >= trajectory_length:
            raise ValueError(f'cross-origin shift must be in [1, {trajectory_length - 1}], got {shift}')
        overlap_length = trajectory_length - shift
        left_anchor_beta = base_beta[:, :shift, ...]
        left_anchor_gamma = base_gamma[:, :shift, ...]
        right_anchor_beta = shifted_beta[:, overlap_length:, ...]
        right_anchor_gamma = shifted_gamma[:, overlap_length:, ...]
        if detach_boundary_anchors:
            left_anchor_beta = left_anchor_beta.detach()
            left_anchor_gamma = left_anchor_gamma.detach()
            right_anchor_beta = right_anchor_beta.detach()
            right_anchor_gamma = right_anchor_gamma.detach()
        left_beta = torch.cat([left_anchor_beta, shifted_beta[:, :overlap_length, ...]], dim=1)
        left_gamma = torch.cat([left_anchor_gamma, shifted_gamma[:, :overlap_length, ...]], dim=1)
        right_beta = torch.cat([base_beta[:, shift:, ...], right_anchor_beta], dim=1)
        right_gamma = torch.cat([base_gamma[:, shift:, ...], right_anchor_gamma], dim=1)
        return {'left_beta': left_beta, 'left_gamma': left_gamma, 'right_beta': right_beta, 'right_gamma': right_gamma, 'base_overlap_beta': base_beta[:, shift:, ...], 'base_overlap_gamma': base_gamma[:, shift:, ...], 'shifted_overlap_beta': shifted_beta[:, :overlap_length, ...], 'shifted_overlap_gamma': shifted_gamma[:, :overlap_length, ...]}

    def _cross_origin_curriculum_horizon(self, mode: str) -> int:
        """Return the forecast support currently open to stitched objectives."""
        pred_len = int(self.params['pred_len'])
        if not self._mode_is_train(mode) or not getattr(self, 'cl', False):
            return pred_len
        return min(max(1, int(self.task_level)), pred_len)

    @staticmethod
    def _slice_cross_origin_parameter_horizon(base_beta: torch.Tensor, base_gamma: torch.Tensor, shifted_beta: torch.Tensor, shifted_gamma: torch.Tensor, *, obs_len: int, forecast_horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep all history parameters and only the active forecast prefix."""
        trajectories = (base_beta, base_gamma, shifted_beta, shifted_gamma)
        if any((tensor.shape != base_beta.shape for tensor in trajectories[1:])):
            raise ValueError('base and shifted beta/gamma trajectories must have identical shapes')
        active_length = int(obs_len) + int(forecast_horizon)
        if active_length < 1 or active_length > base_beta.shape[1]:
            raise ValueError(f'active parameter trajectory length must be within the encoded trajectory, got {active_length} for {base_beta.shape[1]} positions')
        return tuple((tensor[:, :active_length, ...] for tensor in trajectories))

    @staticmethod
    def _capc_overlap_components(base_beta: torch.Tensor, base_gamma: torch.Tensor, shifted_beta: torch.Tensor, shifted_gamma: torch.Tensor, offset: int, beta_weight: float=1.0, gamma_weight: float=1.0) -> dict:
        """Return CAPC parameter-overlap terms for one origin shift."""
        trajectories = (base_beta, base_gamma, shifted_beta, shifted_gamma)
        if any((tensor.dim() < 2 for tensor in trajectories)):
            raise ValueError('consistency parameters must include a time dimension')
        if any((tensor.shape != base_beta.shape for tensor in trajectories[1:])):
            raise ValueError('base and shifted beta/gamma trajectories must have identical shapes')
        if offset < 1 or offset >= base_beta.shape[1]:
            raise ValueError('consistency offset must be within the parameter trajectory')
        beta_weight = float(beta_weight)
        gamma_weight = float(gamma_weight)
        if beta_weight < 0 or gamma_weight < 0:
            raise ValueError('consistency beta/gamma weights must be non-negative')
        if abs(beta_weight + gamma_weight - 2.0) > 1e-08:
            raise ValueError('consistency beta/gamma weights must sum to 2')
        beta_raw = F.l1_loss(base_beta[:, offset:, ...], shifted_beta[:, :-offset, ...])
        gamma_raw = F.l1_loss(base_gamma[:, offset:, ...], shifted_gamma[:, :-offset, ...])
        return {'beta_raw': beta_raw, 'gamma_raw': gamma_raw, 'beta_weighted': beta_raw * beta_weight, 'gamma_weighted': gamma_raw * gamma_weight, 'weighted_total': beta_raw * beta_weight + gamma_raw * gamma_weight}

    @staticmethod
    def _capc_overlap_audit(base_beta: torch.Tensor, base_gamma: torch.Tensor, shifted_beta: torch.Tensor, shifted_gamma: torch.Tensor, offset: int, beta_weight: float=1.0, gamma_weight: float=1.0) -> dict:
        beta_error = torch.abs(base_beta[:, offset:, ...] - shifted_beta[:, :-offset, ...])
        gamma_error = torch.abs(base_gamma[:, offset:, ...] - shifted_gamma[:, :-offset, ...])
        beta_mean = beta_error.mean()
        gamma_mean = gamma_error.mean()
        return {'beta_numerator': float(beta_error.detach().sum().item()), 'beta_denominator': int(beta_error.numel()), 'gamma_numerator': float(gamma_error.detach().sum().item()), 'gamma_denominator': int(gamma_error.numel()), 'beta_raw': float(beta_mean.detach().item()), 'gamma_raw': float(gamma_mean.detach().item()), 'beta_weight': float(beta_weight), 'gamma_weight': float(gamma_weight), 'beta_weighted': float((beta_mean * float(beta_weight)).detach().item()), 'gamma_weighted': float((gamma_mean * float(gamma_weight)).detach().item()), 'mean': float((beta_mean * float(beta_weight) + gamma_mean * float(gamma_weight)).detach().item())}

    @staticmethod
    def _masked_elementwise_mean(loss_tensor: torch.Tensor, mask: torch.Tensor | None):
        finite_mask = torch.isfinite(loss_tensor)
        if mask is None:
            expanded_mask = finite_mask
        else:
            mask = mask.to(device=loss_tensor.device, dtype=torch.bool)
            try:
                expanded_mask = torch.broadcast_to(mask, loss_tensor.shape)
            except RuntimeError as exc:
                raise ValueError(f'loss mask {tuple(mask.shape)} cannot broadcast to loss {tuple(loss_tensor.shape)}') from exc
            expanded_mask = expanded_mask & finite_mask
        safe_loss = torch.where(expanded_mask, loss_tensor, torch.zeros_like(loss_tensor))
        denominator = expanded_mask.sum()
        if denominator.item() <= 0:
            return safe_loss.sum() * 0.0
        return safe_loss.sum() / denominator

    @staticmethod
    def _finite_target_count(target: torch.Tensor) -> int:
        return int(torch.isfinite(target).sum().item())

    def _loader_finite_target_count(self, data_loader: dict, mode: str) -> int:
        cached = data_loader.get('finite_target_counts', {}).get(mode)
        if cached is not None:
            return int(cached)
        try:
            return adaptor.count_finite_targets(data_loader[mode], int(self.params['pred_len']))
        except (KeyError, TypeError):
            return int(len(data_loader[mode].dataset))

    @staticmethod
    def _masked_loss_audit(loss_tensor: torch.Tensor, mask: torch.Tensor) -> dict:
        expanded_mask = torch.broadcast_to(mask.to(device=loss_tensor.device, dtype=loss_tensor.dtype), loss_tensor.shape)
        numerator = (loss_tensor * expanded_mask).sum()
        denominator = expanded_mask.sum()
        mean = numerator / denominator if denominator.item() > 0 else loss_tensor.sum() * 0.0
        return {'numerator': float(numerator.detach().item()), 'denominator': int(denominator.detach().item()), 'mean': float(mean.detach().item())}

    @staticmethod
    def _interval_event_loss_audit(events: dict[str, torch.Tensor], loss: torch.Tensor) -> dict:
        normalized_errors = torch.abs(events['predicted_count'] - events['observed_count']) / events['interval']
        return {'numerator': float(normalized_errors.detach().sum().item()), 'denominator': int(normalized_errors.numel()), 'mean': float(loss.detach().item()), 'unit': 'report_event'}

    def _stitched_common_support_mask(self, target_metadata: dict, loss_tensor: torch.Tensor) -> torch.Tensor | None:
        max_shift = None
        return None

    def _append_stitched_common_support_audit(self, *, epoch: int, batch_index: int, main_loss_tensor: torch.Tensor, bundle: dict, target_metadata: dict) -> None:
        audit = bundle.get('common_support_audit')
        if audit is None:
            return
        origins = target_metadata['origin_index'].detach().cpu().reshape(-1).tolist()
        support = audit['support_mask']
        record = {'epoch': int(epoch), 'batch_index': int(batch_index), 'absolute_origins': [int(value) for value in origins], 'common_support_rows': [bool(value) for value in support], 'main_reduction_full_batch': self._masked_loss_audit(main_loss_tensor, torch.ones((main_loss_tensor.shape[0],) + (1,) * max(0, main_loss_tensor.dim() - 1), device=main_loss_tensor.device, dtype=torch.bool)), 'capc_reduction_full_batch': float(bundle['raw_overlap'].detach().item()), 'capc_scaled_full_batch': float(bundle['scaled_overlap'].detach().item()), 'consistency_lambda': float(bundle['consistency_lambda']), 'capc_shift_reductions_full_batch': audit['capc_shift_reductions'], 'policy': audit['policy'], 'left_time_mask': audit['left_time_mask'], 'right_time_mask': audit['right_time_mask'], 'left': audit['left'], 'right': audit['right'], 'stitched_scaled_total': float((bundle['left_scaled'] + bundle['right_scaled']).detach().item()), 'consistency_scaled_total': float(bundle['consistency_scaled_total'].detach().item())}
        path = Path(self.params['output_dir']) / 'stitched_common_support_batches.jsonl'
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')

    def _cross_origin_right_mask(self, target_metadata: dict, batch_size: int, dtype: torch.dtype, device: torch.device, horizon: int | None=None):
        """Build the mask for y[shift:shift+pred_len].

        The first ``pred_len-shift`` dates are part of the original 14-day
        target and are always valid.  ``extra_mask`` gates only the appended
        tail labels, preventing a training origin from reading across a split
        boundary.
        """
        shift = 7
        pred_len = int(self.params['pred_len'])
        horizon = pred_len if horizon is None else int(horizon)
        if 7 > pred_len:
            raise ValueError('cross_origin_shift cannot exceed pred_len')
        if horizon < 1 or horizon > pred_len:
            raise ValueError('stitched curriculum horizon must be within pred_len')
        prefix_length = min(horizon, pred_len - 7)
        tail_length = horizon - prefix_length
        extra_mask = target_metadata.get('extra_mask')
        if extra_mask is None:
            return None
        if extra_mask.shape[0] != batch_size or extra_mask.shape[1] != 7:
            raise ValueError(f'extra_mask must have shape [batch, cross_origin_shift, ...], got {tuple(extra_mask.shape)}')
        prefix_shape = (batch_size, prefix_length) + tuple(extra_mask.shape[2:])
        prefix = torch.ones(prefix_shape, device=device, dtype=dtype)
        tail = extra_mask[:, :tail_length, ...].to(device=device, dtype=dtype)
        return torch.cat([prefix, tail], dim=1)

    def _rollout_full_family_from_params(self, x_SIR: torch.Tensor, x_od: torch.Tensor, param_beta: torch.Tensor, param_gamma: torch.Tensor, *, apply_adaptor: bool, report_mask: torch.Tensor | None=None):
        """Decode an externally assembled beta/gamma trajectory without re-encoding."""
        expected_param_length = int(self.params['obs_len']) + int(self.params['pred_len'])
        if param_beta.shape[1] != expected_param_length or param_gamma.shape[1] != expected_param_length:
            raise ValueError(f'stitched beta/gamma must cover obs_len + pred_len dates; expected {expected_param_length}, got beta={param_beta.shape[1]}, gamma={param_gamma.shape[1]}')
        latency = self.model.get_latency_distribution(apply_adaptor=apply_adaptor)
        return self.model._rollout_full_family(SIR=x_SIR, od=x_od, max_od=self.max_od, param_b=param_beta, param_g=param_gamma, inverse_norm_infect_detact_latency=latency, report_mask=report_mask)

    @staticmethod
    def _cross_origin_history_report_mask(target_metadata: dict, *, offset: int, obs_len: int) -> torch.Tensor | None:
        """Return the causal history mask for one cross-origin encoder/rollout."""
        offset = int(offset)
        obs_len = int(obs_len)
        base_mask = target_metadata.get('history_report_mask')
        if offset == 0:
            selected = base_mask
        else:
            stacked = target_metadata.get('consistency_history_report_mask')
            if stacked is None:
                if base_mask is None:
                    return None
                raise ValueError('cross-origin reporting payload is missing consistency_history_report_mask')
            if offset < 1 or offset > stacked.shape[1]:
                raise ValueError(f'cross-origin report-mask offset {offset} is outside the available range 1..{stacked.shape[1]}')
            selected = stacked[:, offset - 1, ...]
        if selected is None:
            return None
        if selected.dim() != 4 or selected.shape[1] < obs_len:
            raise ValueError(f'history report mask must have shape [batch, time, node, 1] with at least {obs_len} dates, got {tuple(selected.shape)}')
        return selected[:, :obs_len, ...]

    @contextmanager
    def _preserve_recorded_model_state(self):
        """Restore main-forward diagnostics after auxiliary encoder/decoder calls."""
        recorded = {name: value for (name, value) in vars(self.model).items() if name.startswith('recorded_')}
        try:
            yield
        finally:
            for name in tuple(vars(self.model)):
                if name.startswith('recorded_') and name not in recorded:
                    delattr(self.model, name)
            for (name, value) in recorded.items():
                setattr(self.model, name, value)

    @staticmethod
    def _assemble_cross_origin_objective_components(*, raw_overlap: torch.Tensor, consistency_lambda: float, left_raw: torch.Tensor, left_weight: float, right_raw: torch.Tensor, right_weight: float) -> dict:
        """Scale and combine the cross-origin terms without side effects."""
        scaled_overlap = raw_overlap * float(consistency_lambda)
        left_scaled = left_raw * float(left_weight)
        right_scaled = right_raw * float(right_weight)
        consistency_scaled_total = scaled_overlap + left_scaled + right_scaled
        return {'raw_overlap': raw_overlap, 'scaled_overlap': scaled_overlap, 'consistency_lambda': float(consistency_lambda), 'left_raw': left_raw, 'left_scaled': left_scaled, 'right_raw': right_raw, 'right_scaled': right_scaled, 'consistency_scaled_total': consistency_scaled_total, 'auxiliary_scaled_total': consistency_scaled_total}

    def _cross_origin_forward_loss_bundle(self, *, mode: str, consistency_method: str, x_node: torch.Tensor, x_SIR: torch.Tensor, x_od_device: torch.Tensor, y_true: torch.Tensor, target_metadata: dict, main_loss: torch.Tensor, apply_adaptor: bool, epoch: int) -> dict:
        """Run the canonical cross-origin forwards and return pure loss terms.

        This is the single orchestration path shared by optimization and
        train-only diagnostics.  It performs no backward, optimizer, logging,
        checkpoint, validation, or test action.
        """
        if consistency_method not in {'capc_src'}:
            raise ValueError('cross-origin forward/loss bundle requires a stitched method')
        shift = 7
        obs_len = int(self.params['obs_len'])
        pred_len = int(self.params['pred_len'])
        curriculum_horizon = self._cross_origin_curriculum_horizon(mode)
        required_context = obs_len + pred_len + 7
        required_target = pred_len + 7
        if x_node.shape[1] < required_context:
            raise ValueError(f'cross-origin stitched rollout requires obs_len + pred_len + shift input dates ({required_context}), got {x_node.shape[1]}')
        if y_true.shape[1] < required_target:
            raise ValueError(f'cross-origin stitched rollout requires pred_len + shift target dates ({required_target}), got {y_true.shape[1]}')
        base_beta = self._consistency_param_view(self.model.recorded_beta)
        base_gamma = self._consistency_param_view(self.model.recorded_gamma)
        base_report_mask = self._cross_origin_history_report_mask(target_metadata, offset=0, obs_len=obs_len)
        shifted_report_mask = self._cross_origin_history_report_mask(target_metadata, offset=7, obs_len=obs_len)
        shifted_reporting_kwargs = {}
        if shifted_report_mask is not None:
            shifted_reporting_kwargs['report_mask'] = shifted_report_mask
        with self._preserve_recorded_model_state():
            (shifted_beta, shifted_gamma, _) = self.model(x_node[:, 7:7 + obs_len, ...].transpose(1, 3), x_SIR[:, 7:7 + obs_len, ...], x_od_device[:, 7:7 + obs_len, ...], self.max_od, only_params=True, apply_adaptor=apply_adaptor, **shifted_reporting_kwargs)
        shifted_beta = self._consistency_param_view(shifted_beta)
        shifted_gamma = self._consistency_param_view(shifted_gamma)
        stitched_params = self._build_cross_origin_stitched_params(base_beta, base_gamma, shifted_beta, shifted_gamma, 7, detach_boundary_anchors=False)
        with self._preserve_recorded_model_state():
            left_prediction = self._rollout_full_family_from_params(x_SIR[:, :obs_len, ...], x_od_device[:, :obs_len, ...], stitched_params['left_beta'], stitched_params['left_gamma'], apply_adaptor=apply_adaptor, report_mask=base_report_mask)
        with self._preserve_recorded_model_state():
            right_prediction = self._rollout_full_family_from_params(x_SIR[:, 7:7 + obs_len, ...], x_od_device[:, 7:7 + obs_len, ...], stitched_params['right_beta'], stitched_params['right_gamma'], apply_adaptor=apply_adaptor, report_mask=shifted_report_mask)
        interval_stitched = self._uses_report_loss()
        common_support_mask = self._stitched_common_support_mask(target_metadata, left_prediction)
        left_interval_events = None
        right_interval_events = None
        left_interval_metadata = None
        right_interval_metadata = None
        if interval_stitched:
            curriculum_time_mask = (torch.arange(pred_len, device=left_prediction.device) < curriculum_horizon).reshape(1, pred_len, 1, 1)
            left_interval_mask = curriculum_time_mask
            right_interval_mask = curriculum_time_mask
            if common_support_mask is not None:
                left_interval_mask = left_interval_mask & common_support_mask
                right_interval_mask = right_interval_mask & common_support_mask
            (left_raw, left_interval_events, left_interval_metadata) = self._interval_stitched_rollout_loss(left_prediction, y_true[:, :pred_len, ...], target_metadata, origin_mask=left_interval_mask)
            (right_raw, right_interval_events, right_interval_metadata) = self._interval_stitched_rollout_loss(right_prediction, y_true[:, 7:7 + pred_len, ...], target_metadata, prefix='shifted_', origin_mask=right_interval_mask)
            right_mask = None
        else:
            left_loss_tensor = self._prediction_loss_tensor(left_prediction[:, :curriculum_horizon, ...], y_true[:, :curriculum_horizon, ...])
            left_raw = self._reduce_prediction_loss(left_loss_tensor, common_support_mask)
            right_loss_tensor = self._prediction_loss_tensor(right_prediction[:, :curriculum_horizon, ...], y_true[:, 7:7 + curriculum_horizon, ...])
            right_mask = self._cross_origin_right_mask(target_metadata, y_true.shape[0], right_loss_tensor.dtype, right_loss_tensor.device, horizon=curriculum_horizon)
            if common_support_mask is not None:
                right_mask = common_support_mask if right_mask is None else right_mask.to(dtype=torch.bool) & common_support_mask
            right_raw = self._reduce_prediction_loss(right_loss_tensor, right_mask)
        beta_consistency_weight = float(self.params.get('consistency_beta_weight', 1.0))
        gamma_consistency_weight = float(self.params.get('consistency_gamma_weight', 1.0))
        (active_base_beta, active_base_gamma, active_shifted_beta, active_shifted_gamma) = self._slice_cross_origin_parameter_horizon(base_beta, base_gamma, shifted_beta, shifted_gamma, obs_len=obs_len, forecast_horizon=curriculum_horizon)
        shifted_parameters = {7: (active_shifted_beta, active_shifted_gamma)}
        for idx_offset in range(1, pred_len // 2 + 1):
            if idx_offset == 7:
                continue
            next_report_mask = self._cross_origin_history_report_mask(target_metadata, offset=idx_offset, obs_len=obs_len)
            next_reporting_kwargs = {}
            if next_report_mask is not None:
                next_reporting_kwargs['report_mask'] = next_report_mask
            with self._preserve_recorded_model_state():
                (next_beta, next_gamma, _) = self.model(x_node[:, idx_offset:idx_offset + obs_len, ...].transpose(1, 3), x_SIR[:, idx_offset:idx_offset + obs_len, ...], x_od_device[:, idx_offset:idx_offset + obs_len, ...], self.max_od, only_params=True, apply_adaptor=apply_adaptor, **next_reporting_kwargs)
            next_beta = self._consistency_param_view(next_beta)
            next_gamma = self._consistency_param_view(next_gamma)
            (_, _, next_beta, next_gamma) = self._slice_cross_origin_parameter_horizon(base_beta, base_gamma, next_beta, next_gamma, obs_len=obs_len, forecast_horizon=curriculum_horizon)
            shifted_parameters[idx_offset] = (next_beta, next_gamma)
        capc_shift_components = {idx_offset: self._capc_overlap_components(active_base_beta, active_base_gamma, shifted_parameters[idx_offset][0], shifted_parameters[idx_offset][1], idx_offset, beta_weight=beta_consistency_weight, gamma_weight=gamma_consistency_weight) for idx_offset in range(1, pred_len // 2 + 1)}
        capc_shift_losses = {idx_offset: components['weighted_total'] for (idx_offset, components) in capc_shift_components.items()}
        capc_shift_audits = {idx_offset: self._capc_overlap_audit(active_base_beta, active_base_gamma, shifted_parameters[idx_offset][0], shifted_parameters[idx_offset][1], idx_offset, beta_weight=beta_consistency_weight, gamma_weight=gamma_consistency_weight) for idx_offset in range(1, pred_len // 2 + 1)}
        raw_overlap = torch.stack(list(capc_shift_losses.values())).sum()
        raw_overlap_beta = torch.stack([components['beta_raw'] for components in capc_shift_components.values()]).sum()
        raw_overlap_gamma = torch.stack([components['gamma_raw'] for components in capc_shift_components.values()]).sum()
        consistency_lambda = self._resolve_batch_consistency_lambda(main_loss, raw_overlap, epoch)
        bundle = self._assemble_cross_origin_objective_components(raw_overlap=raw_overlap, consistency_lambda=consistency_lambda, left_raw=left_raw, left_weight=float(self.params.get('stitched_left_loss_weight', 1.0)), right_raw=right_raw, right_weight=float(self.params.get('stitched_right_loss_weight', 1.0)))
        bundle.update({'curriculum_horizon': int(curriculum_horizon), 'parameter_active_length': int(obs_len + curriculum_horizon), 'raw_overlap_beta': raw_overlap_beta, 'raw_overlap_gamma': raw_overlap_gamma, 'scaled_overlap_beta': raw_overlap_beta * float(self.params.get('consistency_beta_weight', 1.0)) * float(consistency_lambda), 'scaled_overlap_gamma': raw_overlap_gamma * float(self.params.get('consistency_gamma_weight', 1.0)) * float(consistency_lambda), 'left_report_event_count': int(left_interval_events['predicted_count'].numel()) if left_interval_events is not None else None, 'right_report_event_count': int(right_interval_events['predicted_count'].numel()) if right_interval_events is not None else None})
        bundle['gradient_terms'] = [bundle['scaled_overlap'], bundle['left_scaled'], bundle['right_scaled']]
        if interval_stitched:
            left_mask = left_interval_metadata.get('target_valid_mask')
            if left_mask is None:
                left_mask = torch.ones((left_prediction.shape[0], pred_len, 1, 1), device=left_prediction.device, dtype=torch.bool)
            right_mask = right_interval_metadata.get('target_valid_mask')
            if right_mask is None:
                right_mask = torch.ones((right_prediction.shape[0], pred_len, 1, 1), device=right_prediction.device, dtype=torch.bool)
            left_mask = torch.broadcast_to(left_mask.to(device=left_prediction.device, dtype=torch.bool), left_prediction.shape)
            right_mask = torch.broadcast_to(right_mask.to(device=right_prediction.device, dtype=torch.bool), right_prediction.shape)
            left_audit = self._interval_event_loss_audit(left_interval_events, left_raw)
            right_audit = self._interval_event_loss_audit(right_interval_events, right_raw)
            left_loss_shape = left_prediction.shape
            right_loss_shape = right_prediction.shape
        else:
            left_mask = common_support_mask if common_support_mask is not None else torch.ones((left_loss_tensor.shape[0], 1, 1, 1), device=left_loss_tensor.device, dtype=torch.bool)
            if right_mask is None:
                right_mask = torch.ones((right_loss_tensor.shape[0], 1, 1, 1), device=right_loss_tensor.device, dtype=torch.bool)
            left_audit = self._masked_loss_audit(left_loss_tensor, left_mask)
            right_audit = self._masked_loss_audit(right_loss_tensor, right_mask)
            left_loss_shape = left_loss_tensor.shape
            right_loss_shape = right_loss_tensor.shape
        support_rows = left_mask.detach().cpu().reshape(-1).tolist()
        if interval_stitched:
            support_rows = (torch.ones(left_prediction.shape[0], dtype=torch.bool) if common_support_mask is None else common_support_mask.detach().cpu().reshape(-1)).tolist()
        left_audit['scaled'] = float(bundle['left_scaled'].detach().item())
        right_audit['scaled'] = float(bundle['right_scaled'].detach().item())
        bundle['common_support_audit'] = {'policy': 'common_complete_origin' if common_support_mask is not None else 'native_boundary', 'support_mask': support_rows, 'left_time_mask': torch.broadcast_to(left_mask, left_loss_shape)[:, :, 0, 0].detach().cpu().bool().tolist(), 'right_time_mask': torch.broadcast_to(right_mask, right_loss_shape)[:, :, 0, 0].detach().cpu().bool().tolist(), 'left': left_audit, 'right': right_audit, 'capc_shift_reductions': {str(offset): capc_shift_audits[offset] for offset in capc_shift_audits}, 'parameter_overlap_totals': {'beta_raw': float(raw_overlap_beta.detach().item()), 'gamma_raw': float(raw_overlap_gamma.detach().item()), 'beta_scaled': float(bundle['scaled_overlap_beta'].detach().item()), 'gamma_scaled': float(bundle['scaled_overlap_gamma'].detach().item()), 'beta_weight': float(self.params.get('consistency_beta_weight', 1.0)), 'gamma_weight': float(self.params.get('consistency_gamma_weight', 1.0))}}
        return bundle

    def _mode_uses_adaptor(self, mode: str) -> bool:
        if not adaptor.model_has_adaptor(self.model):
            return False
        return adaptor.mode_uses_adaptor(mode, stage='post')

    def _mode_is_train(self, mode: str) -> bool:
        return adaptor.mode_is_train(mode)

    def _validation_modes(self, modes: list) -> list:
        if 'all_adaptor' in adaptor.REPLAY_POST_TRAIN_MODES:
            validation_modes = [mode for mode in modes if mode in ['validate', 'adapt_val']]
            if validation_modes:
                return validation_modes
        if 'adapt_val' in modes:
            return ['adapt_val']
        if 'validate' in modes:
            return ['validate']
        return []

    @staticmethod
    def _mode_loader_has_batches(data_loader: dict, mode: str) -> bool:
        loader = data_loader.get(mode)
        if loader is None:
            return False
        try:
            return len(loader) > 0
        except TypeError:
            return True

    def _combine_validation_losses(self, validation_loss_items, data_loader):
        """Combine split-level losses according to the post-training protocol."""
        losses = np.asarray([float(loss) for (_, loss) in validation_loss_items], dtype=float)
        weights = np.ones(len(validation_loss_items), dtype=float)
        if np.any(weights <= 0):
            raise ValueError('Validation aggregation requires positive split lengths')
        weights = weights / weights.sum()
        combined = float(np.dot(losses, weights))
        return (combined, {mode: float(weight) for ((mode, _), weight) in zip(validation_loss_items, weights)})

    def _train_loss_scales(self, data_loader: dict, modes: list) -> dict:
        train_modes = [mode for mode in modes if self._mode_is_train(mode)]
        if 'all_adaptor' not in adaptor.REPLAY_POST_TRAIN_MODES or len(train_modes) <= 1:
            return {mode: 1.0 for mode in train_modes}
        batch_counts = {mode: max(1, len(data_loader[mode])) for mode in train_modes}
        total_batches = sum(batch_counts.values())
        return {mode: total_batches / (len(train_modes) * batch_count) for (mode, batch_count) in batch_counts.items()}

    def _consistency_param_view(self, param: torch.Tensor):
        return param

    def get_checkpoint_path(self, stage=None, exp_name=None, checkpoint_suffix=None):
        stage = stage or ('base' if checkpoint_suffix == 'base' else 'post')
        return str(Path(self.params['output_dir']) / (stage + '.pt'))

    def _load_model_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.params['GPU'], weights_only=True)
        if checkpoint.get('format_version') != 1:
            raise ValueError('Unsupported CoSTEP checkpoint format')
        if checkpoint['data_identity'] != self.params['data_identity']:
            raise ValueError('Checkpoint data identity mismatch')
        config = dict(checkpoint['config'])
        current = dict(self.params['release_config'])
        runtime = {'device', 'output', 'epochs', 'post_epochs', 'patience', 'post_patience', 'post_start_epoch', 'threads', 'post'}
        if {k: v for (k, v) in config.items() if k not in runtime} != {k: v for (k, v) in current.items() if k not in runtime}:
            raise ValueError('Checkpoint scientific configuration mismatch')
        if checkpoint['stage'] == 'post':
            adaptor.ensure_linear_param_adaptor(self.model)
            adaptor.enable_adaptor_latency_finetune(self.model)
        self.model.load_state_dict(checkpoint['state_dict'])
        return checkpoint

    def _checkpoint_state(self, epoch):
        return {'format_version': 1, 'epoch': epoch, 'stage': self.training_stage, 'state_dict': self.model.state_dict(), 'config': self.params['release_config'], 'data_identity': self.params['data_identity'], 'epoch_mode_losses': dict(getattr(self, 'last_epoch_mode_losses', {}))}

    def run_post_training(self, base_data_loader, checkpoint_path):
        checkpoint = self._load_model_checkpoint(checkpoint_path)
        if checkpoint['stage'] != 'base':
            raise ValueError('Post training requires a base checkpoint')
        self.training_stage = 'post'
        self.cl = False
        self.task_level = self.params['pred_len']
        info = adaptor.configure_adaptor_training(self.model, 'all_adaptor', 'finetune')
        self.optimizer = optim.Adam([p for (_, p) in info['trainable_params']], lr=self.params['post_train_lr'], weight_decay=self.params['weight_decay'])
        post_loaders = adaptor.build_adaptor_tail_loader(base_data_loader, self.params['batch_size'], evaluation=False, tail_len=10, adapt_val_len=2, pred_len=self.params['pred_len'], min_val_label_len=14, include_short_in_adapt_train=False, embargo_len=0)
        audit = adaptor.build_post_split_audit(post_loaders, obs_len=14, pred_len=14)
        (Path(self.params['output_dir']) / 'post_split_audit.json').write_text(json.dumps(audit, indent=2))
        before = Path(checkpoint_path).read_bytes()
        old_epochs = self.params['num_epochs']
        self.params['num_epochs'] = self.params['post_train_epochs']
        try:
            self.train(post_loaders, ['train', 'adapt_train', 'validate', 'adapt_val'], early_stop_patience=self.params['post_train_patience'], checkpoint_path=self.get_checkpoint_path(stage='post'), epoch_offset=checkpoint['epoch'])
        finally:
            self.params['num_epochs'] = old_epochs
        if Path(checkpoint_path).read_bytes() != before:
            raise RuntimeError('Base checkpoint changed during adaptation')
        if not Path(self.get_checkpoint_path(stage='post')).exists():
            raise RuntimeError('No eligible post checkpoint; increase post_epochs or lower post_start_epoch')
        return post_loaders

    def get_model(self):
        return mechanistic_model.CoSTEP(num_nodes=self.raw_data['node'].shape[1], in_dim=self.raw_data['node'].shape[-1] + 2 + (2 if self.params['reporting_mode'] == 'interval' else 0), in_len=self.params['obs_len'], out_len=self.params['pred_len'], blocks=self.params['obs_len'] // 7, layers=3, dropout=0.5, stock_estimator=self.params['stock_estimator'], curve=self.params.get('curve', True))

    def _uses_report_loss(self):
        return self.params.get('loss') == 'report_mae' or self.params.get('reporting_mode') == 'interval'

    def get_loss(self):
        if self.params['loss'] == 'report_mae':
            return None
        if self.params['loss'] == 'MAE':
            return nn.L1Loss()
        raise ValueError('CoSTEP supports MAE or report_mae')

    def _prediction_loss_tensor(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.params.get('loss') == 'report_mae':
            raise RuntimeError('report_mae must use reporting metadata, not elementwise MAE')
        return build_elementwise_loss_tensor(prediction, target, self.params['loss'])

    def _reduce_prediction_loss(self, loss_tensor: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        reduced = self._masked_elementwise_mean(loss_tensor, mask)
        return reduced

    def get_optimizer(self):
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.params['learn_rate'], weight_decay=self.params['weight_decay'])
        return optimizer

    def train(self, data_loader: dict, modes: list, early_stop_patience=20, checkpoint_path=None, epoch_offset=0):
        training_started_at = datetime.now()
        checkpoint = self._checkpoint_state(0)
        val_loss = np.inf
        patience_count = early_stop_patience
        checkpoint_path = checkpoint_path or self.get_checkpoint_path(stage='base')
        self.curriculum_horizon_trace = []
        self.curriculum_horizon_ready_epoch = None
        self.curriculum_checkpoint_eligible_epoch = None
        self.active_early_stop_patience = int(early_stop_patience)
        self.logger.info(f"{'CoSTEP'} model training begins:")
        self._epoch_offset_for_temp = epoch_offset
        train_loss_scales = self._train_loss_scales(data_loader, modes)
        active_train_modes = [mode for mode in modes if self._mode_is_train(mode) and self._mode_loader_has_batches(data_loader, mode)]
        if len(active_train_modes) > 1:
            scaled_batch_totals = {mode: len(data_loader[mode]) * train_loss_scales[mode] for mode in active_train_modes}
            scale_total = sum(scaled_batch_totals.values())
            strategy = 'balanced'
            self.logger.info(f'Post-training optimization uses {strategy} split aggregation: ' + ', '.join((f'{mode}(n={len(data_loader[mode].dataset)},batches={len(data_loader[mode])},scale={train_loss_scales[mode]:.6g},effective_weight={scaled_batch_totals[mode] / scale_total:.4f})' for mode in active_train_modes)))
        validation_modes = self._validation_modes(modes)
        active_validation_modes = [mode for mode in validation_modes if self._mode_loader_has_batches(data_loader, mode)]
        if len(active_validation_modes) > 1:
            strategy = 'balanced'
            self.logger.info(f'Validation checkpoint metric uses {strategy} split aggregation: ' + ', '.join((f'{mode}(n={len(data_loader[mode].dataset)},finite_targets={self._loader_finite_target_count(data_loader, mode)})' for mode in active_validation_modes)))
        elif not active_validation_modes:
            self.logger.info('No validation loader with batches; checkpoint early stopping is disabled.')
        self.last_epoch_mode_losses = {}
        self.best_checkpoint_mode_losses = {}
        self._training_peak_memory_allocated_bytes = 0
        self._training_peak_memory_reserved_bytes = 0
        if str(self.params.get('GPU', '')).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(self.params['GPU']))
        for epoch in range(1, 1 + self.params['num_epochs']):
            self._epoch_reporting_totals = {}
            starttime = datetime.now()
            if str(self.params.get('GPU', '')).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(torch.device(self.params['GPU']))
            effective_horizon = self._advance_curriculum_epoch(epoch)
            global_epoch = epoch + epoch_offset
            horizon_ready = self._curriculum_horizon_ready()
            if horizon_ready and self.curriculum_horizon_ready_epoch is None:
                self.curriculum_horizon_ready_epoch = int(global_epoch)
                self.curriculum_checkpoint_eligible_epoch = int(global_epoch)
            self.curriculum_horizon_trace.append({'local_epoch': int(epoch), 'global_epoch': int(global_epoch), 'effective_horizon': int(effective_horizon), 'horizon_ready': bool(horizon_ready)})
            running_loss = {mode: 0.0 for mode in modes}
            self.last_epoch_mode_losses = {}
            for mode in modes:
                running_sub_losses = {'main': 0.0, 'consistency': 0.0, 'raw_consistency': 0.0, 'raw_consistency_beta': 0.0, 'raw_consistency_gamma': 0.0, 'scaled_consistency_beta': 0.0, 'scaled_consistency_gamma': 0.0, 'consistency_lambda': 0.0, 'stitched_left': 0.0, 'stitched_right': 0.0}
                is_train_mode = self._mode_is_train(mode)
                apply_adaptor = self._mode_uses_adaptor(mode)
                if is_train_mode:
                    self.model.train()
                else:
                    self.model.eval()
                step = 0
                for (batch_index, batch) in enumerate(data_loader[mode]):
                    target_metadata = self._cross_origin_target_metadata(batch)
                    (x_od, x_node, x_SIR, y_true) = self._unpack_batch(batch)
                    base_y_true = y_true[:, :self.params['pred_len'], ...]
                    with torch.set_grad_enabled(mode=is_train_mode):
                        x_od_device = x_od.float().to(x_SIR.device)
                        model_extra_kwargs = {}
                        model_extra_kwargs['apply_adaptor'] = apply_adaptor
                        if 'history_report_mask' in target_metadata:
                            model_extra_kwargs['report_mask'] = target_metadata['history_report_mask'][:, :self.params['obs_len'], ...]
                        y_pred = self.model(x_node[:, :self.params['obs_len'], ...].transpose(1, 3), x_SIR[:, :self.params['obs_len'], ...], x_od[:, :self.params['obs_len'], ...].float().to(x_SIR.device), self.max_od, **model_extra_kwargs)
                        batch_size = y_true.shape[0]
                        unscaled_loss_for_epoch = None
                        loss_weight = self._finite_target_count(base_y_true)
                        if loss_weight <= 0:
                            self.logger.warning('Skip %s batch with no finite forecast targets.', mode)
                            continue
                        if is_train_mode:
                            self.optimizer.zero_grad()
                            use_curriculum_slice = self.cl
                            train_loss_prediction = y_pred[:, :self.task_level, :, :] if use_curriculum_slice else y_pred
                            train_loss_target = base_y_true[:, :self.task_level, :, :] if use_curriculum_slice else base_y_true
                            if self._uses_report_loss():
                                (loss, interval_events) = interval_normalized_mae_loss(y_pred, base_y_true, self._interval_metadata(target_metadata))
                                if loss is None:
                                    self.logger.warning('Skip %s batch with no complete reporting interval.', mode)
                                    continue
                                normalized_errors = torch.abs(interval_events['predicted_count'] - interval_events['observed_count']) / interval_events['interval']
                                main_loss_tensor = normalized_errors
                                train_loss_tensor = normalized_errors
                                loss_weight = int(normalized_errors.numel())
                            else:
                                main_loss_tensor = self._prediction_loss_tensor(y_pred, base_y_true)
                                train_loss_tensor = self._prediction_loss_tensor(train_loss_prediction, train_loss_target)
                                loss = self._reduce_prediction_loss(train_loss_tensor)
                            main_loss = loss.item()
                            batch_consistency = 0.0
                            batch_raw_consistency = 0.0
                            batch_raw_consistency_beta = 0.0
                            batch_raw_consistency_gamma = 0.0
                            batch_scaled_consistency_beta = 0.0
                            batch_scaled_consistency_gamma = 0.0
                            batch_consistency_lambda = self.consistency_loss_lambda
                            batch_stitched_left = 0.0
                            batch_stitched_right = 0.0
                            consistency_method = self._effective_consistency_method(mode)
                            if consistency_method in {'capc_src'}:
                                cross_origin_bundle = self._cross_origin_forward_loss_bundle(mode=mode, consistency_method=consistency_method, x_node=x_node, x_SIR=x_SIR, x_od_device=x_od_device, y_true=y_true, target_metadata=target_metadata, main_loss=loss, apply_adaptor=apply_adaptor, epoch=epoch)
                                self._append_stitched_common_support_audit(epoch=epoch, batch_index=batch_index, main_loss_tensor=train_loss_tensor, bundle=cross_origin_bundle, target_metadata=target_metadata)
                                batch_raw_consistency = float(cross_origin_bundle['raw_overlap'].detach().item())
                                batch_raw_consistency_beta = float(cross_origin_bundle['raw_overlap_beta'].detach().item())
                                batch_raw_consistency_gamma = float(cross_origin_bundle['raw_overlap_gamma'].detach().item())
                                batch_scaled_consistency_beta = float(cross_origin_bundle['scaled_overlap_beta'].detach().item())
                                batch_scaled_consistency_gamma = float(cross_origin_bundle['scaled_overlap_gamma'].detach().item())
                                batch_consistency_lambda = cross_origin_bundle['consistency_lambda']
                                batch_stitched_left = cross_origin_bundle['left_scaled']
                                batch_stitched_right = cross_origin_bundle['right_scaled']
                                batch_consistency = cross_origin_bundle['consistency_scaled_total']
                                loss = loss + cross_origin_bundle['auxiliary_scaled_total']
                            objective = loss * train_loss_scales.get(mode, 1.0)
                            if not torch.isfinite(objective):
                                raise FloatingPointError('Non-finite training objective')
                            objective.backward()
                            if self.clip is not None:
                                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, error_if_nonfinite=True)
                                self._finite_gradient_steps = getattr(self, '_finite_gradient_steps', 0) + 1
                                self._max_gradient_norm = max(getattr(self, '_max_gradient_norm', 0.0), float(grad_norm))
                            self.optimizer.step()
                            running_sub_losses['main'] += main_loss * loss_weight
                            running_sub_losses['consistency'] += (float(batch_consistency.detach().item()) if torch.is_tensor(batch_consistency) else float(batch_consistency)) * loss_weight
                            running_sub_losses['raw_consistency'] += batch_raw_consistency * loss_weight
                            running_sub_losses['raw_consistency_beta'] += batch_raw_consistency_beta * loss_weight
                            running_sub_losses['raw_consistency_gamma'] += batch_raw_consistency_gamma * loss_weight
                            running_sub_losses['scaled_consistency_beta'] += batch_scaled_consistency_beta * loss_weight
                            running_sub_losses['scaled_consistency_gamma'] += batch_scaled_consistency_gamma * loss_weight
                            running_sub_losses['consistency_lambda'] += batch_consistency_lambda * loss_weight
                            running_sub_losses['stitched_left'] += (float(batch_stitched_left.detach().item()) if torch.is_tensor(batch_stitched_left) else float(batch_stitched_left)) * loss_weight
                            running_sub_losses['stitched_right'] += (float(batch_stitched_right.detach().item()) if torch.is_tensor(batch_stitched_right) else float(batch_stitched_right)) * loss_weight
                            unscaled_loss_for_epoch = float(loss.detach().item())
                        elif is_report_mae_selection_metric(self.params.get('selection_metric')):
                            (loss, interval_events) = interval_normalized_mae_loss(y_pred, base_y_true, self._interval_metadata(target_metadata))
                            if loss is None:
                                self.logger.warning('Skip %s batch with no complete reporting interval.', mode)
                                continue
                            loss_weight = int(interval_events['predicted_count'].numel())
                        else:
                            validation_loss_tensor = self._prediction_loss_tensor(y_pred, base_y_true)
                            loss = self._reduce_prediction_loss(validation_loss_tensor)
                    running_loss[mode] += (unscaled_loss_for_epoch if unscaled_loss_for_epoch is not None else loss.item()) * loss_weight
                    step += loss_weight
                if step == 0:
                    if self.params.get('loss') == 'report_mae':
                        raise ValueError(f'{mode} split has no usable report events')
                    self.logger.info(f'Epoch {epoch}, {mode} data is empty; skip.')
                    continue
                epoch_mode_loss = running_loss[mode] / step
                self.last_epoch_mode_losses[mode] = epoch_mode_loss
                consistency_enabled = self._effective_consistency_method(mode) != 'none'
                if is_train_mode:
                    self.last_epoch_mode_losses[f'{mode}_main'] = running_sub_losses['main'] / step
                if is_train_mode and consistency_enabled:
                    main_epoch = running_sub_losses['main'] / step
                    raw_epoch = running_sub_losses['raw_consistency'] / step
                    scaled_epoch = running_sub_losses['consistency'] / step
                    lambda_epoch = running_sub_losses['consistency_lambda'] / step
                    self.logger.info(f"Epoch {epoch}, consistency diagnostics: main={main_epoch:.6g}, raw={raw_epoch:.6g}, scaled={scaled_epoch:.6g}, raw_beta={running_sub_losses['raw_consistency_beta'] / step:.6g}, raw_gamma={running_sub_losses['raw_consistency_gamma'] / step:.6g}, scaled_beta={running_sub_losses['scaled_consistency_beta'] / step:.6g}, scaled_gamma={running_sub_losses['scaled_consistency_gamma'] / step:.6g}, lambda={lambda_epoch:.6g}, stitched_left={running_sub_losses['stitched_left'] / step:.6g}, stitched_right={running_sub_losses['stitched_right'] / step:.6g}, scaled/main={scaled_epoch / max(main_epoch, 1e-12):.6g}")
                if active_validation_modes and mode == active_validation_modes[-1]:
                    self.curriculum_horizon_trace[-1].update({'effective_horizon_end': min(int(self.task_level), int(self.params['pred_len'])), 'horizon_ready_end': self._curriculum_horizon_ready()})
                    validation_loss_items = [(val_mode, self.last_epoch_mode_losses[val_mode]) for val_mode in active_validation_modes if val_mode in self.last_epoch_mode_losses]
                    if not validation_loss_items:
                        continue
                    (epoch_validation_loss, validation_weights) = self._combine_validation_losses(validation_loss_items, data_loader)
                    if len(validation_loss_items) > 1:
                        validation_detail = ' (' + ', '.join((f'{val_mode}={loss:.5},w={validation_weights[val_mode]:.4f}' for (val_mode, loss) in validation_loss_items)) + ')'
                        self.last_epoch_mode_losses['validation_combined'] = epoch_validation_loss
                    else:
                        validation_detail = ''
                    task_level_not_finished = not self._curriculum_horizon_ready()
                    post_checkpoint_warmup = getattr(self, 'training_stage', 'base') == 'post' and epoch < int(self.params.get('post_train_checkpoint_start_epoch', 0))
                    if post_checkpoint_warmup:
                        self.logger.info(f"Epoch {epoch}, validation loss {epoch_validation_loss:.5}{validation_detail}. Before post checkpoint start epoch {self.params.get('post_train_checkpoint_start_epoch', 0)}")
                    elif task_level_not_finished:
                        self.logger.info(f'Epoch {epoch}, validation loss {epoch_validation_loss:.5}{validation_detail}. Curriculum has not reached the full prediction horizon')
                    elif epoch_validation_loss <= val_loss:
                        self.logger.info(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_validation_loss:.5}{validation_detail}. ')
                        self.logger.info(f'Update model checkpoint..used {(datetime.now() - starttime).seconds}s')
                        val_loss = epoch_validation_loss
                        checkpoint = self._checkpoint_state(epoch + epoch_offset)
                        torch.save(checkpoint, checkpoint_path)
                        self.best_checkpoint_mode_losses = dict(self.last_epoch_mode_losses)
                        patience_count = early_stop_patience
                    else:
                        self.logger.info(f'Epoch {epoch}, validation loss {epoch_validation_loss}{validation_detail} does not improve from {val_loss:.5}.')
                        self.logger.info(f'used {(datetime.now() - starttime).seconds}s')
                        patience_count -= 1
                        if patience_count == 0:
                            epoch_reporting.record_epoch(self, data_loader, epoch, epoch_offset)
                            self.logger.info(f"Early stopping at epoch {epoch}. {'CoSTEP'} model training ends.")
                            global_epoch = epoch + epoch_offset
                            return
                if mode == 'test':
                    self.logger.info(f'Epoch {epoch}, test loss {epoch_mode_loss}. ')
            epoch_reporting.record_epoch(self, data_loader, epoch, epoch_offset)
            if str(self.params.get('GPU', '')).startswith('cuda') and torch.cuda.is_available():
                device = torch.device(self.params['GPU'])
                epoch_peak_allocated = torch.cuda.max_memory_allocated(device)
                epoch_peak_reserved = torch.cuda.max_memory_reserved(device)
                self._training_peak_memory_allocated_bytes = max(self._training_peak_memory_allocated_bytes, epoch_peak_allocated)
                self._training_peak_memory_reserved_bytes = max(self._training_peak_memory_reserved_bytes, epoch_peak_reserved)
                self.logger.info('CUDA memory epoch %d: peak_allocated_mib=%.2f, peak_reserved_mib=%.2f', epoch, epoch_peak_allocated / 1048576, epoch_peak_reserved / 1048576)
        self.logger.info(f"{'CoSTEP'} model training ends.")
        final_epoch = epoch + epoch_offset
        if not self.best_checkpoint_mode_losses and checkpoint_path is not None:
            checkpoint = self._checkpoint_state(epoch + epoch_offset)
            torch.save(checkpoint, checkpoint_path)
        return

    def predict(self, loader, checkpoint_path, apply_adaptor=False):
        checkpoint = self._load_model_checkpoint(checkpoint_path)
        if apply_adaptor != (checkpoint['stage'] == 'post'):
            raise ValueError('Prediction stage does not match checkpoint')
        self.model.eval()
        (predictions, targets, metadata) = ([], [], {})
        with torch.no_grad():
            for batch in loader:
                (x_od, x_node, x_sir, y) = self._unpack_batch(batch)
                meta = self._cross_origin_target_metadata(batch)
                n = self.params['obs_len']
                kwargs = {'apply_adaptor': apply_adaptor}
                if 'history_report_mask' in meta:
                    kwargs['report_mask'] = meta['history_report_mask'][:, :n]
                pred = self.model(x_node[:, :n].transpose(1, 3), x_sir[:, :n], x_od[:, :n].float().to(x_sir.device), self.max_od, **kwargs)
                predictions.append(pred.cpu().numpy())
                targets.append(y[:, :self.params['pred_len']].cpu().numpy())
                for (key, value) in meta.items():
                    if key.startswith('future_') or key == 'origin_index':
                        metadata.setdefault(key, []).append(value.detach().cpu().numpy())
        return (np.concatenate(predictions), np.concatenate(targets), {k: np.concatenate(v) for (k, v) in metadata.items()})
