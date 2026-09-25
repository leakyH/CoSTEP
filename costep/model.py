"""
Part of code is adapted from https://github.com/nnzhan/Graph-WaveNet
"""
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .mechanism import DEFAULT_LATENCY_PROFILE, DEFAULT_LATENCY_SDLOG_MIN, get_latency_profile_config, DynamicEpidemicStep

def apply_gamma_floor(param_g: torch.Tensor, minimum: float | torch.Tensor=0.01, straight_through: bool=False) -> torch.Tensor:
    """Apply the rollout gamma floor, optionally with an identity surrogate gradient."""
    minimum_tensor = torch.as_tensor(minimum, device=param_g.device, dtype=param_g.dtype)
    floored = torch.maximum(param_g, minimum_tensor)
    if straight_through:
        return param_g + (floored - param_g).detach()
    return floored

def compute_observation_gamma_floor(documented_i: torch.Tensor, susceptible: torch.Tensor, absolute_minimum: float=0.01) -> torch.Tensor:
    """Return a per-window/node lower bound for the reporting rate.

    ``documented_i`` contains daily documented incidence in the observation
    window.  The data-dependent bound is ``sum_t documented_i[t] / S0``, where
    ``S0`` is the susceptible stock at the first date in the window.  It is
    constant over time and broadcasts to a full gamma trajectory.
    """
    if documented_i.ndim != 4 or susceptible.ndim != 4:
        raise ValueError('documented_i and susceptible must have shape [B,T,N,1]')
    if documented_i.shape[0] != susceptible.shape[0] or documented_i.shape[2:] != susceptible.shape[2:]:
        raise ValueError('documented_i and susceptible batch/node dimensions must match')
    if documented_i.shape[1] < 1 or susceptible.shape[1] < 1:
        raise ValueError('the observation window must contain at least one date')
    documented_total = documented_i.clamp_min(0.0).sum(dim=1, keepdim=True)
    s0 = susceptible[:, :1, ...].clamp_min(torch.finfo(documented_i.dtype).eps)
    data_floor = documented_total / s0
    absolute_floor = torch.as_tensor(absolute_minimum, device=documented_i.device, dtype=documented_i.dtype)
    return torch.maximum(data_floor, absolute_floor).clamp_max(1.0)

def _build_piecewise_bezier_basis(target_len, segment_count, degree):
    control_count = segment_count * degree + 1
    eval_points = torch.linspace(0, segment_count, target_len, dtype=torch.float32)
    left_idx = torch.floor(eval_points).long().clamp(max=segment_count - 1)
    u = (eval_points - left_idx.float()).clamp(0.0, 1.0)
    basis = torch.zeros(target_len, control_count, dtype=torch.float32)
    for row in range(target_len):
        seg = int(left_idx[row].item())
        start = seg * degree
        local_u = float(u[row].item())
        if degree == 2:
            weights = [(1.0 - local_u) ** 2, 2.0 * (1.0 - local_u) * local_u, local_u ** 2]
        elif degree == 3:
            weights = [(1.0 - local_u) ** 3, 3.0 * (1.0 - local_u) ** 2 * local_u, 3.0 * (1.0 - local_u) * local_u ** 2, local_u ** 3]
        else:
            raise ValueError(f'Unsupported bezier degree: {degree}')
        basis[row, start:start + degree + 1] = torch.tensor(weights, dtype=torch.float32)
    return basis

def _build_periodic_basis(target_len, period=7.0, harmonics=2, trend=False, include_intercept=True):
    positions = torch.arange(target_len, dtype=torch.float32)
    basis = []
    if include_intercept:
        basis.append(torch.ones_like(positions))
    for harmonic in range(1, harmonics + 1):
        phase = 2.0 * np.pi * harmonic * positions / float(period)
        basis.append(torch.sin(phase))
        basis.append(torch.cos(phase))
    if trend:
        basis.append(torch.linspace(-1.0, 1.0, target_len, dtype=torch.float32))
    return torch.stack(basis, dim=0)

class CurveParameterDecoder(nn.Module):
    """Bezier/periodic curves, or independent daily logits for no-curve ablation."""

    def __init__(self, target_len, curve=True):
        super().__init__()
        self.target_len = int(target_len)
        self.curve = curve
        if not curve:
            return
        segment_count = max(1, int(np.ceil(self.target_len / 7.0)))
        beta_basis = _build_piecewise_bezier_basis(self.target_len, segment_count, degree=3)
        self.register_buffer('beta_basis', beta_basis)
        gamma_basis = _build_periodic_basis(self.target_len, period=7.0, harmonics=2, trend=True, include_intercept=True)
        self.register_buffer('gamma_basis', gamma_basis)

    @property
    def beta_channels(self):
        return int(self.beta_basis.shape[1]) if self.curve else self.target_len

    @property
    def gamma_channels(self):
        return int(self.gamma_basis.shape[0]) if self.curve else self.target_len

    def decode_beta(self, beta_logits):
        if not self.curve:
            return beta_logits[:, :self.target_len, ...]
        beta_logits = beta_logits[:, :self.beta_channels, ...]
        basis = self.beta_basis.to(device=beta_logits.device, dtype=beta_logits.dtype)
        beta_coeff = beta_logits.squeeze(-1).permute(0, 2, 1)
        return torch.einsum('bnk,tk->bnt', beta_coeff, basis).permute(0, 2, 1).unsqueeze(-1)

    def decode_gamma(self, gamma_coeff):
        if not self.curve:
            return gamma_coeff
        basis = self.gamma_basis.to(device=gamma_coeff.device, dtype=gamma_coeff.dtype)
        gamma_coeff = gamma_coeff.squeeze(-1).permute(0, 2, 1)
        gamma_curve = torch.einsum('bnk,kt->bnt', gamma_coeff, basis)
        return gamma_curve.permute(0, 2, 1).unsqueeze(-1)

class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        module_input = x
        avg_out = self.fc(self.avg_pool(x))
        return module_input * avg_out

class nconv(nn.Module):

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        if len(A.shape) == 2:
            x = torch.einsum('vw, ncwl->ncvl', A, x)
        elif len(A.shape) == 3:
            x = torch.einsum('nvw, ncwl->ncvl', A, x)
        elif len(A.shape) == 4:
            x = torch.einsum('nlvw, ncwl->ncvl', A, x)
        return x.contiguous()

class linear(nn.Module):

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):

    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = order * support_len * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = []
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class CausalGraphSelector(nn.Module):

    def __init__(self, idx_start: int=None, idx_end: int=None, idx: Union[List[int] | np.ndarray | torch.Tensor | None]=None):
        super().__init__()
        if idx is None:
            if idx_start is None or idx_end is None:
                raise ValueError('Either idx or both idx_start and idx_end must be provided')
            self.register_buffer('idx', torch.arange(idx_start, idx_end, dtype=int))
        else:
            if isinstance(idx, list):
                idx = torch.tensor(idx, dtype=int)
            elif isinstance(idx, np.ndarray):
                idx = torch.from_numpy(idx).long()
            elif not isinstance(idx, torch.Tensor):
                raise ValueError('idx must be a list, numpy array, torch tensor, or slice')
            self.register_buffer('idx', idx)

    def forward(self, H, As):
        assert H.shape[-1] == len(self.idx), f'Input H has time dimension {H.shape[-1]}, but selector is configured for indices {self.idx}'
        return [A[:, self.idx, ...] for A in As]

class CoSTEPParameterEncoder(nn.Module):
    """CoSTEP graph-temporal parameter encoder with causal graph selection."""

    def __init__(self, num_nodes, dropout, in_dim, in_len, out_len, residual_channels, dilation_channels, skip_channels, end_channels, kernel_size, blocks, layers, curve=True):
        super().__init__()
        self.supports_raw_param_adaptor = True
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.align = True
        self.temporal_shift = False
        self.latent_channels = skip_channels * blocks * layers
        self.calibration_feature_channels = skip_channels
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.temporal_causal_selectors = nn.ModuleList()
        receptive_field = 1
        self.supports_len = 2
        self.feature_attention = ChannelAttention(in_dim)
        self.start_dropout = nn.Dropout2d(p=dropout)
        for b in range(blocks):
            additional_scope = 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.temporal_causal_selectors.append(CausalGraphSelector(idx_start=receptive_field + additional_scope - 2, idx_end=in_len))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                norm_time = (2 ** layers - 1) * blocks + 2 - receptive_field
                self.ln.append(nn.LayerNorm([residual_channels, num_nodes, norm_time]))
                self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
        self.curve_decoder = CurveParameterDecoder(out_len, curve=curve)
        beta_out_channels = self.curve_decoder.beta_channels
        gamma_out_channels = self.curve_decoder.gamma_channels
        self.end_conv_b1 = nn.Conv2d(in_channels=skip_channels * blocks * layers, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_b2 = nn.Conv2d(in_channels=end_channels, out_channels=beta_out_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_g1 = nn.Conv2d(in_channels=skip_channels * blocks * layers, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.dow_proj = nn.Conv2d(2, 8, kernel_size=(1, 1))
        self.end_conv_g2 = nn.Conv2d(in_channels=end_channels, out_channels=gamma_out_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_n1 = nn.Conv2d(in_channels=skip_channels * blocks * layers, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_n2 = nn.Conv2d(in_channels=end_channels, out_channels=out_len, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field
        self.end_activate_n = nn.Softplus()

    def encode_latent(self, input, adp_g):
        """Return the align-aware TCN/GNN trunk output before parameter heads."""
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0
        for i in range(self.blocks * self.layers):
            res = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            if i == 0:
                skip = s
            else:
                skip = torch.cat((s, skip[:, :, :, -s.size(3):]), dim=1)
            x = self.gconv[i](x, self.temporal_causal_selectors[i](x, adp_g))
            if i == 0:
                dense = 0
            else:
                dense = dense[:, :, :, -x.size(3):]
            dense = res[:, :, :, -x.size(3):] + dense
            gate = torch.sigmoid(x)
            x = x * gate + dense * (1 - gate)
            x = self.ln[i](x)
        return skip

    def forward(self, input, adp_g):
        skip = self.encode_latent(input, adp_g)
        param_b = F.relu(skip)
        param_b = F.relu(self.end_conv_b1(param_b))
        param_b = self.end_conv_b2(param_b)
        param_g = F.relu(skip)
        param_g = F.relu(self.end_conv_g1(param_g))
        param_g = self.end_conv_g2(param_g)
        param_b = self.curve_decoder.decode_beta(param_b)
        param_g = self.curve_decoder.decode_gamma(param_g)
        param_b = torch.sigmoid(param_b)
        param_g = torch.sigmoid(param_g)
        param_n = F.relu(skip)
        param_n = F.relu(self.end_conv_n1(param_n))
        param_n = self.end_activate_n(self.end_conv_n2(param_n))
        return (param_b, param_g, param_n)

class EpidemicStep(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, param_g, kernel, S, I_scaled, time_weight):
        inf_pressure = compute_new_infections_pressure(I_scaled, kernel)
        propagation = S * inf_pressure
        I_new = torch.einsum('blac,l->bac', propagation, time_weight)
        previous_stock = I_scaled[:, -1, ...]
        previous_gamma = param_g[:, -2, ...]
        current_gamma = param_g[:, -1, ...]
        new_I_scaled = previous_stock * (1.0 - previous_gamma) + I_new
        new_documented_I = new_I_scaled * current_gamma
        S_t = S[:, -1, ...] - propagation[:, -1, :, :]
        _raise_nonfinite_debug('new_I_scaled', new_I_scaled, location='new_I_scaled')
        return (new_documented_I, S_t, new_I_scaled)

def _raise_nonfinite_debug(name: str, tensor: torch.Tensor, **context):
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return
    bad_idx = torch.nonzero(~finite_mask, as_tuple=False)[0]
    bad_value = tensor[tuple((int(v) for v in bad_idx.tolist()))].detach().cpu().item()
    context_bits = [f'{k}={v}' for (k, v) in context.items()]
    context_str = ', '.join(context_bits)
    raise RuntimeError(f'Non-finite detected in {name}; shape={tuple(tensor.shape)}; first_bad_index={tuple((int(v) for v in bad_idx.tolist()))}; first_bad_value={bad_value}; {context_str}')

def compute_pcf_core(od_core, pop, param_b):
    """Row-normalized OD makes this kernel invariant to positive global scale."""
    od_core = od_core.squeeze(-1)
    odsum = od_core.sum(dim=3, keepdim=True)
    M = od_core / odsum
    Nw = torch.matmul(M.transpose(-1, -2), pop).squeeze(-1)
    W_vec = 1 / Nw * param_b.squeeze(-1)
    K_all = torch.matmul(M * W_vec.unsqueeze(-2), M.transpose(-1, -2))
    return K_all

def compute_new_infections_pressure(i, K_all=None, *, od_core=None, pop=None, param_b=None):
    if K_all is None:
        K_all = compute_pcf_core(od_core, pop, param_b)
    core_infection_pressure = torch.matmul(K_all, i)
    _raise_nonfinite_debug('inf_pressure', core_infection_pressure, location='compute_new_infections_pressure')
    return core_infection_pressure

class CoSTEP(nn.Module):

    def __init__(self, num_nodes, dropout=0.5, in_dim=4, in_len=14, out_len=14, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=2, layers=3, latency_window=None, future_od_period=7, stock_estimator='legacy_daily', curve=True, sir_decoder_variant='delay'):
        super(CoSTEP, self).__init__()
        if sir_decoder_variant not in {'delay', 'dynamic'}:
            raise ValueError('sir_decoder_variant must be delay or dynamic')
        if sir_decoder_variant == 'dynamic' and (stock_estimator != 'legacy_daily' or not curve):
            raise ValueError('dynamic decoder requires daily CoSTEP with curves')
        self.sir_decoder_variant = sir_decoder_variant
        self.stock_estimator = stock_estimator
        if self.stock_estimator not in {'legacy_daily', 'causal_interval_anchor'}:
            raise ValueError('stock_estimator must be legacy_daily or causal_interval_anchor')
        self.latency_window_length = int(in_len if latency_window is None else latency_window)
        if self.latency_window_length < 1:
            raise ValueError('latency_window must be >= 1')
        self.latency_method = 'learnempirical'
        self.gamma_floor_straight_through = True
        self.gamma_floor_mode = 'observation_stock'
        self.latency_sdlog_min = DEFAULT_LATENCY_SDLOG_MIN
        para_out_len = in_len + out_len
        self.stcell = CoSTEPParameterEncoder(num_nodes, dropout, in_dim, in_len, para_out_len, residual_channels, dilation_channels, skip_channels, end_channels, kernel_size, blocks, layers, curve=curve)
        self.out_dim = out_len
        self.in_len = in_len
        self.glm_type = 'FullDynamic'
        self.future_od_method = 'repeat_weekly'
        self.future_od_period = max(1, int(future_od_period))
        if self.sir_decoder_variant == 'dynamic':
            self.SIRcell = DynamicEpidemicStep()
            self.inc_init = nn.Parameter(torch.empty(out_len, in_len))
            nn.init.normal_(self.inc_init, 1, 0.01)
            return
        self.SIRcell = EpidemicStep()
        latency_config = get_latency_profile_config(DEFAULT_LATENCY_PROFILE)
        self.latency_meanlog = nn.Parameter(torch.tensor(latency_config['meanlog'], dtype=torch.float32))
        init_sdlog = torch.tensor(latency_config['sdlog'] - self.latency_sdlog_min, dtype=torch.float32)
        self.latency_raw_sdlog = nn.Parameter(self._inverse_softplus(init_sdlog))

    def _apply_param_adaptor(self, param_b: torch.Tensor, param_g: torch.Tensor, apply_adaptor: bool=False):
        if not apply_adaptor:
            return (param_b, param_g)
        param_adaptor = getattr(self, 'param_adaptor', None)
        if param_adaptor is None:
            return (param_b, param_g)
        return param_adaptor(param_b, param_g)

    def _record_beta_processing_stages(self, raw_beta: torch.Tensor, adaptor_beta: torch.Tensor):
        self.recorded_beta_raw = raw_beta.detach()
        self.recorded_beta_after_adaptor = adaptor_beta.detach()

    @staticmethod
    def _inverse_softplus(value):
        return torch.log(torch.expm1(value).clamp_min(1e-12))

    @staticmethod
    def _lognormal_latency_probs(window_length, meanlog=None, sdlog=None, device=None, dtype=None):
        if meanlog is None or sdlog is None:
            latency_config = get_latency_profile_config(DEFAULT_LATENCY_PROFILE)
            if meanlog is None:
                meanlog = latency_config['meanlog']
            if sdlog is None:
                sdlog = latency_config['sdlog']
        if torch.is_tensor(meanlog):
            device = meanlog.device
            dtype = meanlog.dtype
        elif torch.is_tensor(sdlog):
            device = sdlog.device
            dtype = sdlog.dtype
        dtype = dtype or torch.float32
        meanlog = torch.as_tensor(meanlog, device=device, dtype=dtype)
        sdlog = torch.as_tensor(sdlog, device=device, dtype=dtype)
        sdlog = torch.clamp(sdlog, min=torch.finfo(dtype).eps)
        days = torch.arange(1, int(window_length) + 1, device=device, dtype=dtype)
        lower = torch.clamp(days - 0.5, min=1e-06)
        upper = days + 0.5
        denom = sdlog * np.sqrt(2.0)
        cdf_lower = 0.5 * (1.0 + torch.erf((torch.log(lower) - meanlog) / denom))
        cdf_upper = 0.5 * (1.0 + torch.erf((torch.log(upper) - meanlog) / denom))
        cdf_lower = torch.cat([torch.zeros_like(cdf_lower[:1]), cdf_lower[1:]], dim=0)
        cdf_upper = torch.cat([cdf_upper[:-1], torch.ones_like(cdf_upper[-1:])], dim=0)
        probs = torch.clamp(cdf_upper - cdf_lower, min=0.0)
        probs = torch.flip(probs, dims=[0])
        return probs / torch.clamp(probs.sum(), min=torch.finfo(dtype).tiny)

    def _current_latency_sdlog(self):
        return F.softplus(self.latency_raw_sdlog) + self.latency_sdlog_min

    def _current_adaptor_latency_sdlog(self):
        return F.softplus(self.adaptor_latency_raw_sdlog) + self.latency_sdlog_min

    def get_latency_distribution(self, apply_adaptor: bool=False):
        if apply_adaptor and hasattr(self, 'adaptor_latency_meanlog') and hasattr(self, 'adaptor_latency_raw_sdlog'):
            return self._lognormal_latency_probs(self.latency_window_length, meanlog=self.adaptor_latency_meanlog, sdlog=self._current_adaptor_latency_sdlog())
        return self._lognormal_latency_probs(self.latency_window_length, meanlog=self.latency_meanlog, sdlog=self._current_latency_sdlog())

    def _repeat_weekly_future_od(self, od_history, horizon):
        history_len = od_history.shape[1]
        if history_len == 0:
            return torch.zeros(od_history.shape[0], horizon, od_history.shape[2], od_history.shape[3], device=od_history.device, dtype=od_history.dtype)
        base_len = min(history_len, self.future_od_period)
        base = od_history[:, -base_len:, ...]
        repeat_times = int(np.ceil(horizon / base_len))
        return base.repeat(1, repeat_times, 1, 1)[:, :horizon, ...]

    def predict_future_od(self, od_history, horizon=None):
        horizon = int(horizon or self.out_dim)
        od_core = od_history.squeeze(-1) if od_history.dim() == 5 else od_history
        pred = self._repeat_weekly_future_od(od_core, horizon)
        return pred.unsqueeze(-1)

    def _causal_interval_history_stock(self, documented_rate: torch.Tensor, report_mask: torch.Tensor, param_g: torch.Tensor, susceptible: torch.Tensor, kernel: torch.Tensor, *, od_core: torch.Tensor | None=None, pop: torch.Tensor | None=None, param_b: torch.Tensor | None=None) -> torch.Tensor:
        """Propagate one latent stock and hard-anchor it at report events."""
        history_len = documented_rate.shape[1]
        gamma = param_g[:, :history_len, ...].clamp_min(1e-06)
        anchors = documented_rate / gamma
        mask = report_mask[:, :history_len, ...].to(dtype=torch.bool)
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        state = anchors[:, 0, ...]
        states = [state]
        for time_index in range(1, history_len):
            local_kwargs = {}
            if od_core is not None:
                local_kwargs['od_core'] = od_core[:, time_index - 1:time_index, ...]
            if pop is not None:
                local_kwargs['pop'] = pop[:, time_index - 1:time_index, ...]
            if param_b is not None:
                local_kwargs['param_b'] = param_b[:, time_index - 1:time_index, ...]
            pressure = compute_new_infections_pressure(state.unsqueeze(1), kernel[:, time_index - 1:time_index, ...], **local_kwargs)[:, 0, ...]
            new_infections = susceptible[:, time_index - 1, ...] * pressure
            propagated = state * (1.0 - gamma[:, time_index - 1, ...]) + new_infections
            state = torch.where(mask[:, time_index, ...], anchors[:, time_index, ...], propagated)
            states.append(state)
        return torch.stack(states, dim=1)

    def _rollout_full_family(self, SIR, od, max_od, param_b, param_g, inverse_norm_infect_detact_latency, report_mask=None):
        outputs_SIR = []
        preserve_long_SIR = torch.empty(SIR.shape[0], self.in_len + self.out_dim, SIR.shape[2], 3, device=SIR.device)
        pop = SIR[:, [0], :, 0:3].sum(axis=-1, keepdims=True).expand(-1, self.in_len + self.out_dim, -1, -1)
        future_od = self.predict_future_od(od[:, :self.in_len, ...], horizon=self.out_dim)
        od_includes_future = torch.concat([od[:, :self.in_len, ...], future_od], dim=1)
        K_all = compute_pcf_core(od_includes_future, pop, param_b)
        _raise_nonfinite_debug('K_all', K_all, location='_rollout_full_family', glm_type='FullDynamic')
        documented_I = SIR[:, :self.in_len, :, [-1]]
        if self.stock_estimator == 'causal_interval_anchor':
            if report_mask is None:
                raise ValueError('causal_interval_anchor requires history report_mask')
            I_scaled = self._causal_interval_history_stock(documented_I, report_mask, param_g, SIR[:, :self.in_len, :, [0]], K_all[:, :self.in_len, ...], od_core=od_includes_future[:, :self.in_len, ...], pop=pop[:, :self.in_len, ...], param_b=param_b[:, :self.in_len, ...])
        else:
            I_scaled = documented_I / param_g[:, :self.in_len, ...]
        inf_pressure = compute_new_infections_pressure(I_scaled, K_all[:, :self.in_len, ...], od_core=od[:, :self.in_len, ...], pop=pop[:, :self.in_len, ...], param_b=param_b[:, :self.in_len, ...])
        S_exclude_I_scaled = SIR[:, :, :, [0]] - I_scaled
        S_exclude_I_scaled_T = S_exclude_I_scaled[:, self.in_len - 1, :, :]
        factor = 1 - inf_pressure
        inverse_cum_factor = torch.flip(torch.cumprod(torch.flip(factor, dims=[1])[:, 1:, :], dim=1), dims=[1])
        cumweight = torch.cumsum(inverse_norm_infect_detact_latency, dim=0)[:-1]
        len_cumweight = len(cumweight)
        consider_start = max(0, self.in_len - 1 - len_cumweight)
        consider_end = self.in_len - 1
        actual_length = consider_end - consider_start
        coeff = inf_pressure[:, consider_start:consider_end, :] * cumweight[None, len(cumweight) - actual_length:, None, None] / inverse_cum_factor[:, consider_start:consider_end, :]
        denom = 1.0 + coeff.sum(dim=1)
        S_precise_T = S_exclude_I_scaled_T / denom
        S_precise_T_expand = S_precise_T.unsqueeze(1)
        history_S = torch.cat([S_precise_T_expand / inverse_cum_factor, S_precise_T_expand], dim=1)
        preserve_long_SIR[:, :history_S.shape[1], :, [0]] = history_S
        preserve_long_SIR[:, :documented_I.shape[1], :, [1]] = documented_I
        preserve_long_SIR[:, :I_scaled.shape[1], :, [2]] = I_scaled
        for i in range(self.out_dim):
            (I_new, S_t, new_I_scaled) = self.SIRcell(param_g[:, i + self.in_len - self.latency_window_length:i + self.in_len + 1, ...], K_all[:, i + self.in_len - self.latency_window_length:i + self.in_len], preserve_long_SIR[:, i + self.in_len - self.latency_window_length:i + self.in_len, :, [0]], preserve_long_SIR[:, i + self.in_len - self.latency_window_length:i + self.in_len, :, [2]], inverse_norm_infect_detact_latency)
            preserve_long_SIR[:, i + self.in_len, ..., [0]] = S_t
            preserve_long_SIR[:, i + self.in_len, ..., [1]] = I_new
            preserve_long_SIR[:, i + self.in_len, ..., [2]] = new_I_scaled
            outputs_SIR.append(I_new)
        outputs = torch.stack(outputs_SIR, dim=1)
        return outputs

    def decode_parameters(self, SIR, od, max_od, param_b, param_g, *, apply_adaptor=False, report_mask=None):
        if self.sir_decoder_variant == 'delay':
            return self._rollout_full_family(
                SIR, od, max_od, param_b, param_g,
                self.get_latency_distribution(apply_adaptor=apply_adaptor), report_mask=report_mask,
            )
        expected = self.in_len + self.out_dim
        if param_b.shape[1] != expected or param_g.shape[1] != expected:
            raise ValueError('Dynamic decoder requires the complete observation + future parameter trajectory')
        beta, gamma = param_b[:, -self.out_dim:], param_g[:, -self.out_dim:]
        mobility = torch.einsum('kl,blnmc->bknmc', torch.softmax(self.inc_init, dim=1), od[:, :self.in_len]).squeeze(-1)
        # JAPAN FullDynamic research configuration uses OD scale factor 6.
        mobility = mobility * max_od * 6.0
        state = SIR[:, -1]
        predictions = []
        for horizon in range(self.out_dim):
            result = self.SIRcell(beta[:, horizon], gamma[:, horizon], mobility[:, horizon], state)
            state = result[..., 1:]
            predictions.append(result[..., :1])
        return torch.stack(predictions, dim=1)

    def forward(self, x_node, SIR, od, max_od, only_params=False, apply_adaptor=False, report_mask=None):
        g = od.squeeze(-1)
        g_t = g.permute(0, 1, 3, 2)
        g_dyn = [g / g.sum(3, True), g_t / g_t.sum(3, True)]
        stcell_output = self.stcell(x_node, g_dyn)
        (param_b, param_g, param_n) = stcell_output
        beta_raw = param_b
        (param_b, param_g) = self._apply_param_adaptor(param_b, param_g, apply_adaptor=apply_adaptor)
        beta_after_adaptor = param_b
        self._record_beta_processing_stages(beta_raw, beta_after_adaptor)
        self.recorded_gamma_pre_floor = param_g
        observation_gamma_floor = compute_observation_gamma_floor(SIR[:, :self.in_len, :, -1:], SIR[:, :self.in_len, :, 0:1], absolute_minimum=0.01)
        self.recorded_gamma_floor = observation_gamma_floor
        param_g = apply_gamma_floor(param_g, minimum=observation_gamma_floor, straight_through=True)
        if only_params:
            return (param_b, param_g, param_n)
        outputs = self.decode_parameters(SIR, od, max_od, param_b, param_g, apply_adaptor=apply_adaptor, report_mask=report_mask)
        self.recorded_beta = param_b
        self.recorded_gamma = param_g
        return outputs

class CoSTEPIR(CoSTEP):

    def __init__(self, *args, **kwargs):
        kwargs['stock_estimator'] = 'causal_interval_anchor'
        super().__init__(*args, **kwargs)
