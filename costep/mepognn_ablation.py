"""Fixed native MepoGNN encoder/decoder ablation (JAPAN and US_STATE).

Adapted from model/MepoGNN.py stcell and SIRcell(modify=False).
See licenses/MepoGNN_LICENSE and licenses/GraphWaveNet_LICENSE.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import CoSTEP, apply_gamma_floor, compute_observation_gamma_floor, gcn


class NativeEncoder(nn.Module):
    def __init__(self, num_nodes, in_dim, out_len=14):
        super().__init__()
        residual, dilation, skip, end = 32, 32, 256, 512
        self.filter_convs, self.gate_convs = nn.ModuleList(), nn.ModuleList()
        self.residual_convs, self.skip_convs = nn.ModuleList(), nn.ModuleList()
        self.ln, self.gconv = nn.ModuleList(), nn.ModuleList()
        self.start_conv = nn.Conv2d(in_dim, residual, (1, 1))
        receptive = 1
        for _ in range(2):
            for layer in range(3):
                rate = 2 ** layer
                self.filter_convs.append(nn.Conv2d(residual, dilation, (1, 2), dilation=rate))
                self.gate_convs.append(nn.Conv2d(residual, dilation, (1, 2), dilation=rate))
                self.residual_convs.append(nn.Conv2d(dilation, residual, (1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation, skip, (1, 1)))
                receptive += rate
                self.ln.append(nn.LayerNorm([residual, num_nodes, 16 - receptive]))
                self.gconv.append(gcn(dilation, residual, .5, support_len=2))
        self.end_conv_b1 = nn.Conv2d(skip * 6, end, (1, 1))
        self.end_conv_b2 = nn.Conv2d(end, out_len, (1, 1))
        self.end_conv_g1 = nn.Conv2d(skip * 6, end, (1, 1))
        self.end_conv_g2 = nn.Conv2d(end, out_len, (1, 1))
        self.receptive_field = receptive

    def forward(self, node, graphs):
        x = F.pad(node, (max(0, self.receptive_field - node.shape[-1]), 0, 0, 0))
        x = self.start_conv(x)
        for i in range(6):
            residual = x
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            s = self.skip_convs[i](x)
            skip = s if i == 0 else torch.cat((s, skip[..., -s.shape[-1]:]), dim=1)
            x = self.gconv[i](x, graphs)
            dense = residual[..., -x.shape[-1]:] + (0 if i == 0 else dense[..., -x.shape[-1]:])
            gate = torch.sigmoid(x)
            x = self.ln[i](x * gate + dense * (1 - gate))
        beta = torch.sigmoid(self.end_conv_b2(F.relu(self.end_conv_b1(F.relu(skip)))))
        gamma = torch.sigmoid(self.end_conv_g2(F.relu(self.end_conv_g1(F.relu(skip)))))
        return beta, gamma


class MepoGNNAblation(nn.Module):
    """Native Dynamic with direct Future14 parameters and no latency/floor."""
    sir_decoder_variant = 'dynamic'
    latency_method = 'learnempirical'  # Post metadata only; no latency state.

    def __init__(self, num_nodes, in_dim):
        super().__init__()
        self.stcell = NativeEncoder(num_nodes, in_dim)
        self.inc_init = nn.Parameter(torch.empty(14, 14))
        nn.init.normal_(self.inc_init, 1, .01)

    def forward(self, x_node, SIR, od, max_od, only_params=False, apply_adaptor=False, report_mask=None):
        mobility = torch.einsum('kl,blnmc->bknmc', torch.softmax(self.inc_init, dim=1), od[:, :14]).squeeze(-1)
        graph = mobility.mean(1)
        transpose = graph.transpose(1, 2)
        graphs = [g / g.sum(2, keepdim=True).clamp_min(1e-12) for g in (graph, transpose)]
        beta, gamma = self.stcell(x_node, graphs)
        if apply_adaptor and hasattr(self, 'param_adaptor'):
            beta, gamma = self.param_adaptor(beta, gamma)
        if only_params:
            return beta, gamma, None
        self.recorded_beta, self.recorded_gamma = beta, gamma
        state = SIR[:, -1, :, :3]
        predictions = []
        for h in range(14):
            s, i, r = state.split(1, dim=-1)
            population = (s + i + r).expand(-1, s.shape[1], s.shape[1])
            mob = mobility[:, h] * max_od * 3.0
            expanded_i = i.expand_as(population)
            propagation = (mob / population * expanded_i).sum(1) + (mob / population * expanded_i.transpose(1, 2)).sum(2)
            incidence = beta[:, h] * propagation.unsqueeze(-1)
            state = torch.cat((s - incidence, i + incidence - i * gamma[:, h], r + i * gamma[:, h]), dim=-1)
            predictions.append(incidence)
        return torch.stack(predictions, dim=1)


class USMepoGNNAblation(CoSTEP):
    """US native_norm_no_curve: native encoder with interval-aware delayed rollout."""

    def __init__(self, num_nodes, in_dim):
        # Keep native encoder initialization aligned with the research model.
        with torch.random.fork_rng(devices=[]):
            super().__init__(num_nodes, in_dim=in_dim, curve=False,
                             stock_estimator='causal_interval_anchor')
        self.stcell = NativeEncoder(num_nodes, in_dim, out_len=28)

    def forward(self, x_node, SIR, od, max_od, only_params=False,
                apply_adaptor=False, report_mask=None):
        latency = self.get_latency_distribution(apply_adaptor=apply_adaptor)
        weights = latency.new_zeros((self.out_dim, self.in_len + self.out_dim))
        for horizon in range(self.out_dim):
            weights[horizon, horizon:self.in_len + horizon] = latency
        weights = torch.softmax(weights[:, :self.in_len] + 1e-8, dim=0)
        mobility = torch.einsum('kl,blnmc->bknmc', weights, od[:, :self.in_len]).squeeze(-1)
        graph = mobility.mean(1)
        graphs = [g / g.sum(2, keepdim=True) for g in (graph, graph.transpose(1, 2))]
        beta, gamma = self.stcell(x_node, graphs)
        beta_raw = beta
        beta, gamma = self._apply_param_adaptor(beta, gamma, apply_adaptor=apply_adaptor)
        self._record_beta_processing_stages(beta_raw, beta)
        self.recorded_gamma_pre_floor = gamma
        floor = compute_observation_gamma_floor(SIR[:, :self.in_len, :, -1:],
                                               SIR[:, :self.in_len, :, :1])
        self.recorded_gamma_floor = floor
        gamma = apply_gamma_floor(gamma, floor, straight_through=True)
        if only_params:
            return beta, gamma, None
        prediction = self.decode_parameters(SIR, od, max_od, beta, gamma,
                                            apply_adaptor=apply_adaptor, report_mask=report_mask)
        self.recorded_beta, self.recorded_gamma = beta, gamma
        return prediction
