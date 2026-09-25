DEFAULT_LATENCY_PROFILE = "origin"
import torch
from torch import nn


class DynamicEpidemicStep(nn.Module):
    """MepoGNN SIRcell(modify=True), without a reporting delay."""

    def forward(self, beta, gamma, mobility, state):
        susceptible, infected, removed = state[..., :3].split(1, dim=-1)
        population = susceptible + infected + removed
        normalized = mobility / population
        destination_population = torch.einsum('bo,bod->bd', population.squeeze(-1), normalized)
        destination_infected = torch.einsum('bo,bod->bd', infected.squeeze(-1), normalized)
        pressure = destination_infected * (beta.squeeze(-1) / destination_population)
        incidence = torch.einsum('bo,bd,bod->bo', susceptible.squeeze(-1), pressure, normalized).unsqueeze(-1)
        return torch.cat((incidence, susceptible - incidence,
                          infected + incidence - infected * gamma,
                          removed + infected * gamma), dim=-1)

DEFAULT_LATENCY_SDLOG_MIN = 0.001
LATENCY_PROFILE_CONFIGS = {
    "origin": {"meanlog": 1.621, "sdlog": 0.418},
}
LATENCY_PROFILE_CHOICES = tuple(LATENCY_PROFILE_CONFIGS.keys())


def get_latency_profile_config(profile: str = DEFAULT_LATENCY_PROFILE) -> dict:
    if profile not in LATENCY_PROFILE_CONFIGS:
        raise ValueError(
            f"Unknown latency profile '{profile}'. Expected one of: {', '.join(LATENCY_PROFILE_CHOICES)}."
        )
    return dict(LATENCY_PROFILE_CONFIGS[profile])
