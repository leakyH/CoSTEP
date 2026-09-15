DEFAULT_LATENCY_PROFILE = "origin"
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
