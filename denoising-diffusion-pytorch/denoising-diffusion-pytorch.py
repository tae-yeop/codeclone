


def linear_beta_scheduler(timesteps):
    """
    linear scheduler
    [0.0001, ..., 0.02000] torch.float64
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

