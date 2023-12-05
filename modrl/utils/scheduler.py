
def linear_scheduler(epsilon_start: float, epsilon_end: float, curr: int, n_steps: int):
    slope = (epsilon_end - epsilon_start) / n_steps
    return max(slope * curr + epsilon_start, epsilon_end)