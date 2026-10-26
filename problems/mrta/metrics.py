"""Common deadline semantics for training, sequence replay and evaluation."""
import torch


def deadline_cost(finish_times, deadlines):
    ratios = finish_times / deadlines
    return torch.where(finish_times > deadlines, ratios, torch.zeros_like(ratios)).sum(-1)


def completed_on_time(finish_times, deadlines, visited):
    # Completion at the deadline is feasible, consistent with zero cost.
    return ((finish_times <= deadlines) & visited.bool()).sum(-1, keepdim=True)
