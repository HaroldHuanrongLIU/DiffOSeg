import unittest
import torch
from reproduction.preference_gradients import projected_mean


class PreferenceGradientTests(unittest.TestCase):
    def test_opposing_components_match_analytic_projection(self):
        gradients = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])
        orders = torch.tensor([[1], [0]])
        torch.testing.assert_close(projected_mean(gradients, orders), torch.tensor([0.25, 0.75]))

    def test_nonconflicting_and_zero_tasks_keep_the_original_mean(self):
        gradients = torch.tensor([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0]])
        orders = torch.tensor([[1, 2], [2, 0], [0, 1]])
        torch.testing.assert_close(projected_mean(gradients, orders), gradients.mean(0))

    def test_shared_gradient_delta_preserves_an_auxiliary_gradient(self):
        tasks = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])
        auxiliary = torch.tensor([0.7, -0.4])
        orders = torch.tensor([[1], [0]])
        ordinary = tasks.mean(0) + auxiliary
        corrected = ordinary + projected_mean(tasks, orders) - tasks.mean(0)
        torch.testing.assert_close(corrected, torch.tensor([0.25, 0.75]) + auxiliary)
