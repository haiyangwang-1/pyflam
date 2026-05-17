import unittest

import numpy as np

from pyflam import (
    DofLayout,
    DofSpace,
    StructuredOperator,
    TensorInteraction,
    rskelf_mv,
    rskelf_structured,
)


def _channel_kernel(target_points, source_points, out_component, in_component):
    target = target_points.ravel()
    source = source_points.ravel()
    scale = np.array([[1.0, -0.35], [0.55, 0.8]])[out_component, in_component]
    width = 0.18 + 0.05 * out_component + 0.03 * in_component
    block = scale * np.exp(-np.abs(target[:, None] - source[None, :]) / width)
    if out_component == in_component:
        block = block + 2.5 * np.isclose(target[:, None], source[None, :])
    return block


class StructuredOperatorTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0.0, 1.0, 18).reshape(1, -1)
        self.layout = DofLayout((DofSpace("boundary", x, component_count=2),))

    def test_submatrix_evaluates_component_major_tensor_blocks(self):
        operator = StructuredOperator(
            self.layout,
            self.layout,
            (TensorInteraction("boundary", "boundary", _channel_kernel, coefficient=1.0 + 0.25j),),
            proxy_points=self.proxy_points,
        )
        rows = np.r_[self.layout.dofs("boundary", 1, [0, 3]), self.layout.dofs("boundary", 0, [2])]
        cols = np.r_[self.layout.dofs("boundary", 0, [1, 4]), self.layout.dofs("boundary", 1, [3])]

        actual = operator.submatrix(rows, cols)
        expected = np.empty((rows.size, cols.size), dtype=complex)
        for row_pos, row in enumerate(rows):
            out_component = self.layout.component_by_dof[row]
            target = self.layout.points_by_dof[:, [row]]
            for col_pos, col in enumerate(cols):
                in_component = self.layout.component_by_dof[col]
                source = self.layout.points_by_dof[:, [col]]
                expected[row_pos, col_pos] = (1.0 + 0.25j) * _channel_kernel(
                    target,
                    source,
                    out_component,
                    in_component,
                )[0, 0]

        np.testing.assert_allclose(actual, expected)

    def test_rskelf_structured_samples_each_proxy_channel_before_id(self):
        proxy_calls = []

        def proxy_kernel(target_points, source_points, out_component, in_component):
            proxy_calls.append((out_component, in_component, target_points.shape[1], source_points.shape[1]))
            return _channel_kernel(target_points, source_points, out_component, in_component)

        operator = StructuredOperator(
            self.layout,
            self.layout,
            (
                TensorInteraction(
                    "boundary",
                    "boundary",
                    _channel_kernel,
                    proxy_kernel=proxy_kernel,
                ),
            ),
            proxy_points=self.proxy_points,
        )
        dense = operator.submatrix(np.arange(operator.shape[0]), np.arange(operator.shape[1]))
        rhs = np.random.default_rng(42).standard_normal((operator.shape[1], 3))

        factor = rskelf_structured(operator, occ=4, rank_or_tol=1e-12, opts={"symm": "n"})

        self.assertEqual({call[:2] for call in proxy_calls}, {(0, 0), (0, 1), (1, 0), (1, 1)})
        self.assertGreater(len(factor.factors), 0)
        np.testing.assert_allclose(rskelf_mv(factor, rhs), dense @ rhs, rtol=1e-9, atol=1e-9)

    @staticmethod
    def proxy_points(box_size, center, interaction, out_component, in_component, side):
        del interaction, out_component, in_component, side
        half_width = np.max(np.asarray(box_size)) / 2.0
        offsets = np.array([-1.75, -1.25, 1.25, 1.75]) * half_width
        return np.asarray(center).reshape(-1, 1) + offsets.reshape(1, -1)


if __name__ == "__main__":
    unittest.main()
