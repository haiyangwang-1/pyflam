"""Structured operator adapters for channel-aware FLAM compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from .core import StructMixin, _as_points, chksymm
from .rskelf import RSkelFFactor, rskelf

TensorKernel = Callable[[np.ndarray, np.ndarray, int, int], ArrayLike]
ProxyPointRule = Callable[[np.ndarray, np.ndarray, "TensorInteraction", int, int, str], ArrayLike]


@dataclass
class DofSpace(StructMixin):
    """Point-supported degrees of freedom for one row or column space.

    Components are ordered component-major within a space:
    ``flat_id = offset + component * point_count + point_id``.
    """

    name: str
    points: ArrayLike
    component_count: int = 1

    def __post_init__(self) -> None:
        self.points = _as_points(self.points)
        self.component_count = int(self.component_count)
        if self.component_count < 1:
            raise ValueError("component_count must be positive")

    @property
    def point_count(self) -> int:
        return int(self.points.shape[1])

    @property
    def coordinate_dim(self) -> int:
        return int(self.points.shape[0])


@dataclass
class DofLayout(StructMixin):
    """Flat indexing metadata for a collection of point-supported spaces."""

    spaces: tuple[DofSpace, ...]
    offsets: np.ndarray = field(init=False)
    total_dofs: int = field(init=False)
    coordinate_dim: int = field(init=False)
    points_by_dof: np.ndarray = field(init=False)
    space_by_dof: np.ndarray = field(init=False)
    component_by_dof: np.ndarray = field(init=False)
    point_by_dof: np.ndarray = field(init=False)
    _name_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.spaces = tuple(self.spaces)
        if not self.spaces:
            raise ValueError("DofLayout requires at least one DofSpace")

        dims = {space.coordinate_dim for space in self.spaces}
        if len(dims) != 1:
            raise ValueError("all DofSpace objects in a layout must use the same coordinate dimension")
        self.coordinate_dim = dims.pop()

        names = [space.name for space in self.spaces]
        if len(set(names)) != len(names):
            raise ValueError("DofSpace names must be unique within a layout")
        self._name_to_index = {name: idx for idx, name in enumerate(names)}

        offsets = []
        total = 0
        for space in self.spaces:
            offsets.append(total)
            total += space.point_count * space.component_count
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.total_dofs = int(total)

        self.points_by_dof = np.empty((self.coordinate_dim, self.total_dofs), dtype=np.result_type(*[s.points for s in self.spaces]))
        self.space_by_dof = np.empty(self.total_dofs, dtype=np.int64)
        self.component_by_dof = np.empty(self.total_dofs, dtype=np.int64)
        self.point_by_dof = np.empty(self.total_dofs, dtype=np.int64)

        for space_idx, space in enumerate(self.spaces):
            start = int(self.offsets[space_idx])
            stop = start + space.point_count * space.component_count
            component_ids = np.repeat(np.arange(space.component_count, dtype=np.int64), space.point_count)
            point_ids = np.tile(np.arange(space.point_count, dtype=np.int64), space.component_count)
            self.points_by_dof[:, start:stop] = np.tile(space.points, space.component_count)
            self.space_by_dof[start:stop] = space_idx
            self.component_by_dof[start:stop] = component_ids
            self.point_by_dof[start:stop] = point_ids

    def space_index(self, space: int | str) -> int:
        if isinstance(space, str):
            try:
                return self._name_to_index[space]
            except KeyError as exc:
                raise ValueError(f"unknown DofSpace name {space!r}") from exc
        idx = int(space)
        if idx < 0 or idx >= len(self.spaces):
            raise ValueError(f"DofSpace index {idx} is out of range")
        return idx

    def dofs(self, space: int | str, component: int, points: ArrayLike | None = None) -> np.ndarray:
        space_idx = self.space_index(space)
        space_obj = self.spaces[space_idx]
        component = int(component)
        if component < 0 or component >= space_obj.component_count:
            raise ValueError(f"component {component} is out of range for space {space_obj.name!r}")
        point_ids = (
            np.arange(space_obj.point_count, dtype=np.int64)
            if points is None
            else np.asarray(points, dtype=np.int64).reshape(-1)
        )
        return self.offsets[space_idx] + component * space_obj.point_count + point_ids


@dataclass
class TensorInteraction(StructMixin):
    """One tensor-valued block interaction between row and column spaces.

    ``kernel(target_points, source_points, output_component, input_component)``
    must return a scalar channel matrix with shape
    ``(target_points.shape[1], source_points.shape[1])``.
    """

    row_space: int | str
    col_space: int | str
    kernel: TensorKernel
    coefficient: complex = 1.0
    row_components: tuple[int, ...] | None = None
    col_components: tuple[int, ...] | None = None
    proxy_kernel: TensorKernel | None = None
    proxy_points: ProxyPointRule | None = None


@dataclass
class StructuredOperator(StructMixin):
    """Block/tensor operator that preserves channel structure for proxy IDs."""

    row_layout: DofLayout
    col_layout: DofLayout
    interactions: tuple[TensorInteraction, ...]
    proxy_points: ProxyPointRule | None = None
    dtype: Any = float
    factor_points: ArrayLike | None = None

    def __post_init__(self) -> None:
        self.interactions = tuple(self.interactions)
        if not self.interactions:
            raise ValueError("StructuredOperator requires at least one TensorInteraction")
        if self.row_layout.coordinate_dim != self.col_layout.coordinate_dim:
            raise ValueError("row_layout and col_layout must use the same coordinate dimension")
        if self.factor_points is not None:
            self.factor_points = _as_points(self.factor_points)

    @property
    def shape(self) -> tuple[int, int]:
        return self.row_layout.total_dofs, self.col_layout.total_dofs

    def factor_coordinates(self) -> np.ndarray:
        if self.row_layout.total_dofs != self.col_layout.total_dofs:
            raise ValueError("rskelf_structured requires equal row and column dof counts")
        if self.factor_points is not None:
            pts = np.asarray(self.factor_points)
            if pts.shape[1] != self.col_layout.total_dofs:
                raise ValueError("factor_points must have one column per operator dof")
            return pts
        if not np.allclose(self.row_layout.points_by_dof, self.col_layout.points_by_dof):
            raise ValueError("row and column coordinates differ; pass explicit factor_points")
        return self.row_layout.points_by_dof

    def submatrix(self, rows: ArrayLike, cols: ArrayLike) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        if np.any((rows < 0) | (rows >= self.row_layout.total_dofs)):
            raise IndexError("row index out of bounds")
        if np.any((cols < 0) | (cols >= self.col_layout.total_dofs)):
            raise IndexError("column index out of bounds")

        out: np.ndarray | None = None
        for interaction in self.interactions:
            row_space = self.row_layout.space_index(interaction.row_space)
            col_space = self.col_layout.space_index(interaction.col_space)
            for out_component in self._components(self.row_layout, row_space, interaction.row_components):
                row_pos = self._selection_positions(rows, self.row_layout, row_space, out_component)
                if row_pos.size == 0:
                    continue
                target_points = self._points_for_positions(rows, row_pos, self.row_layout)
                for in_component in self._components(self.col_layout, col_space, interaction.col_components):
                    col_pos = self._selection_positions(cols, self.col_layout, col_space, in_component)
                    if col_pos.size == 0:
                        continue
                    source_points = self._points_for_positions(cols, col_pos, self.col_layout)
                    block = self._evaluate(
                        interaction.kernel,
                        target_points,
                        source_points,
                        out_component,
                        in_component,
                    )
                    block = interaction.coefficient * block
                    out = self._accumulate(out, rows.size, cols.size, row_pos, col_pos, block)

        if out is None:
            return np.zeros((rows.size, cols.size), dtype=self.dtype)
        return out

    def rskelf_proxy_callback(self, symm: str = "n") -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        symm = chksymm(symm)

        def pxyfun(x: np.ndarray, slf: np.ndarray, nbr: np.ndarray, box_size: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            del x
            return self.proxy_sample(slf, nbr, box_size, center, symm=symm)

        return pxyfun

    def proxy_sample(
        self,
        slf: ArrayLike,
        nbr: ArrayLike,
        box_size: ArrayLike,
        center: ArrayLike,
        *,
        symm: str = "n",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return channel-wise proxy rows for a square rskelf column cluster."""

        slf = np.asarray(slf, dtype=np.int64).reshape(-1)
        nbr = np.asarray(nbr, dtype=np.int64).reshape(-1)
        box_size = np.asarray(box_size)
        center = np.asarray(center)
        parts: list[np.ndarray] = []
        for interaction in self.interactions:
            parts.extend(self._proxy_column_parts(interaction, slf, box_size, center))
            if chksymm(symm) == "n":
                parts.extend(self._proxy_row_parts(interaction, slf, box_size, center))
        if not parts:
            return np.zeros((0, slf.size), dtype=self.dtype), nbr
        return np.vstack(parts), nbr

    def _proxy_column_parts(
        self,
        interaction: TensorInteraction,
        slf: np.ndarray,
        box_size: np.ndarray,
        center: np.ndarray,
    ) -> list[np.ndarray]:
        col_space = self.col_layout.space_index(interaction.col_space)
        parts = []
        for in_component in self._components(self.col_layout, col_space, interaction.col_components):
            col_pos = self._selection_positions(slf, self.col_layout, col_space, in_component)
            if col_pos.size == 0:
                continue
            source_points = self._points_for_positions(slf, col_pos, self.col_layout)
            for out_component in self._components(
                self.row_layout,
                self.row_layout.space_index(interaction.row_space),
                interaction.row_components,
            ):
                proxy_points = self._proxy_points(interaction, box_size, center, out_component, in_component, "target")
                block = self._evaluate(
                    interaction.proxy_kernel or interaction.kernel,
                    proxy_points,
                    source_points,
                    out_component,
                    in_component,
                )
                block = interaction.coefficient * block
                full = np.zeros((block.shape[0], slf.size), dtype=np.result_type(self.dtype, block))
                full[:, col_pos] = block
                parts.append(full)
        return parts

    def _proxy_row_parts(
        self,
        interaction: TensorInteraction,
        slf: np.ndarray,
        box_size: np.ndarray,
        center: np.ndarray,
    ) -> list[np.ndarray]:
        row_space = self.row_layout.space_index(interaction.row_space)
        parts = []
        for out_component in self._components(self.row_layout, row_space, interaction.row_components):
            row_pos = self._selection_positions(slf, self.row_layout, row_space, out_component)
            if row_pos.size == 0:
                continue
            target_points = self._points_for_positions(slf, row_pos, self.row_layout)
            for in_component in self._components(
                self.col_layout,
                self.col_layout.space_index(interaction.col_space),
                interaction.col_components,
            ):
                proxy_points = self._proxy_points(interaction, box_size, center, out_component, in_component, "source")
                block = self._evaluate(
                    interaction.proxy_kernel or interaction.kernel,
                    target_points,
                    proxy_points,
                    out_component,
                    in_component,
                )
                block = interaction.coefficient * block
                full = np.zeros((block.shape[1], slf.size), dtype=np.result_type(self.dtype, block))
                full[:, row_pos] = np.conj(block).T
                parts.append(full)
        return parts

    def _proxy_points(
        self,
        interaction: TensorInteraction,
        box_size: np.ndarray,
        center: np.ndarray,
        out_component: int,
        in_component: int,
        side: str,
    ) -> np.ndarray:
        rule = interaction.proxy_points or self.proxy_points
        if rule is None:
            raise ValueError("StructuredOperator proxy compression requires proxy_points")
        return _as_points(rule(box_size, center, interaction, out_component, in_component, side))

    @staticmethod
    def _components(layout: DofLayout, space: int, requested: tuple[int, ...] | None) -> tuple[int, ...]:
        count = layout.spaces[space].component_count
        components = tuple(range(count)) if requested is None else tuple(int(c) for c in requested)
        if any(c < 0 or c >= count for c in components):
            raise ValueError(f"component selection {components!r} is out of range for space {layout.spaces[space].name!r}")
        return components

    @staticmethod
    def _selection_positions(indices: np.ndarray, layout: DofLayout, space: int, component: int) -> np.ndarray:
        return np.flatnonzero(
            (layout.space_by_dof[indices] == space) & (layout.component_by_dof[indices] == component)
        )

    @staticmethod
    def _points_for_positions(indices: np.ndarray, positions: np.ndarray, layout: DofLayout) -> np.ndarray:
        return layout.points_by_dof[:, indices[positions]]

    @staticmethod
    def _evaluate(
        kernel: TensorKernel,
        target_points: np.ndarray,
        source_points: np.ndarray,
        out_component: int,
        in_component: int,
    ) -> np.ndarray:
        block = np.asarray(kernel(target_points, source_points, out_component, in_component))
        expected = (target_points.shape[1], source_points.shape[1])
        if block.shape != expected:
            raise ValueError(f"kernel channel returned shape {block.shape}, expected {expected}")
        return block

    def _accumulate(
        self,
        out: np.ndarray | None,
        row_count: int,
        col_count: int,
        row_pos: np.ndarray,
        col_pos: np.ndarray,
        block: np.ndarray,
    ) -> np.ndarray:
        if out is None:
            out = np.zeros((row_count, col_count), dtype=np.result_type(self.dtype, block))
        elif not np.can_cast(block.dtype, out.dtype, casting="same_kind"):
            out = out.astype(np.result_type(out.dtype, block.dtype), copy=False)
        out[np.ix_(row_pos, col_pos)] += block
        return out


def rskelf_structured(
    operator: StructuredOperator,
    occ: int,
    rank_or_tol: float,
    opts: dict[str, Any] | None = None,
) -> RSkelFFactor:
    """Factor a square structured operator with channel-aware proxy sampling."""

    opts_in = dict(opts or {})
    symm = chksymm(next((value for key, value in opts_in.items() if key.lower() == "symm"), "n"))
    return rskelf(
        operator.submatrix,
        operator.factor_coordinates(),
        occ,
        rank_or_tol,
        pxyfun=operator.rskelf_proxy_callback(symm),
        opts=opts,
    )
