"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict

import torch

from fairchem.core.graph.radius_graph_pbc import (
    get_counts,
    radius_graph_pbc_v2,
    sum_partitions,
    system_pbc_flags,
    validate_graph_parameters,
)


class VerletNeighborCache:
    """
    Cache only an uncapped outer neighbor topology with an exact rebuild guard.

    Distances, cutoff filtering and neighbor capping are recomputed on every
    call. Cell/metadata changes and periodic wrapping invalidate the cache.
    Saved positions and topology are owned detached tensors, never a graph or
    a caller's mutable view. A bounded LRU limits the number of retained lists.
    """

    def __init__(
        self, skin: float = 0.5, max_entries: int = 1, max_bytes: int | None = None
    ):
        if not math.isfinite(skin) or skin <= 0:
            raise ValueError("neighbor skin must be finite and positive")
        self.skin = float(skin)
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 1
        ):
            raise ValueError("max_entries must be a positive integer")
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("max_bytes must be a nonnegative integer or None")
        self.max_bytes = max_bytes
        self.reused_graphs = 0
        self.max_entries = max_entries
        self.rebuilds = 0
        self.hits = 0
        self.tie_fallbacks = 0
        self.cap_fallbacks = 0
        self.boundary_fallbacks = 0
        self._entry = None
        self._entries = OrderedDict()
        self._generation = 0
        self._lock = threading.RLock()

    def __deepcopy__(self, memo):
        copied = type(self)(self.skin, self.max_entries, self.max_bytes)
        memo[id(self)] = copied
        return copied

    def __getstate__(self):
        # A model checkpoint must not retain derived topology, accelerator
        # events or the thread lock. Rebuild them on its destination device.
        return {
            "skin": self.skin,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def clear(self):
        """
        Release topology and snapshots without changing accumulated counters.
        """
        with self._lock:
            self._entry = None
            self._entries.clear()

    @property
    def stats(self):
        with self._lock:
            entries = self._entries.values()
            return {
                "rebuilds": self.rebuilds,
                "hits": self.hits,
                "reused_graphs": self.reused_graphs,
                "tie_fallbacks": self.tie_fallbacks,
                "cap_fallbacks": self.cap_fallbacks,
                "boundary_fallbacks": self.boundary_fallbacks,
                "candidate_edges": sum(e["edges"].shape[1] for e in entries),
                "retained_bytes": sum(e["bytes"] for e in entries),
                "entries": len(entries),
            }

    @staticmethod
    def _metadata(data, pbc):
        values = {
            name: getattr(data, name, None)
            for name in ("cell", "natoms", "batch", "atomic_numbers", "node_partition")
        }
        if values["node_partition"] is not None:
            # Match graph-v2's accepted index containers before snapshotting.
            # NumPy/list partitions must work with caching just as without it.
            values["node_partition"] = torch.as_tensor(
                values["node_partition"], device=data.pos.device
            )
        values["pbc"] = data.pbc if pbc is None else pbc
        return values

    @staticmethod
    def _matches(entry, current, pos, skin):
        checks = []
        for name, value in current.items():
            old = entry["metadata"][name]
            if value is None or old is None:
                if value is not old:
                    return False
            elif (
                value.shape != old.shape
                or value.dtype != old.dtype
                or value.device != old.device
            ):
                return False
            else:
                checks.append((value == old).all().to(pos.device))
        if len(pos):
            displacement = (pos - entry["pos"]).norm(dim=-1).max()
            checks.append(
                2 * displacement < skin * (1 - 32 * torch.finfo(pos.dtype).eps)
            )
        # Keep exact value comparisons, including NumPy-backed tensors, while
        # synchronizing once per viable entry instead of once per field.
        return bool(torch.stack(checks).all())

    @staticmethod
    def _stream_event(device):
        backend = getattr(torch, device.type, None)
        if device.type not in {"cuda", "xpu"}:
            return None
        event = backend.Event()
        event.record(backend.current_stream(device))
        return event

    @torch.no_grad()
    def get_graph(
        self, data, cutoff, max_neighbors, strict=False, *, pbc=None, method="cell"
    ):
        """
        Return the current graph, rebuilding the outer list when required.
        """
        from fairchem.core.graph.compute import edge_cell_offsets

        cutoff, max_neighbors, strict = validate_graph_parameters(
            cutoff, max_neighbors, strict, method
        )
        pbc = system_pbc_flags(data, pbc)
        with self._lock:
            if not math.isfinite(self.skin) or self.skin <= 0:
                raise ValueError("neighbor skin must be finite and positive")
            key = (
                float(self.skin),
                float(cutoff),
                int(max_neighbors),
                bool(strict),
                method,
                data.pos.device,
                data.pos.dtype,
                tuple(data.pos.shape),
            )
            metadata = self._metadata(data, pbc)
            entry = None
            for generation, candidate in reversed(self._entries.items()):
                if candidate["key"] != key:
                    continue
                if candidate["event"] is not None:
                    stream = getattr(torch, data.pos.device.type).current_stream(
                        data.pos.device
                    )
                    stream.wait_event(candidate["event"])
                    for tensor in [
                        candidate["pos"],
                        candidate["edges"],
                        candidate["cells"],
                        candidate["counts"],
                        *candidate["metadata"].values(),
                    ]:
                        if tensor is not None and tensor.device == data.pos.device:
                            tensor.record_stream(stream)
                if self._matches(candidate, metadata, data.pos, self.skin):
                    entry = candidate
                    self._entries.move_to_end(generation)
                    break
            valid = entry is not None
            if not valid:
                margin = (
                    32 * torch.finfo(data.pos.dtype).eps * max(1.0, cutoff + self.skin)
                )
                edges, cells, counts = radius_graph_pbc_v2(
                    data,
                    cutoff + self.skin + margin,
                    0,
                    pbc=pbc,
                    method=method,
                    include_zero_distance=True,
                )
                entry = {
                    "key": key,
                    "metadata": {
                        name: None if value is None else value.detach().clone()
                        for name, value in metadata.items()
                    },
                    "pos": data.pos.detach().clone(),
                    # Graph-v2 constructs owned output tensors. Retain those
                    # directly instead of briefly holding two full outer lists
                    # during each rebuild; public outputs are selected below.
                    "edges": edges.detach(),
                    "cells": cells.detach(),
                    "counts": counts.detach(),
                    "event": None,
                }
                entry["event"] = self._stream_event(data.pos.device)
                entry["bytes"] = sum(
                    t.untyped_storage().nbytes()
                    for t in (
                        entry["pos"],
                        entry["edges"],
                        entry["cells"],
                        entry["counts"],
                        *entry["metadata"].values(),
                    )
                    if t is not None
                )
                if self.max_bytes is None or entry["bytes"] <= self.max_bytes:
                    self._entries[self._generation] = entry
                    self._generation += 1
                    total = sum(e["bytes"] for e in self._entries.values())
                    while len(self._entries) > self.max_entries or (
                        self.max_bytes is not None and total > self.max_bytes
                    ):
                        _, evicted = self._entries.popitem(last=False)
                        total -= evicted["bytes"]
                    self._entry = entry
                self.rebuilds += 1
            else:
                self.hits += 1
                self._entry = entry
        edges, cells = entry["edges"], entry["cells"]
        shifts = edge_cell_offsets(data.cell, cells, entry["counts"])
        # Advanced indexing owns these endpoint buffers. Reuse them for the
        # roundoff bound instead of gathering both coordinates a second time.
        source_pos = data.pos[edges[0]]
        target_pos = data.pos[edges[1]]
        # Match graph-v2's subtraction order, not get_pbc_distances' order.
        distance2 = (target_pos - (source_pos + shifts)).square().sum(-1)
        # Last-bit differences between batched offset products must not
        # decide whether a cutoff-boundary edge exists. Ambiguous cases
        # use the fresh graph's arithmetic and traversal.
        rounding = (
            16
            * torch.finfo(data.pos.dtype).eps
            * source_pos.abs_().add_(target_pos.abs_()).add_(shifts.abs()).sum(-1)
        )
        del source_pos, target_pos, shifts
        distance_error = rounding * (2 * distance2.sqrt() + rounding)
        del rounding
        ambiguous = (distance2 - cutoff * cutoff).abs() <= distance_error
        # Wrapped search and original-frame arithmetic can also disagree
        # at zero for translated atoms. Self loops in the primary image
        # are exactly zero in both frames.
        ambiguous |= (distance2 <= distance_error) & cells.ne(0).any(-1)
        if bool(ambiguous.any()):
            with self._lock:
                self.boundary_fallbacks += 1
            return radius_graph_pbc_v2(
                data, cutoff, max_neighbors, strict, pbc=pbc, method=method
            )
        keep = (distance2 <= cutoff * cutoff) & (distance2 != 0)
        edges, cells, distance2 = edges[:, keep], cells[keep], distance2[keep]
        distance_error = distance_error[keep]
        del keep
        degrees = get_counts(edges[1], len(data.pos))
        if max_neighbors > 0 and len(distance2) > 1:
            overfull = degrees > max_neighbors
            if bool(overfull.any()):
                # Order statistics move by at most the largest input error.
                # Guard both strict k/k+1 selection and the non-strict
                # (k+1)th-distance + 0.01 degeneracy threshold. Checking
                # exact equalities alone would miss near-degenerate pairs.
                selected = torch.nonzero(overfull[edges[1]]).flatten()
                order = selected[torch.argsort(distance2[selected], stable=True)]
                order = order[torch.argsort(edges[1, order], stable=True)]
                starts = degrees[overfull].cumsum(0) - degrees[overfull]
                error_per_atom = distance_error.new_zeros(len(data.pos))
                error_per_atom.scatter_reduce_(
                    0, edges[1], distance_error, reduce="amax"
                )
                if strict:
                    gap = (
                        distance2[order[starts + max_neighbors]]
                        - distance2[order[starts + max_neighbors - 1]]
                    )
                    ambiguous = bool((gap <= 2 * error_per_atom[overfull]).any())
                else:
                    thresholds = distance2.new_full((len(data.pos),), torch.inf)
                    thresholds[overfull] = (
                        distance2[order[starts + max_neighbors]] + 0.01
                    )
                    ambiguous = bool(
                        (
                            (distance2 - thresholds[edges[1]]).abs()
                            <= distance_error + error_per_atom[edges[1]]
                        ).any()
                    )
                if ambiguous:
                    with self._lock:
                        self.cap_fallbacks += 1
                        self.tie_fallbacks += int(strict)
                    return radius_graph_pbc_v2(
                        data, cutoff, max_neighbors, strict, pbc=pbc, method=method
                    )
                # The guarded sort already identified each atom's cutoff.
                # Distances at a strict selection boundary are separated,
                # so thresholding has the same tie semantics as fresh search.
                if strict:
                    thresholds = distance2.new_full((len(data.pos),), torch.inf)
                    thresholds[overfull] = distance2[order[starts + max_neighbors - 1]]
                mask = distance2 <= thresholds[edges[1]]
                if strict:
                    degrees.clamp_max_(max_neighbors)
                else:
                    degrees = get_counts(edges[1, mask], len(data.pos))
                edges, cells = edges[:, mask], cells[mask]
        indptr = torch.cat((data.natoms.new_zeros(1), data.natoms.cumsum(0)))
        counts = sum_partitions(degrees, indptr)
        if valid:
            with self._lock:
                self.reused_graphs += 1
        return edges, cells, counts
