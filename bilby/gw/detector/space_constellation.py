"""Constellation geometry for moving space-based interferometers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from astropy.time import Time


AU_SI = 149597870700.0
C_SI = 299792458.0


class SpaceConstellation:
    """Base interface for a moving space-based detector constellation."""

    arm_length_m: float

    def spacecraft_positions(self, time):
        raise NotImplementedError

    def reference_position(self, time):
        positions = self.spacecraft_positions(time)
        return sum(positions) / len(positions)


@dataclass
class BDECIGOConstellation(SpaceConstellation):
    """Three-spacecraft B-DECIGO heliocentric triangular constellation.

    The relative geometry is the one used in the supplied
    ``B_DECIGO_unified.py``.  After constructing the three spacecraft
    positions, a common translation is applied so that the centroid follows an
    exactly circular 1-AU heliocentric orbit.  Since the translation is common
    to all spacecraft it leaves arm lengths and arm directions unchanged.

    Parameters
    ----------
    orbit_tstart:
        Offset, in seconds, applied to the SSB time before evaluating the
        orbital phase.
    arm_length_m:
        Nominal B-DECIGO arm length.  Default: 100 km.
    orbit_radius_m:
        Radius of the circular centroid orbit.  Default: 1 AU.
    """

    orbit_tstart: float = 0.0
    arm_length_m: float = 1.0e5
    orbit_radius_m: float = AU_SI

    _cache_time: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cache_positions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default=None, init=False, repr=False
    )

    VALID_VERTICES = ("Vertex-1", "Vertex-2", "Vertex-3")

    def _cached(self, time):
        return (
            self._cache_time is not None
            and self._cache_positions is not None
            and self._cache_time.shape == time.shape
            and np.array_equal(self._cache_time, time)
        )

    def spacecraft_positions(self, time):
        """Return spacecraft positions as three arrays with shape ``(N, 3)``.

        ``time`` is interpreted as an SSB time expressed on the same numeric
        GPS-second axis used by Bilby.  The use of ``astropy.time.Time`` here is
        solely to preserve the orbital phase convention of the supplied
        standalone implementation.
        """

        time = np.atleast_1d(np.asarray(time, dtype=float)).copy()
        time = time + float(self.orbit_tstart)

        if self._cached(time):
            return self._cache_positions

        time_jyear = np.atleast_1d(np.asarray(Time(val=time, format="gps", scale="utc").jyear))

        a = float(self.orbit_radius_m)
        L = float(self.arm_length_m)

        alpha = L / (2.0 * a)
        e = alpha / np.sqrt(3.0)
        epsilon = alpha

        omega_year = 2.0 * np.pi
        two_pi_over_3 = 2.0 * np.pi / 3.0
        orbital_phase = omega_year * time_jyear

        psi1 = orbital_phase - e * np.sin(orbital_phase)
        psi2 = orbital_phase - two_pi_over_3 - e * np.sin(orbital_phase - two_pi_over_3)
        psi3 = orbital_phase - 2.0 * two_pi_over_3 - e * np.sin(orbital_phase - 2.0 * two_pi_over_3)

        c2, s2 = np.cos(two_pi_over_3), np.sin(two_pi_over_3)
        c4, s4 = np.cos(2.0 * two_pi_over_3), np.sin(2.0 * two_pi_over_3)

        x1 = a * (np.cos(psi1) + e) * np.cos(epsilon)
        y1 = a * np.sqrt(1.0 - e**2) * np.sin(psi1)
        z1 = a * (np.cos(psi1) + e) * np.sin(epsilon)

        x2 = c2 * a * (np.cos(psi2) + e) * np.cos(epsilon) - s2 * a * np.sqrt(1.0 - e**2) * np.sin(
            psi2
        )
        y2 = s2 * a * (np.cos(psi2) + e) * np.cos(epsilon) + c2 * a * np.sqrt(1.0 - e**2) * np.sin(
            psi2
        )
        z2 = a * (np.cos(psi2) + e) * np.sin(epsilon)

        x3 = c4 * a * (np.cos(psi3) + e) * np.cos(epsilon) - s4 * a * np.sqrt(1.0 - e**2) * np.sin(
            psi3
        )
        y3 = s4 * a * (np.cos(psi3) + e) * np.cos(epsilon) + c4 * a * np.sqrt(1.0 - e**2) * np.sin(
            psi3
        )
        z3 = a * (np.cos(psi3) + e) * np.sin(epsilon)

        sc1 = np.column_stack((x1, y1, z1))
        sc2 = np.column_stack((x2, y2, z2))
        sc3 = np.column_stack((x3, y3, z3))

        raw_centroid = (sc1 + sc2 + sc3) / 3.0
        desired_centroid = np.column_stack(
            (
                a * np.cos(orbital_phase),
                a * np.sin(orbital_phase),
                np.zeros_like(orbital_phase),
            )
        )
        translation = desired_centroid - raw_centroid

        positions = (
            sc1 + translation,
            sc2 + translation,
            sc3 + translation,
        )

        self._cache_time = time.copy()
        self._cache_positions = positions
        return positions

    @staticmethod
    def centroid(positions):
        return (positions[0] + positions[1] + positions[2]) / 3.0

    def reference_position(self, time):
        return self.centroid(self.spacecraft_positions(time))

    @classmethod
    def vertex_indices(cls, vertex: str):
        """Return ``(active_vertex, arm1_end, arm2_end)`` indices.

        The ordering reproduces the supplied standalone B-DECIGO response:

        * Vertex-1: 1->2, 1->3
        * Vertex-2: 2->1, 2->3
        * Vertex-3: 3->2, 3->1
        """

        if vertex == "Vertex-1":
            return 0, 1, 2
        if vertex == "Vertex-2":
            return 1, 0, 2
        if vertex == "Vertex-3":
            return 2, 1, 0
        raise ValueError(f"Unknown B-DECIGO vertex {vertex!r}; choose one of {cls.VALID_VERTICES}.")

    def vertex_links(self, positions, vertex: str):
        i, j, k = self.vertex_indices(vertex)
        vertex_position = positions[i]

        arm1 = positions[j] - vertex_position
        arm2 = positions[k] - vertex_position

        link1 = arm1 / np.linalg.norm(arm1, axis=1)[:, None]
        link2 = arm2 / np.linalg.norm(arm2, axis=1)[:, None]
        return vertex_position, (link1, link2)


__all__ = ["SpaceConstellation", "BDECIGOConstellation", "AU_SI", "C_SI"]
