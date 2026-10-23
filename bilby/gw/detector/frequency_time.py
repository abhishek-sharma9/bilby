"""Frequency-to-time map for moving space-based detectors.

This module is deliberately independent of any particular constellation.

Bilby parameter aliases
-----------------------
The mapper accepts the common Bilby names

    mass_1, mass_2, chi_1z, chi_2z, lambda_1, lambda_2

and a few aliases used by PyCBC/standalone scripts.  If ``chi_1z`` / ``chi_2z``
are absent but ``a_1, tilt_1`` / ``a_2, tilt_2`` are present, aligned spin
components are constructed as a*cos(tilt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import lal
import numpy as np
from scipy.interpolate import CubicSpline


_MISSING = object()


def _get_first(parameters: Dict[str, Any], names, default=_MISSING):
    for name in names:
        if name in parameters:
            return parameters[name]
    if default is not _MISSING:
        return default
    raise KeyError(
        f"None of the parameter aliases {tuple(names)} are present in the parameter dictionary."
    )


def _aligned_spin(parameters: Dict[str, Any], body: int) -> float:
    aliases = (
        f"chi_{body}z",
        f"chi{body}z",
        f"spin_{body}z",
        f"spin{body}z",
    )
    for name in aliases:
        if name in parameters:
            return float(parameters[name])

    a_name = f"a_{body}"
    tilt_name = f"tilt_{body}"
    if a_name in parameters and tilt_name in parameters:
        return float(parameters[a_name]) * np.cos(float(parameters[tilt_name]))

    return 0.0


def _extract_intrinsic_parameters(parameters: Dict[str, Any]):
    mass1 = float(_get_first(parameters, ("mass_1", "mass1")))
    mass2 = float(_get_first(parameters, ("mass_2", "mass2")))
    chi1 = _aligned_spin(parameters, 1)
    chi2 = _aligned_spin(parameters, 2)
    lambda1 = float(_get_first(parameters, ("lambda_1", "lambda1"), default=0.0))
    lambda2 = float(_get_first(parameters, ("lambda_2", "lambda2"), default=0.0))
    return mass1, mass2, chi1, chi2, lambda1, lambda2


def TaylorF2Phasing_6PNQM2SCoeff(mByM):
    return (4703.5 / 8.4 + 2935.0 / 6.0 * mByM - 120.0 * mByM * mByM) * mByM * mByM


def TaylorF2Phasing_6PNSelf2SCoeff(mByM):
    return (-4108.25 / 6.72 - 108.5 / 1.2 * mByM + 125.5 / 3.6 * mByM * mByM) * mByM * mByM


def TaylorF2Phasing_6PNSOCoeff(mByM):
    return np.pi * mByM * (1490.0 / 3.0 + mByM * 260.0)


def TaylorF2Phasing_6PNS1S2OCoeff(eta):
    return (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta


def Phase_deriv_wrt_f(f, mass1, mass2, chi1, chi2, lambda1=0.0, lambda2=0.0):
    """Derivative of the TaylorF2 phase used by the original B-DECIGO code."""

    f = np.asarray(f, dtype=float)

    M = mass1 + mass2
    m1ByM = mass1 / M
    m2ByM = mass2 / M
    eta = min(mass1 * mass2 / M**2, 0.25)
    delta = (mass1 - mass2) / M
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0

    Msec = M * lal.MTSUN_SI
    v = (np.pi * Msec * f) ** (1.0 / 3.0)
    commonFactor = 3.0 * (np.pi * Msec) / (128.0 * eta)

    Psi0_prime = (-5.0 / 3.0) * v ** (-8)
    Psi2_prime = (3715.0 / 756.0 + 55.0 * eta / 9.0) * (-1.0) * v ** (-6)

    Psi3_prime = (
        (-16.0 * np.pi + 113.0 * delta * chi_a / 3.0 + (113.0 / 3.0 - 76.0 * eta / 3.0) * chi_s)
        * (-2.0 / 3.0)
        * v ** (-5)
    )

    Psi4_prime = (
        (
            (15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta**2 / 72.0)
            + (-405.0 / 8.0 + 200.0 * eta) * chi_a**2
            - (405.0 / 4.0) * delta * chi_a * chi_s
            + (-405.0 / 8.0 + (5.0 / 2.0) * eta) * chi_s**2
        )
        * (-1.0 / 3.0)
        * v ** (-4)
    )

    Psi6_prime = (
        (
            (
                11583.231236531 / 4.694215680
                - 640.0 / 3.0 * np.pi**2
                - 684.8 / 2.1 * lal.GAMMA
                + eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * np.pi**2)
                + eta**2 * 76.055 / 1.728
                - eta**3 * 127.825 / 1.296
            )
            + TaylorF2Phasing_6PNSOCoeff(m1ByM) * chi1
            + TaylorF2Phasing_6PNSOCoeff(m2ByM) * chi2
            + TaylorF2Phasing_6PNS1S2OCoeff(eta) * chi1 * chi2
            + (TaylorF2Phasing_6PNQM2SCoeff(m1ByM) + TaylorF2Phasing_6PNSelf2SCoeff(m1ByM))
            * chi1**2
            + (TaylorF2Phasing_6PNQM2SCoeff(m2ByM) + TaylorF2Phasing_6PNSelf2SCoeff(m2ByM))
            * chi2**2
            + (-6848.0 / 21.0) * np.log(4.0 * v)
        )
        * (1.0 / 3.0)
        * v ** (-2)
    )

    Psi7_prime = (
        (
            (
                77096675.0 * np.pi / 254016.0
                + 378515.0 * np.pi * eta / 1512.0
                - 74045.0 * np.pi * eta**2 / 756.0
            )
            + delta
            * (-25150083775.0 / 3048192.0 + (26804935.0 / 6048.0) * eta - (1985.0 / 48.0) * eta**2)
            * chi_a
            + (
                -25150083775.0 / 3048192.0
                + (10566655595.0 / 762048.0) * eta
                - (1042165.0 / 3024.0) * eta**2
                + (5345.0 / 36.0) * eta**3
            )
            * chi_s
        )
        * (2.0 / 3.0)
        * v ** (-1)
    )

    dpsidf_vacuum = commonFactor * (
        Psi0_prime + Psi2_prime + Psi3_prime + Psi4_prime + Psi6_prime + Psi7_prime
    )

    lam_t = (
        8.0
        / 13.0
        * (
            (1.0 + 7.0 * eta - 31.0 * eta**2) * (lambda1 + lambda2)
            + delta * (1.0 + 9.0 * eta - 11.0 * eta**2) * (lambda1 - lambda2)
        )
    )

    delta_lam_t = 0.5 * (
        delta * (1319.0 - 13272.0 * eta + 8944.0 * eta**2) / 1319.0 * (lambda1 + lambda2)
        + (1319.0 - 15910.0 * eta + 32850.0 * eta**2 + 3380.0 * eta**3)
        / 1319.0
        * (lambda1 - lambda2)
    )

    pT10_prime = -39.0 * lam_t * (5.0 / 3.0) * v**2
    pT12_prime = (
        (-3115.0 * lam_t / 64.0 + 6595.0 * np.sqrt(max(0.0, 1.0 - 4.0 * eta)) * delta_lam_t / 364.0)
        * (7.0 / 3.0)
        * v**4
    )

    dpsi_df = dpsidf_vacuum + commonFactor * (pT10_prime + pT12_prime)

    # Match the standalone implementation: the highest frequency used to
    # build this array is the zero of the relative time coordinate.
    return -(dpsi_df - dpsi_df[-1])


@dataclass
class TaylorF2FrequencyTimeMapper:
    """Map Fourier frequency to time relative to the reference/merger time.

    The returned array is the quantity that the standalone B-DECIGO code calls
    ``t_of_f``.  It can be added directly to Bilby's ``geocent_time`` key,
    which in this space-detector implementation is intentionally interpreted
    as an SSB reference time.

    A one-entry spline cache is used.  When the same mapper instance is shared
    by Vertex-1/2/3, the expensive PN spline is therefore constructed once per
    likelihood point, not once per interferometer.
    """

    sparse_points: int = 2**10
    _cache_key: Optional[Tuple[Any, ...]] = field(default=None, init=False, repr=False)
    _cache_spline: Optional[CubicSpline] = field(default=None, init=False, repr=False)

    def _make_key(
        self,
        parameters: Dict[str, Any],
        minimum_frequency: float,
        maximum_frequency: float,
        delta_f: float,
    ) -> Tuple[Any, ...]:
        intrinsic = _extract_intrinsic_parameters(parameters)
        return intrinsic + (
            float(minimum_frequency),
            float(maximum_frequency),
            float(delta_f),
            int(self.sparse_points),
        )

    def _build_spline(
        self,
        parameters: Dict[str, Any],
        minimum_frequency: float,
        maximum_frequency: float,
        delta_f: float,
    ) -> CubicSpline:
        mass1, mass2, chi1, chi2, lambda1, lambda2 = _extract_intrinsic_parameters(parameters)

        f_sparse_min = float(minimum_frequency) - float(delta_f)
        if f_sparse_min <= 0:
            f_sparse_min = float(minimum_frequency)

        freqs = np.logspace(
            np.log10(f_sparse_min),
            np.log10(float(maximum_frequency)),
            int(self.sparse_points),
        )

        dphase_df_sparse = Phase_deriv_wrt_f(
            freqs,
            mass1,
            mass2,
            chi1,
            chi2,
            lambda1,
            lambda2,
        )
        time_before_merger = -(0.5 / np.pi) * dphase_df_sparse
        return CubicSpline(freqs, time_before_merger)

    def __call__(
        self,
        frequencies,
        parameters: Dict[str, Any],
        *,
        minimum_frequency: float,
        maximum_frequency: float,
        delta_f: float,
    ):
        frequencies = np.asarray(frequencies, dtype=float)
        if np.any(frequencies <= 0):
            raise ValueError("Frequency-to-time mapping requires strictly positive frequencies.")

        key = self._make_key(
            parameters,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            delta_f=delta_f,
        )
        if key != self._cache_key or self._cache_spline is None:
            self._cache_spline = self._build_spline(
                parameters,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
                delta_f=delta_f,
            )
            self._cache_key = key

        return np.asarray(self._cache_spline(frequencies), dtype=float)


__all__ = [
    "TaylorF2FrequencyTimeMapper",
    "Phase_deriv_wrt_f",
    "TaylorF2Phasing_6PNQM2SCoeff",
    "TaylorF2Phasing_6PNSelf2SCoeff",
    "TaylorF2Phasing_6PNSOCoeff",
    "TaylorF2Phasing_6PNS1S2OCoeff",
]
