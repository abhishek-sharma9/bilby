"""Bilby B-DECIGO interferometers with static or time/frequency-dependent response.

Each triangle vertex is represented
as an independent Michelson-like :class:`BDECIGOInterferometer`.

IMPORTANT coordinate convention
-------------------------------
To preserve Bilby's public likelihood/injection interfaces without renaming
parameters globally, the ordinary Bilby keys are deliberately reinterpreted:

    ra           = ecliptic longitude (lambda)
    dec          = ecliptic latitude  (beta)
    geocent_time = SSB reference/coalescence time

Use Bilby's default ``reference_frame='sky'`` and do not request Earth-based
sky-frame transformations for these detectors.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

from .frequency_time import TaylorF2FrequencyTimeMapper
from .psd import PowerSpectralDensity
from .space_constellation import BDECIGOConstellation, C_SI
from .space_interferometer import SpaceInterferometer


class BDECIGOInterferometer(SpaceInterferometer):
    """One of the three independent B-DECIGO vertex readouts."""

    VALID_VERTICES = BDECIGOConstellation.VALID_VERTICES

    def __init__(
        self,
        name,
        *,
        vertex=None,
        power_spectral_density=None,
        asd_file=None,
        minimum_frequency=0.1,
        maximum_frequency=100.0,
        constellation=None,
        frequency_time_mapper=None,
        orbit_tstart=0.0,
        response_mode="dynamic_lwl",
        interpolation_points=2**12,
        calibration_model=None,
    ):
        if vertex is None:
            vertex = name
        if vertex not in self.VALID_VERTICES:
            raise ValueError(f"vertex must be one of {self.VALID_VERTICES}; got {vertex!r}.")

        if power_spectral_density is not None and asd_file is not None:
            raise ValueError("Provide either power_spectral_density or asd_file, not both.")
        if power_spectral_density is None:
            if asd_file is None:
                raise ValueError(
                    "BDECIGOInterferometer requires power_spectral_density or asd_file."
                )
            power_spectral_density = PowerSpectralDensity(asd_file=asd_file)

        if constellation is None:
            constellation = BDECIGOConstellation(orbit_tstart=orbit_tstart)
        if frequency_time_mapper is None:
            frequency_time_mapper = TaylorF2FrequencyTimeMapper()

        self.vertex_name = vertex

        super().__init__(
            name=name,
            power_spectral_density=power_spectral_density,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            arm_length_km=constellation.arm_length_m / 1000.0,
            constellation=constellation,
            frequency_time_mapper=frequency_time_mapper,
            response_mode=response_mode,
            interpolation_points=interpolation_points,
            calibration_model=calibration_model,
        )

        self.meta_data.update(
            {
                "detector": "B-DECIGO",
                "configuration": "heliocentric_T",
                "vertex_readout": self.vertex_name,
            }
        )

    # Polarization and source direction in the SSB/ecliptic frame

    @staticmethod
    def reference_pol_tensors(longitude, latitude):
        u = np.array([np.sin(longitude), -np.cos(longitude), 0.0])
        v = np.array(
            [
                -np.sin(latitude) * np.cos(longitude),
                -np.sin(latitude) * np.sin(longitude),
                np.cos(latitude),
            ]
        )
        e_plus = u[:, None] @ u[None, :] - v[:, None] @ v[None, :]
        e_cross = u[:, None] @ v[None, :] + v[:, None] @ u[None, :]
        return e_plus, e_cross

    @staticmethod
    def source_khat(longitude, latitude):
        return np.array(
            [
                -np.cos(longitude) * np.cos(latitude),
                -np.sin(longitude) * np.cos(latitude),
                -np.sin(latitude),
            ]
        )

    @classmethod
    def polarization_tensors(cls, longitude, latitude, psi):
        e_plus, e_cross = cls.reference_pol_tensors(longitude, latitude)
        E_plus = np.cos(2.0 * psi) * e_plus + np.sin(2.0 * psi) * e_cross
        E_cross = -np.sin(2.0 * psi) * e_plus + np.cos(2.0 * psi) * e_cross
        return E_plus, E_cross

    # Static Bilby compatibility methods

    def _static_antenna_response(self, ra, dec, time, psi):
        positions = self.constellation.spacecraft_positions(np.atleast_1d(time))
        _, (link1, link2) = self.constellation.vertex_links(positions, self.vertex_name)

        d1 = link1[:, :, None] @ link1[:, None, :]
        d2 = link2[:, :, None] @ link2[:, None, :]
        E_plus, E_cross = self.polarization_tensors(ra, dec, psi)

        # Standard Michelson detector tensor convention: d = 1/2 (u x u - v x v).
        Fplus = 0.5 * np.trace((d1 - d2) @ E_plus[None, :], axis1=1, axis2=2)
        Fcross = 0.5 * np.trace((d1 - d2) @ E_cross[None, :], axis1=1, axis2=2)
        return Fplus, Fcross

    def _static_reference_delay(self, ra, dec, time):
        center = self.constellation.reference_position(np.atleast_1d(time))[0]
        khat = self.source_khat(ra, dec)
        return np.dot(khat, center) / C_SI

    # Direct dynamic response at requested frequencies

    def _relative_times(self, frequencies, parameters):
        if self.frequency_time_mapper is None:
            raise RuntimeError(
                "A frequency_time_mapper is required for a dynamic space-detector response."
            )
        if self.duration is None:
            raise RuntimeError("Set strain data before evaluating the dynamic response.")

        return self.frequency_time_mapper(
            frequencies,
            parameters,
            minimum_frequency=float(self.minimum_frequency),
            maximum_frequency=float(self.maximum_frequency),
            delta_f=1.0 / float(self.duration),
        )

    def _response_at_frequencies_direct(self, frequencies, parameters, use_lwl=True):
        frequencies = np.asarray(frequencies, dtype=float)

        longitude = float(parameters["ra"])
        latitude = float(parameters["dec"])
        psi = float(parameters["psi"])
        ssb_time = float(parameters["geocent_time"])

        relative_times = self._relative_times(frequencies, parameters)
        detector_times = ssb_time + relative_times

        positions = self.constellation.spacecraft_positions(detector_times)
        center = self.constellation.centroid(positions)
        vertex_position, (link1, link2) = self.constellation.vertex_links(
            positions, self.vertex_name
        )

        E_plus, E_cross = self.polarization_tensors(longitude, latitude, psi)
        d1 = link1[:, :, None] @ link1[:, None, :]
        d2 = link2[:, :, None] @ link2[:, None, :]

        if use_lwl:
            Fplus = 0.5 * np.trace((d1 - d2) @ E_plus[None, :], axis1=1, axis2=2)
            Fcross = 0.5 * np.trace((d1 - d2) @ E_cross[None, :], axis1=1, axis2=2)
        else:
            dplus1 = 0.5 * np.trace(d1 @ E_plus[None, :], axis1=1, axis2=2)
            dcross1 = 0.5 * np.trace(d1 @ E_cross[None, :], axis1=1, axis2=2)
            dplus2 = 0.5 * np.trace(d2 @ E_plus[None, :], axis1=1, axis2=2)
            dcross2 = 0.5 * np.trace(d2 @ E_cross[None, :], axis1=1, axis2=2)

            T1, T2 = self._transfer_functions(
                frequencies,
                center,
                vertex_position,
                link1,
                link2,
                longitude,
                latitude,
            )
            Fplus = dplus1 * T1 - dplus2 * T2
            Fcross = dcross1 * T1 - dcross2 * T2

        khat = self.source_khat(longitude, latitude)
        delay = (khat[None, :] @ center.T).flatten() / C_SI
        return Fplus, Fcross, delay

    def _transfer_functions(
        self,
        frequencies,
        center,
        vertex_position,
        link1,
        link2,
        longitude,
        latitude,
    ):
        khat = self.source_khat(longitude, latitude)

        mu1 = (khat[None, :] @ link1.T).flatten()
        mu2 = (khat[None, :] @ link2.T).flatten()

        L = float(self.constellation.arm_length_m)
        fstar = C_SI / (2.0 * np.pi * L)
        f_by_fstar = np.asarray(frequencies, dtype=float) / fstar

        to_center = center - vertex_position
        center_distance = np.linalg.norm(to_center, axis=1)
        r_i0 = to_center / center_distance[:, None]
        mu_center = (khat[None, :] @ r_i0.T).flatten()

        centroid_phase_term = (2.0 / np.sqrt(3.0)) * mu_center

        T1 = 0.5 * (
            np.sinc(0.5 * f_by_fstar * (1.0 - mu1) / np.pi)
            * np.exp(-1j * 0.5 * f_by_fstar * (3.0 + mu1 - centroid_phase_term))
            + np.sinc(0.5 * f_by_fstar * (1.0 + mu1) / np.pi)
            * np.exp(-1j * 0.5 * f_by_fstar * (1.0 + mu1 - centroid_phase_term))
        )

        T2 = 0.5 * (
            np.sinc(0.5 * f_by_fstar * (1.0 - mu2) / np.pi)
            * np.exp(-1j * 0.5 * f_by_fstar * (3.0 + mu2 - centroid_phase_term))
            + np.sinc(0.5 * f_by_fstar * (1.0 + mu2) / np.pi)
            * np.exp(-1j * 0.5 * f_by_fstar * (1.0 + mu2 - centroid_phase_term))
        )
        return T1, T2

    def _dynamic_response(self, frequencies, parameters, use_lwl):
        frequencies = np.asarray(frequencies, dtype=float)

        # Relative binning usually supplies far fewer than 4096 frequencies.
        # Evaluate those frequencies directly.  On the full FFT grid, preserve
        # the sparse-log-grid + cubic-spline strategy.
        if len(frequencies) <= self.interpolation_points:
            return self._response_at_frequencies_direct(frequencies, parameters, use_lwl=use_lwl)

        sparse_freqs = np.logspace(
            np.log10(frequencies[0]),
            np.log10(frequencies[-1]),
            self.interpolation_points,
        )
        Fp_sparse, Fc_sparse, delay_sparse = self._response_at_frequencies_direct(
            sparse_freqs, parameters, use_lwl=use_lwl
        )

        Fplus = CubicSpline(sparse_freqs, Fp_sparse)(frequencies)
        Fcross = CubicSpline(sparse_freqs, Fc_sparse)(frequencies)
        delay = CubicSpline(sparse_freqs, delay_sparse)(frequencies)
        return Fplus, Fcross, delay

    def frequency_dependent_antenna_response(self, frequencies, parameters, return_delay=False):
        """Evaluate the B-DECIGO response at requested Fourier frequencies.

        For ``dynamic_lwl`` and ``dynamic_full`` this performs the complete
        mapping

            f -> t(f) -> spacecraft positions -> arm directions
              -> Fplus(f), Fcross(f), Doppler delay(f).

        Parameters
        ----------
        frequencies : array_like
            Positive Fourier frequencies in Hz.
        parameters : dict
            Bilby parameter dictionary.  In this extension ``ra`` and ``dec``
            denote ecliptic longitude/latitude and ``geocent_time`` denotes
            the SSB reference time.
        return_delay : bool, optional
            If True, also return the SSB-to-centroid delay array.
        """
        Fplus, Fcross, delay = self._frequency_dependent_response(
            np.asarray(frequencies, dtype=float), parameters
        )
        if return_delay:
            return Fplus, Fcross, delay
        return Fplus, Fcross

    def frequency_to_detector_time(self, frequencies, parameters):
        """Return the absolute SSB times used to evaluate the moving detector."""
        frequencies = np.asarray(frequencies, dtype=float)
        if self.response_mode == "static_lwl":
            return np.full(len(frequencies), float(parameters["geocent_time"]), dtype=float)
        return float(parameters["geocent_time"]) + self._relative_times(frequencies, parameters)

    def _frequency_dependent_response(self, frequencies, parameters):
        frequencies = np.asarray(frequencies, dtype=float)

        if self.response_mode == "static_lwl":
            # Bilby's reference_time feature remains useful for a frozen
            # response.  Dynamic response intentionally ignores reference_time.
            time = (
                float(self.reference_time)
                if self.reference_time is not None
                else float(parameters["geocent_time"])
            )
            Fp, Fc = self._static_antenna_response(
                parameters["ra"], parameters["dec"], time, parameters["psi"]
            )
            delay = self._static_reference_delay(parameters["ra"], parameters["dec"], time)
            return (
                np.full(len(frequencies), np.asarray(Fp).reshape(-1)[0]),
                np.full(len(frequencies), np.asarray(Fc).reshape(-1)[0]),
                np.full(len(frequencies), delay, dtype=float),
            )

        if self.response_mode == "dynamic_lwl":
            return self._dynamic_response(frequencies, parameters, use_lwl=True)

        if self.response_mode == "dynamic_full":
            return self._dynamic_response(frequencies, parameters, use_lwl=False)

        raise RuntimeError(f"Unhandled response_mode {self.response_mode!r}.")


class BDECIGOVertex:
    """Small compatibility wrapper exposing the Bilby detector as ``.ifo``.

    This preserves the usage pattern of the user's current static utilities:

    >>> vertex1 = BDECIGOVertex("Vertex-1", asd_file="asd_B-Decigo.txt")
    >>> ifos = InterferometerList([vertex1.ifo, ...])
    """

    def __init__(self, name, **kwargs):
        self.ifo = BDECIGOInterferometer(name=name, vertex=name, **kwargs)


def make_bdecigo_vertices(
    *,
    asd_file=None,
    power_spectral_density=None,
    minimum_frequency=0.1,
    maximum_frequency=100.0,
    orbit_tstart=0.0,
    response_mode="dynamic_lwl",
    interpolation_points=2**12,
    frequency_time_mapper=None,
    constellation=None,
):
    """Create Vertex-1/2/3 wrappers sharing one constellation and t(f) mapper."""

    if constellation is None:
        constellation = BDECIGOConstellation(orbit_tstart=orbit_tstart)
    if frequency_time_mapper is None:
        frequency_time_mapper = TaylorF2FrequencyTimeMapper()

    common = dict(
        asd_file=asd_file,
        power_spectral_density=power_spectral_density,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        constellation=constellation,
        frequency_time_mapper=frequency_time_mapper,
        response_mode=response_mode,
        interpolation_points=interpolation_points,
    )

    return tuple(BDECIGOVertex(name, **common) for name in BDECIGOConstellation.VALID_VERTICES)


__all__ = [
    "BDECIGOInterferometer",
    "BDECIGOVertex",
    "make_bdecigo_vertices",
]
