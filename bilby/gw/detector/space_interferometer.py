"""Bilby-compatible base class for moving space-based interferometers.

The important design choice is that this class subclasses Bilby's ordinary
:class:`Interferometer`..

Parameter-name convention
-------------------------
To avoid changes elsewhere in Bilby we deliberately retain the standard keys
``ra``, ``dec`` and ``geocent_time``.  For the heliocentric space-detector
classes in this extension they are *labels only* and mean

    ra           -> ecliptic longitude [rad]
    dec          -> ecliptic latitude [rad]
    geocent_time -> SSB reference/coalescence time [s on the GPS numeric axis]

Accordingly, ``time_delay_from_geocenter`` keeps Bilby's method name for API
compatibility but subclasses may interpret it as an SSB-to-detector delay.
"""

from __future__ import annotations

import numpy as np

from .calibration import Recalibrate
from .interferometer import Interferometer

class SpaceInterferometer(Interferometer):
    """Abstract Bilby base class for moving space-detector responses.

    Notes
    -----
    This class is *not* a concrete detector.  Subclasses such as
    ``BDECIGOInterferometer`` must implement ``_frequency_dependent_response``.
    Bilby calls :meth:`get_detector_response`, which then dispatches to that
    subclass implementation.  Therefore the detector-frame waveform can have
    genuinely frequency-dependent antenna factors and Doppler delay.
    """

    RESPONSE_MODES = ("static_lwl", "dynamic_lwl", "dynamic_full")

    def __init__(
        self,
        name,
        power_spectral_density,
        minimum_frequency,
        maximum_frequency,
        *,
        arm_length_km,
        constellation,
        frequency_time_mapper=None,
        response_mode="dynamic_lwl",
        interpolation_points=2**12,
        calibration_model=None,
    ):
        if calibration_model is None:
            calibration_model = Recalibrate()

        if response_mode not in self.RESPONSE_MODES:
            raise ValueError(
                f"response_mode must be one of {self.RESPONSE_MODES}; got {response_mode!r}."
            )

        # Interferometer requires an Earth-fixed geometry object.  For a space
        # detector those fields are not physically used: antenna_response(),
        # time_delay_from_geocenter(), and get_detector_response() are all
        # overridden.  We nevertheless initialize a benign dummy geometry so
        # the rest of Bilby's Interferometer machinery remains available.
        super().__init__(
            name=name,
            power_spectral_density=power_spectral_density,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            length=float(arm_length_km),
            latitude=0.0,
            longitude=0.0,
            elevation=0.0,
            xarm_azimuth=0.0,
            yarm_azimuth=60.0,
            xarm_tilt=0.0,
            yarm_tilt=0.0,
            calibration_model=calibration_model,
        )

        self.constellation = constellation
        self.frequency_time_mapper = frequency_time_mapper
        self.response_mode = response_mode
        self.interpolation_points = int(interpolation_points)

        self.meta_data.update(
            {
                "space_based": True,
                "coordinate_frame": "SSB/ecliptic",
                "ra_semantics": "ecliptic_longitude",
                "dec_semantics": "ecliptic_latitude",
                "geocent_time_semantics": "ssb_reference_time",
                "response_mode": response_mode,
            }
        )

     
    # Methods subclasses provide
     
    def _static_antenna_response(self, ra, dec, time, psi):
        """Return ``(Fplus, Fcross)`` for one frozen detector time."""
        raise NotImplementedError

    def _static_reference_delay(self, ra, dec, time):
        """Return the SSB-to-detector reference-position delay at one time."""
        raise NotImplementedError

    def _frequency_dependent_response(self, frequencies, parameters):
        """Return ``(Fplus(f), Fcross(f), delay(f))``.

        Arrays must have the same length as ``frequencies``.  This method is
        responsible for implementing the selected response mode.
        """
        raise NotImplementedError

     
    # Bilby-compatible scalar/static interfaces
     
    def antenna_response(self, ra, dec, time, psi, mode):
        """Bilby API-compatible static antenna response.

        For this class ``ra``/``dec`` are interpreted according to the
        subclass's space-frame convention, not as ICRS right ascension and
        declination.
        """

        fplus, fcross = self._static_antenna_response(ra, dec, time, psi)
        if mode == "plus":
            return float(np.asarray(fplus).reshape(-1)[0])
        if mode == "cross":
            return float(np.asarray(fcross).reshape(-1)[0])
        if mode == self.name:
            return 1.0
        return 0.0

    def time_delay_from_geocenter(self, ra, dec, time):
        """Retain Bilby's method name while using the space-detector delay.

        In the B-DECIGO implementation this is the SSB-origin to constellation
        centroid delay.  ``geocent_time`` is intentionally treated as an SSB
        time elsewhere, so no Earth/geocenter geometry enters this calculation.
        """

        return float(self._static_reference_delay(ra, dec, time))

     
    # Core Bilby projection hook
     

    def get_detector_response(self, waveform_polarizations, parameters, frequencies=None):
        """Project source polarizations into a moving space interferometer.

        This is the only Bilby projection method that needs to be changed for
        the time/frequency-dependent response.  The method preserves Bilby's
        behavior in two important ways:

        * ``frequencies=None`` returns an array on the full interferometer
          frequency grid, with values only in ``frequency_mask``.
        * an explicit ``frequencies`` array (e.g. relative-binning edges)
          returns a response evaluated directly at those frequencies.

        Bilby's calibration model is applied exactly as in the ordinary
        :class:`Interferometer` implementation.
        """

        if frequencies is None:
            if self.duration is None:
                raise RuntimeError(
                    "Set interferometer strain data before evaluating a dynamic response."
                )
            full_grid = True
            mask = np.asarray(self.frequency_mask, dtype=bool)
            eval_frequencies = np.asarray(self.frequency_array[mask], dtype=float)

            first_mode = next(iter(waveform_polarizations.values()))
            signal_ifo = np.zeros_like(np.asarray(first_mode), dtype=complex)
        else:
            full_grid = False
            eval_frequencies = np.asarray(frequencies, dtype=float)
            mask = np.ones(len(eval_frequencies), dtype=bool)
            signal_ifo = np.zeros(len(eval_frequencies), dtype=complex)

        if len(eval_frequencies) == 0:
            return signal_ifo
        if np.any(eval_frequencies <= 0):
            raise ValueError(
                "Space-detector response was asked to evaluate non-positive frequencies. "
                "Set a positive detector minimum_frequency."
            )

        fplus, fcross, reference_delay = self._frequency_dependent_response(
            eval_frequencies, parameters
        )
        fplus = np.asarray(fplus)
        fcross = np.asarray(fcross)
        reference_delay = np.asarray(reference_delay, dtype=float)

        if fplus.shape != eval_frequencies.shape:
            fplus = np.broadcast_to(fplus, eval_frequencies.shape)
        if fcross.shape != eval_frequencies.shape:
            fcross = np.broadcast_to(fcross, eval_frequencies.shape)
        if reference_delay.shape != eval_frequencies.shape:
            reference_delay = np.broadcast_to(reference_delay, eval_frequencies.shape)

        projected = np.zeros(len(eval_frequencies), dtype=complex)

        for mode, waveform in waveform_polarizations.items():
            if mode == "plus":
                det_response = fplus
            elif mode == "cross":
                det_response = fcross
            elif mode == self.name:
                det_response = np.ones_like(eval_frequencies)
            else:
                # The supplied B-DECIGO response defines tensor +/x only.
                # Preserve Bilby's usual convention of zero response for an
                # unrecognized polarization rather than silently inventing a
                # vector/scalar response.
                det_response = np.zeros_like(eval_frequencies)

            waveform = np.asarray(waveform)
            waveform_eval = waveform[mask] if full_grid else waveform
            if len(waveform_eval) != len(eval_frequencies):
                raise ValueError(
                    f"Waveform polarization {mode!r} has length {len(waveform_eval)}, "
                    f"but the detector response is evaluated at {len(eval_frequencies)} frequencies."
                )
            projected += waveform_eval * det_response

        # Bilby treats the sampled time as a reference time and shifts the
        # source waveform relative to the start of the data segment.  We keep
        # exactly the same operation, but 'geocent_time' is interpreted here as
        # an SSB time and reference_delay is SSB -> B-DECIGO centroid.
        dt_ssb = float(parameters["geocent_time"]) - float(self.strain_data.start_time)
        dt = dt_ssb + reference_delay
        projected *= np.exp(-1j * 2.0 * np.pi * eval_frequencies * dt)

        projected *= self.calibration_model.get_calibration_factor(
            eval_frequencies,
            prefix="recalib_{}_".format(self.name),
            **parameters,
        )

        if full_grid:
            signal_ifo[mask] = projected
        else:
            signal_ifo[:] = projected

        return signal_ifo


__all__ = ["SpaceInterferometer"]
