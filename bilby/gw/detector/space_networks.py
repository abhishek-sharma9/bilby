"""Convenience constructors for networks of space-based Bilby interferometers.

No new InterferometerList subclass is required: SpaceInterferometer inherits
from Bilby's ordinary Interferometer and is therefore accepted by the existing
InterferometerList type checks and injection loops.
"""

from __future__ import annotations

from .b_decigo import make_bdecigo_vertices
from .networks import InterferometerList


def make_bdecigo_network(**kwargs):
    """Return ``(ifos, vertices)`` for the three B-DECIGO readouts.

    Examples
    --------
    >>> ifos, vertices = make_bdecigo_network(
    ...     asd_file="asd_B-Decigo.txt",
    ...     minimum_frequency=0.05,
    ...     maximum_frequency=20.0,
    ...     response_mode="dynamic_lwl",
    ... )
    >>> vertex1, vertex2, vertex3 = vertices
    >>> ifos.inject_signal(waveform_generator=waveform_generator,
    ...                    parameters=injection_parameters)
    """

    vertices = make_bdecigo_vertices(**kwargs)
    ifos = InterferometerList([vertex.ifo for vertex in vertices])
    return ifos, vertices


__all__ = ["make_bdecigo_network"]
