import logging
import lalpulsar
from solar_system_ephemerides import body_ephemeris_path

logger = logging.getLogger(__name__)

DEFAULT_EPHEMERIS = lalpulsar.InitBarycenter(
    str(body_ephemeris_path("earth")),
    str(body_ephemeris_path("sun")),
)
