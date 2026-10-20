"""Continuación reproducible de Spotify Tracks para la evaluación 2."""
import os

# Evitar consulta WMIC obsoleta de joblib en Windows y limitar recursos locales.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(4, os.cpu_count() or 1)))
