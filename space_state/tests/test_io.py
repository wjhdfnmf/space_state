"""
Tests for Save/Load Functionality
"""

import numpy as np
import pytest
import tempfile
import json
from pathlib import Path
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from base.star_io import (
    save_state, load_state,
    save_propagation_result, load_propagation_result,
    export_trajectory_csv, import_trajectory_csv,
    NumpyEncoder, numpy_decoder,
)


class TestNumpyEncoder:
    """Test JSON encoding/decoding for numpy arrays."""

    def test_encode_1d_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        encoded = json.dumps(arr, cls=NumpyEncoder)
        decoded = json.loads(encoded, object_hook=numpy_decoder)
        assert np.allclose(decoded, arr)

    def test_encode_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        encoded = json.dumps(arr, cls=NumpyEncoder)
        decoded = json.loads(encoded, object_hook=numpy_decoder)
        assert np.allclose(decoded, arr)

    def test_encode_float64(self):
        val = np.float64(3.14159)
        encoded = json.dumps(val, cls=NumpyEncoder)
        decoded = json.loads(encoded)
        assert np.isclose(decoded, val)

    def test_encode_int64(self):
        val = np.int64(42)
        encoded = json.dumps(val, cls=NumpyEncoder)
        decoded = json.loads(encoded)
        assert decoded == val


class TestSaveLoadState:
    """Test single state save/load."""

    def test_save_load_roundtrip(self):
        """State survives save/load roundtrip."""
        state = np.array([
            1.0, 0.0, 0.0,       # r
            0.0, 0.01, 0.0,      # v
            50000.0,             # mass
            1.0, 0.0, 0.0, 0.0,  # quaternion
            0.0, 0.0, 0.0        # omega
        ])

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_state(filepath, state, simulation_time=10.5)
            loaded = load_state(filepath)

            assert np.allclose(loaded['state'], state)
            assert np.isclose(loaded['simulation_time'], 10.5)
        finally:
            Path(filepath).unlink()

    def test_save_with_metadata(self):
        """Metadata is preserved."""
        state = np.zeros(14)
        metadata = {"mission": "Test", "version": 1}

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_state(filepath, state, metadata=metadata)
            loaded = load_state(filepath)

            assert loaded['metadata']['mission'] == "Test"
            assert loaded['metadata']['version'] == 1
        finally:
            Path(filepath).unlink()

    def test_save_with_spacecraft_config(self):
        """Spacecraft config is preserved."""
        state = np.zeros(14)
        config = {"name": "Starship", "dry_mass": 100000}

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_state(filepath, state, spacecraft_config=config)
            loaded = load_state(filepath)

            assert loaded['spacecraft']['name'] == "Starship"
            assert loaded['spacecraft']['dry_mass'] == 100000
        finally:
            Path(filepath).unlink()


class TestSaveLoadPropagation:
    """Test propagation result save/load."""

    def test_propagation_roundtrip(self):
        """Propagation result survives roundtrip."""
        times = np.linspace(0, 10, 100)
        states = np.random.rand(100, 14)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_propagation_result(filepath, times, states)
            loaded = load_propagation_result(filepath)

            assert np.allclose(loaded['times'], times)
            assert np.allclose(loaded['states'], states)
        finally:
            Path(filepath).unlink()

    def test_propagation_with_guidance_log(self):
        """Guidance log is preserved."""
        times = np.array([0, 1, 2])
        states = np.zeros((3, 14))
        guidance_log = [
            {"time": 0.5, "burn": "TLI", "dv": 3500},
            {"time": 3.5, "burn": "LOI", "dv": 900}
        ]

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_propagation_result(filepath, times, states, guidance_log=guidance_log)
            loaded = load_propagation_result(filepath)

            assert len(loaded['guidance_log']) == 2
            assert loaded['guidance_log'][0]['burn'] == "TLI"
        finally:
            Path(filepath).unlink()

    def test_handles_transposed_states(self):
        """Handles (14, N) state format."""
        times = np.array([0, 1, 2])
        states_transposed = np.random.rand(14, 3)  # (14, N) format

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_propagation_result(filepath, times, states_transposed)
            loaded = load_propagation_result(filepath)

            # Should be stored as (N, 14)
            assert loaded['states'].shape == (3, 14)
            assert np.allclose(loaded['states'], states_transposed.T)
        finally:
            Path(filepath).unlink()


class TestCSVExport:
    """Test CSV export/import."""

    def test_csv_roundtrip(self):
        """Trajectory survives CSV roundtrip."""
        times = np.array([0.0, 1.0, 2.0])
        states = np.array([
            [1.0, 0, 0, 0, 0.01, 0, 50000, 1, 0, 0, 0, 0, 0, 0],
            [1.1, 0, 0, 0, 0.01, 0, 49900, 1, 0, 0, 0, 0, 0, 0],
            [1.2, 0, 0, 0, 0.01, 0, 49800, 1, 0, 0, 0, 0, 0, 0],
        ])

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            export_trajectory_csv(filepath, times, states)
            loaded = import_trajectory_csv(filepath)

            assert np.allclose(loaded['times'], times, rtol=1e-10)
            assert np.allclose(loaded['states'], states, rtol=1e-10)
        finally:
            Path(filepath).unlink()

    def test_csv_has_header(self):
        """CSV includes column headers."""
        times = np.array([0.0])
        states = np.zeros((1, 14))

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            filepath = f.name

        try:
            export_trajectory_csv(filepath, times, states, header=True)

            with open(filepath) as f:
                first_line = f.readline()

            assert "time_days" in first_line
            assert "r_x_au" in first_line
        finally:
            Path(filepath).unlink()


class TestFileFormats:
    """Test file format handling."""

    def test_json_is_readable(self):
        """Saved JSON is human-readable."""
        state = np.zeros(14)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_state(filepath, state)

            with open(filepath) as f:
                content = f.read()

            # Should be indented (human-readable)
            assert '\n  ' in content
            # Should contain expected keys
            assert 'format_version' in content
            assert 'saved_at' in content
        finally:
            Path(filepath).unlink()

    def test_format_version_included(self):
        """Format version is saved for forward compatibility."""
        state = np.zeros(14)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_state(filepath, state)
            loaded = load_state(filepath)

            assert 'format_version' in loaded
            assert loaded['format_version'] == "1.0"
        finally:
            Path(filepath).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
