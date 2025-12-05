# base/star_io.py
"""
Mission Save/Load Functionality

Provides serialization and deserialization of simulation state for:
- Checkpointing long simulations
- Resuming interrupted runs
- Sharing mission configurations
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import asdict, is_dataclass

from .star_logger import logger
from .star_constants import epoch_jd


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return {"__dataclass__": obj.__class__.__name__, "data": asdict(obj)}
        return super().default(obj)


def numpy_decoder(dct: Dict) -> Any:
    """Custom JSON decoder for NumPy arrays."""
    if "__ndarray__" in dct:
        return np.array(dct["data"], dtype=dct["dtype"])
    return dct


def save_state(
    filepath: Union[str, Path],
    state: np.ndarray,
    spacecraft_config: Optional[Dict] = None,
    simulation_time: float = 0.0,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save a simulation state to a JSON file.

    Args:
        filepath: Path to save the state file
        state: The 14-element state vector [r, v, m, q, ω]
        spacecraft_config: Optional spacecraft configuration dict
        simulation_time: Current simulation time (days from epoch)
        metadata: Optional metadata dict (mission name, notes, etc.)
    """
    filepath = Path(filepath)

    data = {
        "format_version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "epoch_jd": epoch_jd,
        "simulation_time": simulation_time,
        "state": state,
        "spacecraft": spacecraft_config or {},
        "metadata": metadata or {},
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)

    logger.info(f"Saved state to {filepath}")


def load_state(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a simulation state from a JSON file.

    Args:
        filepath: Path to the state file

    Returns:
        Dict containing:
            - state: The 14-element state vector
            - simulation_time: Time in days from epoch
            - spacecraft: Spacecraft configuration
            - metadata: Any metadata
            - epoch_jd: Original epoch Julian date
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        data = json.load(f, object_hook=numpy_decoder)

    # Ensure state is numpy array
    if isinstance(data.get("state"), list):
        data["state"] = np.array(data["state"])

    logger.info(f"Loaded state from {filepath}")
    logger.info(f"  Simulation time: {data['simulation_time']:.4f} days")

    return data


def save_propagation_result(
    filepath: Union[str, Path],
    times: np.ndarray,
    states: np.ndarray,
    spacecraft_config: Optional[Dict] = None,
    guidance_log: Optional[list] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save a complete propagation result to JSON.

    Args:
        filepath: Path to save the result file
        times: Array of time values (days from epoch)
        states: Array of states, shape (N, 14) or (14, N)
        spacecraft_config: Optional spacecraft configuration
        guidance_log: Optional list of guidance cycle dicts
        metadata: Optional metadata
    """
    filepath = Path(filepath)

    # Ensure states are in (N, 14) format for storage
    if states.shape[0] == 14 and states.ndim == 2:
        states = states.T

    data = {
        "format_version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "epoch_jd": epoch_jd,
        "times": times,
        "states": states,
        "spacecraft": spacecraft_config or {},
        "guidance_log": guidance_log or [],
        "metadata": metadata or {},
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)

    logger.info(f"Saved propagation result to {filepath}")
    logger.info(f"  Time span: {times[0]:.4f} to {times[-1]:.4f} days")
    logger.info(f"  States: {len(times)} points")


def load_propagation_result(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a propagation result from JSON.

    Args:
        filepath: Path to the result file

    Returns:
        Dict containing times, states, spacecraft config, guidance log, metadata
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        data = json.load(f, object_hook=numpy_decoder)

    # Convert lists to numpy arrays
    if isinstance(data.get("times"), list):
        data["times"] = np.array(data["times"])
    if isinstance(data.get("states"), list):
        data["states"] = np.array(data["states"])

    logger.info(f"Loaded propagation result from {filepath}")
    logger.info(f"  Time span: {data['times'][0]:.4f} to {data['times'][-1]:.4f} days")
    logger.info(f"  States: {len(data['times'])} points")

    return data


def save_simulation(
    filepath: Union[str, Path],
    simulation: "Simulation",
    save_history: bool = True,
) -> None:
    """
    Save a complete multi-spacecraft Simulation to JSON.

    Args:
        filepath: Path to save the simulation file
        simulation: Simulation object to save
        save_history: Whether to include full propagation history
    """
    filepath = Path(filepath)

    # Build entity data
    entities_data = {}
    for entity_id, entity in simulation.entities.items():
        entity_data = {
            "id": entity.id,
            "name": entity.name,
            "current_state": entity.current_state,
            "active": entity.active,
        }

        # Save spacecraft config if available
        if entity.spacecraft is not None:
            entity_data["spacecraft"] = {
                "name": entity.spacecraft.name,
                "dry_mass": entity.spacecraft.dry_mass,
                "propellant_mass": entity.spacecraft._propellant_mass,
            }

        # Save propagation history if requested
        if save_history and entity.result is not None:
            entity_data["history"] = {
                "times": entity.result.t,
                "states": entity.result.y.T,  # Transpose for storage
            }

        entities_data[entity_id] = entity_data

    data = {
        "format_version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "epoch_jd": epoch_jd,
        "simulation": {
            "name": simulation.name,
            "t_current": simulation._t_current,
        },
        "entities": entities_data,
        "events": simulation._events,
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)

    logger.info(f"Saved simulation '{simulation.name}' to {filepath}")
    logger.info(f"  Entities: {len(simulation.entities)}")
    logger.info(f"  Current time: {simulation._t_current:.4f} days")


def load_simulation_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load simulation data from JSON.

    Note: This returns raw data. Use Simulation.from_file() to
    reconstruct a full Simulation object with propagators.

    Args:
        filepath: Path to the simulation file

    Returns:
        Dict containing simulation configuration and entity states
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        data = json.load(f, object_hook=numpy_decoder)

    # Convert nested arrays
    for entity_id, entity_data in data.get("entities", {}).items():
        if "current_state" in entity_data and isinstance(entity_data["current_state"], list):
            entity_data["current_state"] = np.array(entity_data["current_state"])
        if "history" in entity_data:
            if isinstance(entity_data["history"]["times"], list):
                entity_data["history"]["times"] = np.array(entity_data["history"]["times"])
            if isinstance(entity_data["history"]["states"], list):
                entity_data["history"]["states"] = np.array(entity_data["history"]["states"])

    logger.info(f"Loaded simulation data from {filepath}")

    return data


def export_trajectory_csv(
    filepath: Union[str, Path],
    times: np.ndarray,
    states: np.ndarray,
    header: bool = True,
) -> None:
    """
    Export trajectory to CSV for external tools.

    Args:
        filepath: Path to CSV file
        times: Array of times (days)
        states: Array of states (N, 14)
        header: Include column headers
    """
    filepath = Path(filepath)

    # Ensure states are (N, 14)
    if states.shape[0] == 14 and states.ndim == 2:
        states = states.T

    # Build data array with time as first column
    data = np.column_stack([times, states])

    columns = [
        "time_days", "r_x_au", "r_y_au", "r_z_au",
        "v_x_au_day", "v_y_au_day", "v_z_au_day", "mass_kg",
        "q0", "q1", "q2", "q3",
        "omega_x_rad_day", "omega_y_rad_day", "omega_z_rad_day"
    ]

    with open(filepath, 'w') as f:
        if header:
            f.write(",".join(columns) + "\n")
        for row in data:
            f.write(",".join(f"{v:.12e}" for v in row) + "\n")

    logger.info(f"Exported {len(times)} points to {filepath}")


def import_trajectory_csv(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Import trajectory from CSV.

    Args:
        filepath: Path to CSV file

    Returns:
        Dict with 'times' and 'states' arrays
    """
    filepath = Path(filepath)

    data = np.loadtxt(filepath, delimiter=',', skiprows=1)

    times = data[:, 0]
    states = data[:, 1:]

    logger.info(f"Imported {len(times)} points from {filepath}")

    return {"times": times, "states": states}
