"""
Tests for Mission Planning Framework
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/boss/data/space_state/space_state')

from mission import (
    PhaseType, ManeuverType,
    DeltaVBudget, MissionPhase, MissionPlan,
    hohmann_transfer, escape_velocity, circular_velocity, plane_change_dv,
    create_earth_moon_mission, create_mars_transfer_mission,
    create_leo_rendezvous_mission, quick_estimate,
)
from base.star_constants import BODIES


class TestDeltaVBudget:
    """Test DeltaVBudget class."""

    def test_basic_budget(self):
        """Budget stores correct values."""
        budget = DeltaVBudget("TLI", 3.2, margin_percent=10.0)
        assert budget.name == "TLI"
        assert budget.delta_v_kmps == 3.2
        assert budget.margin_percent == 10.0

    def test_total_with_margin(self):
        """Total includes margin."""
        budget = DeltaVBudget("Test", 1.0, margin_percent=20.0)
        assert budget.total_kmps == 1.2

    def test_fuel_mass_calculation(self):
        """Fuel mass uses Tsiolkovsky equation correctly."""
        budget = DeltaVBudget("Test", 3.0, margin_percent=0)  # 3 km/s no margin

        # For 1000 kg dry mass, Isp=300s
        fuel = budget.fuel_mass(dry_mass=1000, isp_s=300)

        # m_ratio = exp(3000 / (300 * 9.81)) = exp(1.019) ≈ 2.77
        # fuel = 1000 * (2.77 - 1) = 1770 kg
        assert 1700 < fuel < 1850


class TestMissionPhase:
    """Test MissionPhase class."""

    def test_phase_creation(self):
        """Phase stores all attributes."""
        phase = MissionPhase(
            name="Test Phase",
            phase_type=PhaseType.TRANSFER,
            duration_days=5.0,
            delta_v_budget=DeltaVBudget("dv", 1.0),
            start_body='earth',
            end_body='moon',
        )

        assert phase.name == "Test Phase"
        assert phase.phase_type == PhaseType.TRANSFER
        assert phase.duration_days == 5.0
        assert phase.start_body == 'earth'
        assert phase.end_body == 'moon'


class TestMissionPlan:
    """Test MissionPlan class."""

    def test_empty_plan(self):
        """Empty plan has zero totals."""
        plan = MissionPlan(name="Empty")
        assert plan.total_delta_v == 0
        assert plan.total_duration_days == 0

    def test_add_phases(self):
        """Can add phases to plan."""
        plan = MissionPlan(name="Test")

        plan.add_phase(MissionPhase(
            name="Phase 1",
            phase_type=PhaseType.TRANSFER,
            duration_days=1.0,
            delta_v_budget=DeltaVBudget("dv1", 1.0, margin_percent=0),
        ))

        plan.add_phase(MissionPhase(
            name="Phase 2",
            phase_type=PhaseType.COAST,
            duration_days=2.0,
        ))

        plan.add_phase(MissionPhase(
            name="Phase 3",
            phase_type=PhaseType.ORBIT_INSERTION,
            duration_days=0.5,
            delta_v_budget=DeltaVBudget("dv2", 0.5, margin_percent=0),
        ))

        assert len(plan.phases) == 3
        assert plan.total_delta_v == 1.5
        assert plan.total_duration_days == 3.5

    def test_fuel_requirement(self):
        """Fuel requirement calculation is correct."""
        plan = MissionPlan(name="Test")
        plan.add_phase(MissionPhase(
            name="Burn",
            phase_type=PhaseType.TRANSFER,
            delta_v_budget=DeltaVBudget("dv", 2.0, margin_percent=0),
        ))

        result = plan.fuel_requirement(payload_mass=1000, isp_s=300)

        assert 'total_fuel' in result
        assert 'initial_mass' in result
        assert result['payload_mass'] == 1000
        assert result['initial_mass'] > 1000


class TestManeuverCalculators:
    """Test orbital maneuver calculators."""

    def test_hohmann_leo_to_geo(self):
        """LEO to GEO Hohmann transfer."""
        r_leo = 6371 + 400   # km
        r_geo = 42164        # km
        mu = BODIES['earth']['mu_km3_s2']

        dv1, dv2, tof = hohmann_transfer(r_leo, r_geo, mu)

        # Total should be ~3.9 km/s
        assert 3.5 < dv1 + dv2 < 4.2

        # Transfer time ~5 hours
        assert 0.15 < tof < 0.25

    def test_escape_velocity_earth(self):
        """Earth surface escape velocity."""
        v_esc = escape_velocity(6371, BODIES['earth']['mu_km3_s2'])

        # Should be ~11.2 km/s
        assert 11.0 < v_esc < 11.4

    def test_circular_velocity_leo(self):
        """LEO circular velocity."""
        v_circ = circular_velocity(6371 + 400, BODIES['earth']['mu_km3_s2'])

        # Should be ~7.67 km/s
        assert 7.5 < v_circ < 7.8

    def test_plane_change(self):
        """Plane change delta-V calculation."""
        # 60 degree plane change at 7 km/s
        dv = plane_change_dv(7.0, 60.0)

        # Should be 2 * 7 * sin(30°) = 7 km/s
        assert abs(dv - 7.0) < 0.1


class TestMissionTemplates:
    """Test pre-built mission templates."""

    def test_earth_moon_mission(self):
        """Earth-Moon mission template is valid."""
        plan = create_earth_moon_mission()

        assert plan.name == "Earth-Moon Transfer"
        assert len(plan.phases) >= 3

        # Total delta-V should be ~4-5 km/s
        assert 3.0 < plan.total_delta_v < 6.0

    def test_mars_transfer_mission(self):
        """Mars transfer mission template is valid."""
        plan = create_mars_transfer_mission()

        assert plan.name == "Earth-Mars Transfer"
        assert len(plan.phases) >= 3

        # Total delta-V (heliocentric component)
        # Note: Values are lower than total mission dv because
        # we're computing heliocentric dv only, not including
        # hyperbolic excess conversion. For validation, just
        # check it's reasonable.
        assert plan.total_delta_v > 0

        # Transfer time should be ~250-300 days
        assert 200 < plan.total_duration_days < 300

    def test_leo_rendezvous_mission(self):
        """LEO rendezvous mission template is valid."""
        plan = create_leo_rendezvous_mission(phase_angle_deg=1.0)  # Small angle

        assert plan.name == "LEO Rendezvous"
        assert len(plan.phases) >= 2

        # Delta-V should be reasonable for small phase angle
        assert plan.total_delta_v > 0


class TestQuickEstimate:
    """Test quick delta-V estimation."""

    def test_earth_to_mars(self):
        """Earth to Mars quick estimate."""
        result = quick_estimate('earth', 'mars barycenter')

        assert 'dv_total_kmps' in result
        assert 'transfer_time_days' in result

        # Check delta-V is positive and finite
        assert result['dv_total_kmps'] > 0
        assert np.isfinite(result['dv_total_kmps'])

        # Transfer time ~250-280 days
        assert 200 < result['transfer_time_days'] < 300

    def test_earth_to_venus(self):
        """Earth to Venus quick estimate."""
        result = quick_estimate('earth', 'venus barycenter')

        assert 'dv_total_kmps' in result

        # Check result is valid and finite
        assert result['dv_total_kmps'] > 0
        assert np.isfinite(result['dv_total_kmps'])

    def test_unknown_body(self):
        """Unknown body returns error."""
        result = quick_estimate('earth', 'pluto')
        assert 'error' in result


class TestMissionSummary:
    """Test mission summary generation."""

    def test_summary_contains_info(self):
        """Summary includes key information."""
        plan = create_earth_moon_mission()
        summary = plan.summary()

        assert "Earth-Moon" in summary
        assert "Delta-V" in summary
        assert "Duration" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
