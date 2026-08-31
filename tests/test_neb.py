"""
Tests for the NEB module (sparc/src/neb.py).

Covers:
- Band construction and interpolation
- Endpoint validation (atom count, symbol ordering)
- Calculator attachment, including endpoints
- Barrier and reaction-energy analysis
- Configuration parsing and validation
- End-to-end climbing-image NEB against a known EMT benchmark
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from ase import Atoms
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import BFGS
from sparc.src.neb import (
    analyse_band,
    attach_calculators,
    build_band,
    run_neb,
    write_visualization,
)

# Reference barrier for Au diffusion on Al(100) with EMT.
EMT_BARRIER_EV = 0.40
EMT_TOLERANCE_EV = 0.05


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def au_al_endpoints(tmp_path):
    """Relaxed Au-on-Al(100) endpoints for the standard EMT benchmark."""
    slab = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(slab, "Au", 1.7, "hollow")
    slab.center(axis=2, vacuum=4.0)
    slab.set_constraint(FixAtoms(mask=[a.symbol != "Au" for a in slab]))

    initial = slab.copy()
    initial.calc = EMT()
    BFGS(initial, logfile=None).run(fmax=0.05)

    final = initial.copy()
    final[-1].x += final.get_cell()[0, 0] / 2.0
    final.calc = EMT()
    BFGS(final, logfile=None).run(fmax=0.05)

    initial_file = tmp_path / "initial.traj"
    final_file = tmp_path / "final.traj"
    write(initial_file, initial)
    write(final_file, final)

    return str(initial_file), str(final_file), initial, final


@pytest.fixture
def simple_pair():
    """Two trivial structures with identical composition and ordering."""
    a = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    b = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    return a, b


# ============================================================
# Band construction
# ============================================================


class TestBuildBand:
    def test_image_count_includes_endpoints(self, simple_pair):
        a, b = simple_pair
        neb = build_band(a, b, n_images=5)
        assert len(neb.images) == 7

    def test_endpoints_preserved(self, simple_pair):
        a, b = simple_pair
        neb = build_band(a, b, n_images=3)
        assert np.allclose(neb.images[0].get_positions(), a.get_positions())
        assert np.allclose(neb.images[-1].get_positions(), b.get_positions())

    def test_interpolation_is_monotonic(self, simple_pair):
        """Intermediate images should progress steadily between endpoints."""
        a, b = simple_pair
        neb = build_band(a, b, n_images=5, interpolation="linear")
        separations = [img.get_distance(0, 1) for img in neb.images]
        assert all(later > earlier for earlier, later in pairwise(separations))

    def test_idpp_interpolation_runs(self, simple_pair):
        a, b = simple_pair
        neb = build_band(a, b, n_images=4, interpolation="idpp")
        assert len(neb.images) == 6

    def test_rejects_unknown_interpolation(self, simple_pair):
        a, b = simple_pair
        with pytest.raises(ValueError, match="interpolation"):
            build_band(a, b, n_images=3, interpolation="quadratic")

    def test_rejects_zero_images(self, simple_pair):
        a, b = simple_pair
        with pytest.raises(ValueError, match="n_images"):
            build_band(a, b, n_images=0)


# ============================================================
# Endpoint validation
# ============================================================


class TestEndpointValidation:
    """NEB maps atoms by index, so endpoint atom lists must agree exactly."""

    def test_rejects_mismatched_atom_count(self):
        a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])
        b = Atoms("H3", positions=[[0, 0, 0], [0, 0, 0.8], [0, 0, 1.6]])
        with pytest.raises(ValueError, match="atom counts"):
            build_band(a, b, n_images=3)

    def test_rejects_reordered_symbols(self):
        """Same composition, different ordering: a silent-corruption case."""
        a = Atoms("HO", positions=[[0, 0, 0], [0, 0, 1.0]])
        b = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1.0]])
        with pytest.raises(ValueError, match="order"):
            build_band(a, b, n_images=3)

    def test_accepts_consistent_ordering(self):
        a = Atoms("HO", positions=[[0, 0, 0], [0, 0, 1.0]])
        b = Atoms("HO", positions=[[0, 0, 0], [0, 0, 1.3]])
        neb = build_band(a, b, n_images=3)
        assert len(neb.images) == 5


# ============================================================
# Calculator attachment
# ============================================================


class TestAttachCalculators:
    def test_every_image_gets_a_calculator(self, simple_pair):
        a, b = simple_pair
        neb = build_band(a, b, n_images=4)
        attach_calculators(neb, lambda: EMT())
        assert all(img.calc is not None for img in neb.images)

    def test_calculators_are_distinct_instances(self, simple_pair):
        """Sharing one calculator across images corrupts results."""
        a, b = simple_pair
        neb = build_band(a, b, n_images=4)
        attach_calculators(neb, lambda: EMT())
        identities = {id(img.calc) for img in neb.images}
        assert len(identities) == len(neb.images)

    def test_endpoints_included_by_default(self, simple_pair):
        """Endpoints need energies for barrier evaluation, if not forces."""
        a, b = simple_pair
        neb = build_band(a, b, n_images=3)
        attach_calculators(neb, lambda: EMT())
        assert neb.images[0].calc is not None
        assert neb.images[-1].calc is not None

    def test_endpoints_can_be_excluded(self, simple_pair):
        a, b = simple_pair
        neb = build_band(a, b, n_images=3)
        attach_calculators(neb, lambda: EMT(), include_endpoints=False)
        assert neb.images[0].calc is None
        assert neb.images[-1].calc is None

    def test_factory_receives_image_index(self, simple_pair):
        """A one-argument factory is given the image index."""
        a, b = simple_pair
        neb = build_band(a, b, n_images=3)
        seen = []

        def factory(index):
            seen.append(index)
            return EMT()

        attach_calculators(neb, factory)
        assert seen == sorted(seen)
        assert len(seen) == len(neb.images)


# ============================================================
# Band analysis
# ============================================================


class TestAnalyseBand:
    def test_barriers_and_reaction_energy(self, simple_pair):
        a, b = simple_pair
        neb = build_band(a, b, n_images=3)
        attach_calculators(neb, lambda: EMT())

        summary = analyse_band(neb.images)

        assert "forward_barrier_eV" in summary
        assert "reverse_barrier_eV" in summary
        assert "reaction_energy_eV" in summary
        assert len(summary["energies_eV"]) == len(neb.images)

    def test_unit_conversion_is_consistent(self, simple_pair):
        a, b = simple_pair
        neb = build_band(a, b, n_images=3)
        attach_calculators(neb, lambda: EMT())
        summary = analyse_band(neb.images)

        ratio = summary["forward_barrier_kcal"] / summary["forward_barrier_eV"]
        assert ratio == pytest.approx(23.060548, rel=1e-6)

    def test_first_image_is_the_energy_reference(self, simple_pair):
        a, b = simple_pair
        neb = build_band(a, b, n_images=3)
        attach_calculators(neb, lambda: EMT())
        summary = analyse_band(neb.images)
        assert summary["relative_energies_eV"][0] == pytest.approx(0.0)

    def test_energy_file_written(self, simple_pair, tmp_path):
        a, b = simple_pair
        neb = build_band(a, b, n_images=3)
        attach_calculators(neb, lambda: EMT())
        summary = analyse_band(neb.images, workdir=tmp_path)
        assert (tmp_path / "neb_energies.dat").exists()
        assert summary["energy_file"].endswith("neb_energies.dat")


# ============================================================
# Visualization export
# ============================================================


class TestVisualization:
    def test_writes_viewer_readable_formats(self, simple_pair, tmp_path):
        a, b = simple_pair
        neb = build_band(a, b, n_images=3)
        attach_calculators(neb, lambda: EMT())
        written = write_visualization(neb.images, workdir=tmp_path)

        assert (tmp_path / "neb_path.xyz").exists()
        assert "xyz" in written

    def test_frame_count_matches_band(self, simple_pair, tmp_path):
        from ase.io import read

        a, b = simple_pair
        neb = build_band(a, b, n_images=5)
        attach_calculators(neb, lambda: EMT())
        write_visualization(neb.images, workdir=tmp_path)

        frames = read(str(tmp_path / "neb_path.xyz"), index=":")
        assert len(frames) == len(neb.images)


# ============================================================
# End-to-end benchmark
# ============================================================


class TestRunNEB:
    """Au diffusion on Al(100): a symmetric path with a known EMT barrier."""

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("neb_e2e")

        slab = fcc100("Al", size=(2, 2, 3))
        add_adsorbate(slab, "Au", 1.7, "hollow")
        slab.center(axis=2, vacuum=4.0)
        slab.set_constraint(FixAtoms(mask=[a.symbol != "Au" for a in slab]))
        slab.calc = EMT()
        BFGS(slab, logfile=None).run(fmax=0.05)

        final = slab.copy()
        final[-1].x += final.get_cell()[0, 0] / 2.0
        final.calc = EMT()
        BFGS(final, logfile=None).run(fmax=0.05)

        write(tmp / "i.traj", slab)
        write(tmp / "f.traj", final)

        return run_neb(
            initial_file=str(tmp / "i.traj"),
            final_file=str(tmp / "f.traj"),
            calculator_factory=lambda: EMT(),
            n_images=5,
            fmax=0.05,
            steps=200,
            workdir=str(tmp / "out"),
        )

    def test_converges(self, result):
        assert result["converged"]

    def test_barrier_matches_reference(self, result):
        assert result["forward_barrier_eV"] == pytest.approx(
            EMT_BARRIER_EV, abs=EMT_TOLERANCE_EV
        )

    def test_symmetric_path_is_thermoneutral(self, result):
        """The two hollow sites are equivalent, so dE must vanish."""
        assert result["reaction_energy_eV"] == pytest.approx(0.0, abs=0.01)

    def test_forward_and_reverse_barriers_agree(self, result):
        assert result["forward_barrier_eV"] == pytest.approx(
            result["reverse_barrier_eV"], abs=0.01
        )

    def test_saddle_is_interior(self, result):
        """An energy maximum on an endpoint means the band brackets no barrier."""
        assert 0 < result["saddle_index"] < 6

    def test_outputs_written(self, result):
        from pathlib import Path

        assert Path(result["saddle_file"]).exists()
        assert Path(result["path_file"]).exists()

    def test_rejects_unknown_optimizer(self, au_al_endpoints, tmp_path):
        initial_file, final_file, _, _ = au_al_endpoints
        with pytest.raises(ValueError, match="optimizer"):
            run_neb(
                initial_file=initial_file,
                final_file=final_file,
                calculator_factory=lambda: EMT(),
                optimizer="CONJUGATE_GRADIENT",
                workdir=str(tmp_path),
            )


# ============================================================
# Configuration
# ============================================================


class TestNEBConfig:
    def test_defaults_are_inactive(self):
        from sparc.src.utils.read_input import NEBConfig

        cfg = NEBConfig()
        assert cfg.run is False
        assert cfg.climb is True
        assert cfg.interpolation == "idpp"

    def test_disabled_config_skips_validation(self):
        """An absent NEB section must not require structure files."""
        from sparc.src.utils.read_input import NEBConfig

        NEBConfig(run=False, initial_structure=None)

    def test_enabled_config_requires_structures(self):
        from sparc.src.utils.read_input import NEBConfig

        with pytest.raises(Exception, match="initial_structure"):
            NEBConfig(run=True)

    def test_missing_structure_file_is_reported(self, tmp_path):
        from sparc.src.utils.read_input import NEBConfig

        present = tmp_path / "a.xyz"
        write(present, Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]]))

        with pytest.raises(Exception, match="not found"):
            NEBConfig(
                run=True,
                initial_structure=str(present),
                final_structure=str(tmp_path / "missing.xyz"),
            )

    def test_loose_threshold_must_exceed_tight(self, tmp_path):
        from sparc.src.utils.read_input import NEBConfig

        a = tmp_path / "a.xyz"
        b = tmp_path / "b.xyz"
        write(a, Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]]))
        write(b, Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.4]]))

        with pytest.raises(Exception, match="fmax_loose"):
            NEBConfig(
                run=True,
                initial_structure=str(a),
                final_structure=str(b),
                fmax=0.5,
                fmax_loose=0.05,
            )

    def test_config_reaches_sparcconfig(self, tmp_path):
        from sparc.src.utils.read_input import SparcConfig

        cfg = SparcConfig.__dataclass_fields__
        assert "neb" in cfg
