#!/usr/bin/python3

# sparc.py

"""
SPARC (**S**mart **P**otential with **A**tomistic **R**are Events and **C**ontinuous Learning)

Main module coordinating the active learning workflow for reactive ML potentials.

Sequentially triggers:
- Data processing
- MD simulations (ab initio and ML/MD) with NVE/NVT/NPT ensembles
- Model training using DeepMD-kit
- Active learning iterations

Usage:
    python sparc.py -i input.yaml
"""

# ===============================
# Standard library imports
# ===============================
import argparse
import os
import sys
from pathlib import Path
from typing import List, Union

from ase import Atoms, units

# ===============================
# Third-party imports
# ===============================
from ase.io import read

from sparc.src.active_learning import QueryByCommittee
from sparc.src.ase_md import (
    NPT,
    NVE,
    CalculateDFTEnergy,
    ExecuteAbInitioDynamics,
    ExecuteMlpDynamics,
    LangevinNVT,
    NoseNVT,
)
from sparc.src.calculator import dft_calculator
from sparc.src.data_processing import get_data
from sparc.src.deepmd import deepmd_training, get_version, setup_DeepPotential
from sparc.src.finetune import finetune_training
from sparc.src.forcecorrection.corrector import correct_aimd_forces
from sparc.src.plumed_wrapper import modify_forces, umbrella
from sparc.src.utils.banner import banner
from sparc.src.utils.logger import SparcLog, setup_logger

# MDLogger removed — log_md_setup in utils handles logging with proper float conversion
# ===============================
# Local imports
# ===============================
from sparc.src.utils.read_input import (
    AIMDSetupConfig,
    MLIPSetupConfig,
    SparcConfig,
    load_config,
)
from sparc.src.utils.timing import WorkflowTimer
from sparc.src.utils.utils import (
    combine_trajectories,
    create_iteration_dirs,
    get_initial_structure,
    load_progress,
    remove_backup_files,
    restart_progress,
    save_progress,
)

# ===================================================================================================#
# Analysis mode handler
# ===================================================================================================#
if "--analysis" in sys.argv:
    from sparc.src.utils.analysis import main as compute_energy

    iidx = sys.argv.index("--analysis")
    analysis_args = sys.argv[iidx + 1 :]
    sys.argv = [sys.argv[0]] + analysis_args
    compute_energy()
    sys.exit(0)

# ===================================================================================================#
# Force correction mode handler
# ===================================================================================================#
if "--forcecorrect" in sys.argv:
    from sparc.src.forcecorrection.cli import main as correct_forces
    from sparc.src.utils.logger import close_logger, setup_logger

    setup_logger(enable=True)
    iidx = sys.argv.index("--forcecorrect")
    correct_args = sys.argv[iidx + 1 :]
    sys.argv = [sys.argv[0]] + correct_args
    try:
        correct_forces()
    finally:
        close_logger()
    sys.exit(0)

# ===================================================================================================#
# Helper Functions
# ===================================================================================================#


def initialize_thermostat(
    config_block: Union[AIMDSetupConfig, MLIPSetupConfig],
    atoms: Atoms,
    restart: bool = False,
):
    """
    Initialize MD ensemble (NVE/NVT/NPT) from configuration block.

    Parameters
    ----------
    config_block : AIMDSetupConfig or MLIPSetupConfig
        Configuration section containing ensemble, temperature, thermostat settings
    atoms : ase.Atoms
        Atomic system
    restart : bool
        Whether to restart from checkpoint

    Returns
    -------
    dyn : MD dynamics object
        Initialized ensemble
    """
    ensemble = config_block.ensemble
    timestep = config_block.timestep_fs

    if ensemble == "NVE":
        return NVE(system=atoms, timestep=timestep, restart=restart)

    elif ensemble == "NVT":
        thermostat = config_block.thermostat
        thermostat_type = thermostat.type
        temperature = config_block.temperature

        if thermostat_type == "Nose":
            return NoseNVT(
                atoms=atoms,
                timestep=timestep,
                temperature=temperature,
                tdamp=thermostat.tdamp,
                restart=restart,
            )
        elif thermostat_type == "Langevin":
            return LangevinNVT(
                atoms=atoms,
                timestep=timestep,
                temperature=temperature,
                friction=thermostat.friction,
                restart=restart,
            )
        else:
            raise ValueError(f"Unsupported thermostat type: {thermostat_type}")

    elif ensemble == "NPT":
        temperature = config_block.temperature

        return NPT(
            system=atoms,
            timestep=timestep,
            temperature=temperature,
            tau_t=config_block.tau_t,
            pressure=config_block.pressure,
            tau_p=config_block.tau_p,
            compressibility=getattr(config_block, "compressibility", None),
            restart=restart,
        )

    else:
        raise ValueError(f"Unsupported ensemble: {ensemble}")


def setup_plumed_calc(
    calculator,
    system: Atoms,
    config: SparcConfig,
    iteration_dir: Path,
    use_aimd: bool = True,
):
    """
    Wrap calculator with PLUMED if enabled.

    Parameters
    ----------
    calculator : ASE calculator
        Base calculator (DFT or DeepMD)
    system : ase.Atoms
        Atomic system
    config : SparcConfig
        Full configuration object
    iteration_dir : Path
        Directory for PLUMED output
    use_aimd : bool
        If True, use aimd_setup plumed settings; else use mlip_setup plumed settings

    Returns
    -------
    calc : ASE calculator
        Calculator wrapped with PLUMED forces or original if PLUMED disabled
    """
    if use_aimd:
        plumed_config = config.aimd_setup.plumed
        timestep = config.aimd_setup.timestep_fs
    else:
        plumed_config = config.mlip_setup.plumed
        timestep = config.mlip_setup.timestep_fs

    if plumed_config.enabled:
        SparcLog("=" * 80)
        SparcLog("PLUMED ENABLED FOR MD SIMULATION")
        SparcLog("=" * 80)
        return modify_forces(
            calculator=calculator,
            system=system,
            timestep=timestep * units.fs,
            kT=plumed_config.kT,
            restart=plumed_config.restart,
            plumed_input=plumed_config.plumed_file,
            iteration=iteration_dir,
        )

    return calculator


def load_structure(
    structure_file: Union[str, List, Path], index: int = 0, non_periodic: bool = False
) -> Atoms:
    """Load structure from file or list of files."""
    if isinstance(structure_file, list):
        if len(structure_file) == 0:
            raise ValueError(
                "general.structure_file is an empty list."
                "please provide at-least one structure!"
            )
        if index >= len(structure_file):
            raise IndexError(
                f"Requested structure_file[{index}], but only "
                f"{len(structure_file)} structure file(s) were provided.\n"
                "struct_idx should be i // multiple_run — check get_num_samples() logic."
            )
        atoms = read(structure_file[index])
    else:
        atoms = read(structure_file)

    if non_periodic and (atoms.get_pbc().any() or atoms.cell.any()):
        SparcLog("")
        SparcLog(
            "WARNING: Structure has periodic cell/PBC but the calculator is non-periodic.",
            level="WARNING",
        )
        SparcLog(
            "         Removing cell and PBC. This will not affect the calculation.",
            level="WARNING",
        )
        SparcLog("")
        atoms.set_pbc(False)
        atoms.set_cell([0, 0, 0])

    return atoms


def get_num_samples(config: SparcConfig) -> int:
    """Total ML-MD runs = n_structures × multiple_run (cross-product).

    Scenarios:
        1 file,  multiple_run=1 → 1 run
        1 file,  multiple_run=M → M independent shoots from the same file
        N files, multiple_run=1 → N runs (one per file)
        N files, multiple_run=M → N x M runs (M shoots per file)
    """
    sf = config.general.structure_file
    n_structures = len(sf) if isinstance(sf, list) else 1
    multiple_run = config.mlip_setup.multiple_run or 1
    return n_structures * multiple_run


def get_ensemble_name(config_block: Union[AIMDSetupConfig, MLIPSetupConfig]) -> str:
    """Get descriptive name for ensemble setup."""
    ensemble = config_block.ensemble
    if ensemble == "NVT":
        thermostat_type = config_block.thermostat.type
        return f"NVT-{thermostat_type}"
    return ensemble


# ===================================================================================================#
# Main Workflow
# ===================================================================================================#


def main():
    """Main function coordinating the entire SPARC workflow."""
    from sparc.src.utils.logger import close_logger

    try:
        _main_workflow()
    finally:
        close_logger()


def _main_workflow():
    """Inner workflow function - separated so main() can guarantee logger cleanup."""
    # ---------------------------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="SPARC — Smart Potential with Atomistic Rare Events and Continuous Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  sparc -i input.yaml            Run main workflow (AIMD / MLIP training / ML-MD / AL)
  sparc --analysis [args]        Analyse model deviation and training data
  sparc --forcecorrect [args]    Remove PLUMED bias forces from AIMD trajectory

run 'sparc --analysis -h' or 'sparc --forcecorrect -h' for mode-specific options.
        """,
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        default="input.yaml",
        help="Input YAML configuration file (default: input.yaml)",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run analysis mode (use 'sparc --analysis -h' for options)",
    )
    parser.add_argument(
        "--forcecorrect",
        action="store_true",
        help="Run force correction mode (use 'sparc --forcecorrect -h' for options)",
    )
    args = parser.parse_args()

    # Logger created only after -h/--help exits — no stray Sparc.log on help calls
    setup_logger(enable=True)
    banner()
    try:
        config = load_config(args.input_file)
        SparcLog("=" * 80)
        SparcLog(f"{'YAML configuration loaded successfully!':^80}")
        SparcLog("=" * 80)
    except FileNotFoundError:
        SparcLog(f"{'Error: YAML file not found!':^80}")
        sys.exit(1)
    except Exception as e:
        SparcLog(f"Error loading configuration: {e}")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # System initialization
    # ---------------------------------------------------------------------------
    parent_dir = Path(os.getcwd())
    timer = WorkflowTimer(parent_dir)
    structure_file = config.general.structure_file
    non_periodic = config.dft_calculator.engine.upper() in ("GAUSSIAN", "ORCA", "XTB")
    if non_periodic and config.aimd_setup.ensemble == "NPT":
        raise ValueError(
            f"NPT ensemble requires a periodic cell for barostat coupling, "
            f"but {config.dft_calculator.engine} is a non-periodic calculator. "
            f"Use NVT or NVE instead."
        )
    system = load_structure(structure_file, index=0, non_periodic=non_periodic)
    atom_types = list(dict.fromkeys(system.get_chemical_symbols()))

    if system.get_pbc().any():
        system.set_pbc([True, True, True])
        system.center()

    _original_system = system.copy()

    # SparcLog("="*72)
    # SparcLog(f"STRUCTURE INFORMATION: {system}")
    # SparcLog("="*72)
    # System information
    # system = read(config.general.structure_file)
    # Handle single file or list of files
    if isinstance(structure_file, list):
        SparcLog(
            f"Using structure file: {structure_file[0]} (1 of {len(structure_file)})"
        )
    else:
        SparcLog(f"Using structure file: {structure_file}")

    SparcLog("WORKFLOW PARAMETERS :")
    SparcLog("---------------------")
    SparcLog(f"{'Structure FileName/s':<30} {config.general.structure_file}")
    SparcLog(f"{'Number of Atoms':<30} {len(system)}")
    SparcLog(f"{'Chemical Formula':<30} {system.get_chemical_formula()}")
    SparcLog("")
    # DFT settings
    SparcLog(f"{'DFT Engine':<30} {config.dft_calculator.engine}")
    SparcLog(f"{'DFT Tempelate File':<30} {config.dft_calculator.template_file}")
    SparcLog(
        f"{'DFT Executable':<30} {config.dft_calculator.exe_command or 'Auto-detected'}"
    )
    SparcLog("")

    # MLIP training info
    if config.mlip_setup.training or config.active_learning:
        SparcLog(
            f"{'MLIP Training':<30} {'Enabled' if config.mlip_setup.training else 'Disabled'}"
        )
        SparcLog(f"{'MLIP Training Filename':<30} {config.mlip_setup.input_file}")
        SparcLog(f"{'MLIP Training Data':<30} {config.mlip_setup.data_dir}")
        SparcLog(f"{'Number of MLIP Models':<30} {config.mlip_setup.num_models}")
        SparcLog("")

    # Active learning info
    SparcLog(
        f"{'Active Learning':<30} {'Enabled' if config.active_learning else 'Disabled'}"
    )
    if config.active_learning:
        SparcLog(f"{'AL Iterations':<30} {config.iteration}")
        SparcLog(
            f"{'Force deviation range':<30} [{config.model_dev.f_min_dev:.2f}, {config.model_dev.f_max_dev:.2f}] eV/Å"
        )
        if config.learning_restart:
            SparcLog(f"{'Restarting AL Iterations':<30} {'Enabled'}")
            SparcLog(f"{'Latest Model':<30} {config.latest_model}")
        else:
            SparcLog(f"{'Restarting AL Iterations':<30} {'Disabled'}")
        SparcLog("")
    # Distance constraints
    if config.distance_metrics:
        SparcLog(f"{'Distance Constraints':<30} {len(config.distance_metrics)}")

    SparcLog("=" * 80)
    SparcLog("")
    # sys.exit(1)
    # ===========================================================================
    # SECTION 0: Nudged Elastic Band (standalone task)
    # ===========================================================================
    # NEB locates a minimum energy path between two fixed endpoints and does
    # not feed into the AIMD / training / active-learning pipeline, so it runs
    # on its own and the workflow returns once it completes.
    if config.neb.run:
        from sparc.src.neb import run_neb_from_config

        summary = run_neb_from_config(config)

        if not summary["converged"]:
            SparcLog("[NEB] Optimizer did not converge within the step limit.")
            sys.exit(1)
        return

    # Extract workflow flags from config
    DftMDSteps = config.aimd_setup.steps
    dftmd_is = DftMDSteps > 0
    training_is = config.mlip_setup.training
    dpmd_run_is = config.mlip_setup.MdSimulation

    if dftmd_is or training_is or dpmd_run_is:
        iter_structure = create_iteration_dirs(iter_num=0)

    # ===========================================================================
    # SECTION 1: Ab Initio Molecular Dynamics (AIMD)
    # ===========================================================================
    if dftmd_is:
        dft_calc = dft_calculator(config, True)
        ensemble = config.aimd_setup.ensemble
        ensemble_name = get_ensemble_name(config.aimd_setup)
        temperature = config.aimd_setup.temperature

        SparcLog("=" * 80)
        if ensemble == "NVE":
            SparcLog(f"Ab-initio MD using {ensemble_name} ensemble".center(80))
        else:
            SparcLog(f"Ab-initio MD using {ensemble_name} at {temperature}K".center(80))
        SparcLog(f"{'MD Ensemble':<30} {config.aimd_setup.ensemble}")
        SparcLog(f"{'Temperature':<30} {config.aimd_setup.temperature:.1f} K")
        if config.aimd_setup.temp_end is not None:
            SparcLog(
                f"{'Temperature Ramping':<30} {config.aimd_setup.temperature:.1f} K --> {config.aimd_setup.temp_end:.1f} K"
            )
            # WARNING: Check thermostat compatibility
            if config.aimd_setup.thermostat.type == "Nose":
                SparcLog("")
                SparcLog(
                    "WARNING: Temperature ramping with Nose-Hoover may not work properly.",
                    level="WARNING",
                )
                SparcLog(
                    "         Nose-Hoover thermostat resists rapid temperature changes.",
                    level="WARNING",
                )
                SparcLog(
                    "         Recommended: Use Langevin thermostat for temperature ramping.",
                    level="WARNING",
                )
                SparcLog("")
        SparcLog(f"{'MD Thermostat':<30} {config.aimd_setup.thermostat.type}")
        SparcLog(f"{'Timestep':<30} {config.aimd_setup.timestep_fs:.2f} fs")
        SparcLog(f"{'AIMD Steps':<30} {config.aimd_setup.steps}")
        SparcLog(f"{'Output Trajectory':<30} {config.output.aimdtraj_file}")
        SparcLog(f"{'Output Logfile':<30} {config.output.log_file}")
        SparcLog("")

        _dft_step = timer.start_step(
            0, "dft", "00.dft", count=DftMDSteps if dftmd_is else None
        )
        # Initialize ensemble
        dyn_dft = initialize_thermostat(
            config.aimd_setup, system, restart=config.aimd_setup.restart
        )

        # Setup calculator with optional PLUMED
        system.calc = setup_plumed_calc(
            dft_calc, system, config, iter_structure["dft_dir"], use_aimd=True
        )

        # Run AIMD
        ExecuteAbInitioDynamics(
            system=system,
            dyn=dyn_dft,
            steps=DftMDSteps,
            pace=config.aimd_setup.log_frequency,
            log_filename=config.output.log_file,
            trajfile=config.output.aimdtraj_file,
            dir_name=iter_structure["dft_dir"],
            name=ensemble_name,
            temp_start=config.aimd_setup.temperature,
            temp_end=config.aimd_setup.temp_end,
        )

        # Remove PLUMED bias forces so training data has physical forces
        fc = config.aimd_setup.plumed.force_correction
        if config.aimd_setup.plumed.enabled and fc.enabled:
            correct_aimd_forces(
                traj_path=iter_structure["dft_dir"] / config.output.aimdtraj_file,
                dft_dir=iter_structure["dft_dir"],
                cv_atoms=fc.cv_atoms,
                cv_force_file=fc.cv_force_file,
                cv_derivs_file=fc.cv_derivs_file,
            )
        timer.end_step(_dft_step)

    # ===========================================================================
    # SECTION 2: ML Potential Model Training
    # ===========================================================================
    datadir = parent_dir / config.mlip_setup.data_dir

    if training_is:
        SparcLog("=" * 80)
        SparcLog("Processing AIMD Trajectory for MLIP Training")
        SparcLog(f"Training ensemble of {config.mlip_setup.num_models} models")
        SparcLog("Backend: DeePMD-kit")
        SparcLog("=" * 80)
        SparcLog("")

        _train_step = timer.start_step(0, "train", "01.train")
        get_data(
            ase_traj=iter_structure["dft_dir"] / config.output.aimdtraj_file,
            dir_name=datadir,
            skip_min=config.mlip_setup.skip_min,
            skip_max=config.mlip_setup.skip_max,
            seed=config.mlip_setup.seed,
            train_ratio=config.mlip_setup.train_ratio,
        )

        finetune_config = config.finetune
        if finetune_config.enabled:
            finetune_training(
                finetune_config=finetune_config,
                datadir=datadir,
                atom_types=atom_types,
                training_dir=iter_structure["train_dir"],
                num_models=config.mlip_setup.num_models,
                input_file=config.mlip_setup.input_file,
            )
        else:
            deepmd_training(
                active_learning=False,
                training_dir=iter_structure["train_dir"],
                num_models=config.mlip_setup.num_models,
                input_file=config.mlip_setup.input_file,
                datadir=datadir,
                atom_types=atom_types,
            )
        timer.end_step(_train_step)

    # ===========================================================================
    # SECTION 3: ML Potential Molecular Dynamics (ML/MD)
    # ===========================================================================
    if dpmd_run_is:
        n_sample = get_num_samples(config)
        ensemble_name = get_ensemble_name(config.mlip_setup)
        temperature = config.mlip_setup.temperature

        # SparcLog("="*80)
        # SparcLog(f"Running {n_sample} ML-MD simulation(s) with {ensemble_name}")
        # SparcLog("="*80)
        SparcLog("=" * 80)
        if ensemble_name == "NVE":
            SparcLog(
                f"Running {n_sample} ML-MD simulation(s) for {ensemble_name} ensemble".center(
                    80
                )
            )
        else:
            SparcLog(
                f"Running {n_sample} ML-MD simulation(s) for {ensemble_name} at {temperature}K".center(
                    80
                )
            )
        SparcLog(f"{'MD Ensemble':<30} {config.mlip_setup.ensemble}")
        SparcLog(f"{'Temperature':<30} {config.mlip_setup.temperature:.1f} K")
        if config.mlip_setup.temp_end is not None:
            SparcLog(
                f"{'Temperature Ramping':<30} {config.mlip_setup.temperature:.1f} K --> {config.mlip_setup.temp_end:.1f} K"
            )
            # WARNING: Check thermostat compatibility
            if config.mlip_setup.thermostat.type == "Nose":
                SparcLog("")
                SparcLog(
                    "WARNING: Temperature ramping with Nose-Hoover may not work properly.",
                    level="WARNING",
                )
                SparcLog(
                    "         Nose-Hoover thermostat resists rapid temperature changes.",
                    level="WARNING",
                )
                SparcLog(
                    "         Recommended: Use Langevin thermostat for temperature ramping.",
                    level="WARNING",
                )
                SparcLog("")
        SparcLog(f"{'MD Thermostat':<30} {config.mlip_setup.thermostat.type}")
        SparcLog(f"{'Timestep':<30} {config.mlip_setup.timestep_fs:.2f} fs")
        SparcLog(f"{'ML-MD Steps':<30} {config.mlip_setup.md_steps}")
        SparcLog(f"{'Output Trajectory':<30} {config.output.dptraj_file}")
        SparcLog("")

        dp_path = iter_structure["train_dir"]
        # Auto-detect model file by checking what exists on disk
        pth_model = os.path.join(str(dp_path), "training_1", "frozen_model_1.pth")
        pb_model = os.path.join(str(dp_path), "training_1", "frozen_model_1.pb")
        if os.path.exists(pth_model):
            dp_model = "training_1/frozen_model_1.pth"
        elif os.path.exists(pb_model):
            dp_model = "training_1/frozen_model_1.pb"
        else:
            # Fall back to backend detection
            _, backend = get_version()
            dp_model = (
                "training_1/frozen_model_1.pth"
                if backend == "pytorch"
                else "training_1/frozen_model_1.pb"
            )

        MDsteps = config.mlip_setup.md_steps
        writePace = config.mlip_setup.log_frequency

        # Check for umbrella sampling
        plumed_config = config.mlip_setup.plumed
        umbrella_enabled = plumed_config.umbrella_sampling.enabled

        _mlmd_step = timer.start_step(0, "mlmd", "02.dpmd")
        if umbrella_enabled:
            SparcLog("=" * 80)
            SparcLog(
                "Umbrella Sampling Enabled — Running ML-MD Windows with PLUMED".center(
                    80
                )
            )
            SparcLog("=" * 80)

            umbrella(
                config=config, us_dir=iter_structure, dp_path=dp_path, dp_model=dp_model
            )

        else:
            # Run standard MLMD — cross-product of structures × multiple_run
            _multiple_run = config.mlip_setup.multiple_run or 1
            for i in range(n_sample):
                struct_idx = i // _multiple_run  # which structure file
                run_idx = i % _multiple_run  # which shoot for that structure
                struct_name = (
                    structure_file[struct_idx]
                    if isinstance(structure_file, list)
                    else structure_file
                )
                SparcLog(
                    f"ML-MD Run {i + 1}/{n_sample} | Pre-AL | Structure: {struct_name} (s{struct_idx}) | Shoot {run_idx + 1}/{_multiple_run}"
                )
                dp_system = load_structure(structure_file, index=struct_idx)

                dp_atoms, dp_calc = setup_DeepPotential(
                    atoms=dp_system, model_path=dp_path, model_name=dp_model
                )

                # Initialize ensemble
                dyn_dp = initialize_thermostat(config.mlip_setup, dp_atoms)

                # Setup calculator with optional PLUMED (iter 0 pre-AL run)
                if plumed_config.enabled and plumed_config.start_iteration == 0:
                    remove_backup_files(file_ext="bck.*")
                    dp_atoms.calc = setup_plumed_calc(
                        dp_calc,
                        dp_atoms,
                        config,
                        iter_structure["dpmd_dir"],
                        use_aimd=False,
                    )

                # Run MLMD
                ExecuteMlpDynamics(
                    system=dp_atoms,
                    dyn=dyn_dp,
                    steps=MDsteps,
                    pace=writePace,
                    log_filename=f"Iter0_dpmd_s{struct_idx}_r{run_idx}.log",
                    trajfile=config.output.dptraj_file,
                    dir_name=iter_structure["dpmd_dir"],
                    distance_metrics=config.distance_metrics,
                    name=ensemble_name,
                    epot_threshold=getattr(config.mlip_setup, "epot_threshold", None),
                    temp_start=config.mlip_setup.temperature,
                    temp_end=config.mlip_setup.temp_end,
                    restart=config.mlip_setup.restart,
                )
        timer.end_step(_mlmd_step)

        # Query by committee for active learning
        if config.active_learning:
            _qbc_step = timer.start_step(0, "qbc", "02.dpmd")
            candidate_found_is, candidates_file, candidate_idx, latest_models = (
                QueryByCommittee(
                    trajfile=iter_structure["dpmd_dir"] / config.output.dptraj_file,
                    model_path=iter_structure["train_dir"],
                    num_models=config.mlip_setup.num_models,
                    min_lim=config.model_dev.f_min_dev,
                    max_lim=config.model_dev.f_max_dev,
                    dpmd_data_path=iter_structure["dpmd_dir"],
                    iteration=0,
                    rmsd_threshold=config.model_dev.rmsd_threshold,
                    exclude_hydrogen=config.model_dev.exclude_hydrogen,
                    deal_config=config.deal if config.deal.enabled else None,
                    structure_file=(
                        config.general.structure_file[0]
                        if isinstance(config.general.structure_file, list)
                        else config.general.structure_file
                    ),
                    deepmd_input_json=config.mlip_setup.input_file,
                )
            )
            timer.end_step(_qbc_step)
            save_progress(
                {
                    "state": str(iter_structure["dft_dir"]),
                    "iteration": 1,
                    "candidate": candidate_idx,
                    "idx": 1,
                }
            )

            if candidate_idx < config.min_candidates:
                SparcLog("=" * 80)
                SparcLog(
                    f"Candidates found ({candidate_idx}) below threshold ({config.min_candidates}). Stopping AL."
                )
                SparcLog("End of Active Learning Loop")
                SparcLog("=" * 80)
                timer.log_summary()
                return

    # ===========================================================================
    # SECTION 4: Active Learning Protocol
    # ===========================================================================
    learning_is = config.active_learning
    learning_restart = config.learning_restart
    al_iter = config.iteration

    # Initialize active learning state with safe defaults
    # These may have been set by Section 3 (ML/MD), or need defaults
    if "candidate_found_is" not in dir():
        candidate_found_is = False
    if "candidates_file" not in dir():
        candidates_file = ""
    if "candidate_idx" not in dir():
        candidate_idx = 0
    if "latest_models" not in dir():
        _ver, _bknd = get_version()
        _default_model = "frozen_model.pth" if _bknd == "pytorch" else "frozen_model.pb"
        latest_models = [getattr(config, "latest_model", _default_model)]

    SparcLog("")
    SparcLog("-" * 80)
    SparcLog(
        f"ACTIVE LEARNING CYCLE : {'ENABLED' if learning_is else 'DISABLED'}".center(80)
    )
    SparcLog("-" * 80)
    # SparcLog(f"Status: {'ENABLED' if learning_is else 'DISABLED'}")
    if learning_is:
        SparcLog(f"Maximum Iterations: {al_iter}")
        SparcLog(
            f"Force Deviation Thresholds: [{config.model_dev.f_min_dev}, {config.model_dev.f_max_dev}] eV/Å"
        )
        SparcLog(f"MLIP Ensemble Size: {config.mlip_setup.num_models} models")
    SparcLog("=" * 80)
    SparcLog("")

    if not learning_is:
        timer.log_summary()
        return

    # Handle restart
    i_start = 1
    iter = 1

    if learning_restart:
        try:
            iter, i_start, candidate_idx, candidate_found_is, candidates_file = (
                restart_progress(start_iteration=load_progress())
            )
            version, backend = get_version()
            default_model = (
                "frozen_model.pth" if backend == "pytorch" else "frozen_model.pb"
            )
            latest_models = [getattr(config, "latest_model", default_model)]
        except Exception as e:
            SparcLog(f"Warning: Could not load restart progress: {e}")
            SparcLog("Starting from iteration 1")
            candidate_found_is = False
            candidates_file = ""
            candidate_idx = 0
            version, backend = get_version()
            default_model = (
                "frozen_model.pth" if backend == "pytorch" else "frozen_model.pb"
            )
            latest_models = [getattr(config, "latest_model", default_model)]
            iter = 1
            i_start = 1
    # SparcLog(f"{'Total AL iterations: ' + str(al_iter):^72}")

    # ---------------------------------------------------------------------------
    # Active Learning Loop
    # ---------------------------------------------------------------------------
    # Each iteration:
    # 1. Label candidates with DFT
    # 2. Retrain models with expanded dataset
    # 3. Run new ML-MD with updated models
    # 4. Query-by-Committee to find new candidates

    while candidate_found_is and iter <= al_iter:
        # SparcLog("="*80)
        # SparcLog(f"{'Starting Iteration ' + str(iter):^72}")
        # SparcLog("="*80)
        # --------------------------------------------------------------------------------
        SparcLog("")
        SparcLog("-" * 80)
        SparcLog(f"ACTIVE LEARNING ITERATION {iter}/{al_iter}".center(80))
        SparcLog("-" * 80)

        SparcLog(f"{'Iteration Directory':<30} iter_{iter:06d}")
        SparcLog("")

        # MD Configuration
        SparcLog("MD Configuration:")
        SparcLog(f"  {'Ensemble':<28} {config.mlip_setup.ensemble}")
        SparcLog(f"  {'Thermostat':<28} {config.mlip_setup.thermostat.type}")
        SparcLog(f"  {'Temperature':<28} {config.mlip_setup.temperature:.1f} K")

        # Temperature ramping info
        if config.mlip_setup.temp_end is not None:
            SparcLog(
                f"  {'Temperature Ramping':<28} {config.mlip_setup.temperature:.1f} K --> {config.mlip_setup.temp_end:.1f} K"
            )

            # Thermostat warning
            if config.mlip_setup.thermostat.get("type") == "Nose":
                SparcLog("")
                SparcLog(
                    "  WARNING: Temperature ramping with Nose-Hoover may be ineffective.",
                    level="WARNING",
                )
                SparcLog(
                    "           Recommended: Switch to Langevin thermostat.",
                    level="WARNING",
                )

        SparcLog(f"  {'MD Steps':<28} {config.mlip_setup.md_steps}")
        SparcLog(f"  {'Timestep':<28} {config.mlip_setup.timestep_fs:.2f} fs")
        SparcLog(f"  {'Log Frequency':<28} {config.mlip_setup.log_frequency}")
        SparcLog(f"{'Restart Exploration':<30} {'DISABLED (iteration 0)'}")
        SparcLog("")

        # Model deviation thresholds
        SparcLog("Model Deviation Criteria:")
        SparcLog(f"  {'Min Force Deviation':<28} {config.model_dev.f_min_dev} eV/Ang.")
        SparcLog(f"  {'Max Force Deviation':<28} {config.model_dev.f_max_dev} eV/Ang.")
        SparcLog("")

        # Training configuration
        SparcLog("Training Configuration:")
        SparcLog(f"  {'Number of Models':<28} {config.mlip_setup.num_models}")
        SparcLog(f"  {'Training Data Dir':<28} {config.mlip_setup.data_dir}")
        SparcLog("")

        SparcLog("=" * 80)
        SparcLog("")
        # -----------------------------------------------------------------------
        # Step 1: DFT Labeling of Candidates
        # -----------------------------------------------------------------------
        iter_structure = create_iteration_dirs(iter_num=iter)

        _dft_step = timer.start_step(iter, "dft", "00.dft", count=candidate_idx)
        # Before the loop, create calculator ONCE
        SparcLog("=" * 80)
        SparcLog(
            f"Iteration {iter} | Computing DFT energies/forces for {candidate_idx} candidates"
        )
        SparcLog("=" * 80)

        # Create DFT calculator once (it will print setup automatically)
        dft_calc = dft_calculator(config)  # Remove printscreen parameter

        # Read all candidate frames from the single trajectory file
        if not os.path.exists(candidates_file):
            SparcLog(
                f"Warning: Candidates file {candidates_file} not found, skipping DFT labelling..."
            )
        else:
            candidate_frames = read(candidates_file, index=":")

            # Label candidates with DFT
            for idx, NewCandidate in enumerate(
                candidate_frames[i_start - 1 :], start=i_start
            ):
                NewCandidate.calc = dft_calc  # Reuse the same calculator

                CalculateDFTEnergy(
                    idx=idx,
                    header=(idx == 1),
                    system=NewCandidate,
                    log_filename=f"Iter_{iter}_{config.output.log_file}",
                    trajfile=config.output.aimdtraj_file,
                    dir_name=iter_structure["dft_dir"],
                )

                save_progress(
                    {
                        "state": str(iter_structure["dft_dir"]),
                        "iteration": iter,
                        "candidate": candidate_idx,
                        "idx": idx,
                    }
                )
        timer.end_step(_dft_step)

        # -----------------------------------------------------------------------
        # Step 2: Model Retraining with Expanded Dataset
        # -----------------------------------------------------------------------
        SparcLog("=" * 80)
        SparcLog("Processing Data for MLIP Re-Training")
        SparcLog("=" * 80)

        _train_step = timer.start_step(iter, "train", "01.train")
        combined_traj = combine_trajectories(
            trajfilename=config.output.aimdtraj_file,
            current_iter=iter_structure["iter_num"],
        )

        get_data(
            ase_traj=combined_traj,
            dir_name=config.mlip_setup.data_dir,
            skip_min=0,
            skip_max=None,
            seed=config.mlip_setup.seed,
            train_ratio=config.mlip_setup.train_ratio,
        )

        finetune_config = config.finetune
        if finetune_config.enabled:
            finetune_training(
                finetune_config=finetune_config,
                datadir=datadir,
                atom_types=atom_types,
                training_dir=iter_structure["train_dir"],
                num_models=config.mlip_setup.num_models,
                input_file=config.mlip_setup.input_file,
            )
        else:
            deepmd_training(
                active_learning=True,
                training_dir=iter_structure["train_dir"],
                num_models=config.mlip_setup.num_models,
                input_file=config.mlip_setup.input_file,
                datadir=datadir,
                atom_types=atom_types,
            )
        timer.end_step(_train_step)

        # -----------------------------------------------------------------------
        # Step 3: ML-MD with Updated Models
        # -----------------------------------------------------------------------
        n_sample = get_num_samples(config)
        ensemble_name = get_ensemble_name(config.mlip_setup)

        SparcLog("=" * 80)
        SparcLog(f"Running {n_sample} ML-MD with updated models ({ensemble_name})")
        # Restart exploration status (only relevant here, not in iter 0)
        if config.mlip_setup.restart_exploration:
            SparcLog(
                f"Restart Exploration: ENABLED (frame: {config.mlip_setup.restart_frame})"
            )
        else:
            SparcLog("Restart Exploration: DISABLED (using original structure)")
        if config.mlip_setup.temp_end is not None:
            SparcLog(
                f"{'Temperature Ramping':<30} {config.mlip_setup.temperature:.1f} K --> {config.mlip_setup.temp_end:.1f} K"
            )
        SparcLog("=" * 80)

        # Use models trained in this iteration (mirrors pre-AL model detection)
        al_dp_path = iter_structure["train_dir"]
        al_pth = os.path.join(str(al_dp_path), "training_1", "frozen_model_1.pth")
        al_pb = os.path.join(str(al_dp_path), "training_1", "frozen_model_1.pb")
        if os.path.exists(al_pth):
            al_dp_model = "training_1/frozen_model_1.pth"
        elif os.path.exists(al_pb):
            al_dp_model = "training_1/frozen_model_1.pb"
        else:
            _, _backend = get_version()
            al_dp_model = (
                "training_1/frozen_model_1.pth"
                if _backend == "pytorch"
                else "training_1/frozen_model_1.pb"
            )

        # Invariant: al_dp_model must be a relative path. An absolute path would
        # cause setup_DeepPotential (deepmd.py:173) to silently discard al_dp_path,
        # loading a stale model from a previous iteration instead of the one just trained.
        assert not Path(al_dp_model).is_absolute(), (
            f"[BUG] AL iter {iter}: al_dp_model must be relative, got absolute: {al_dp_model}"
        )

        # Check if PLUMED is enabled
        plumed_config = config.mlip_setup.plumed
        umbrella_enabled = plumed_config.umbrella_sampling.enabled

        _mlmd_step = timer.start_step(iter, "mlmd", "02.dpmd")
        if umbrella_enabled:
            umbrella(
                config=config,
                us_dir=iter_structure,
                dp_path=al_dp_path,
                dp_model=al_dp_model,
            )
        else:
            # Cross-product: n_structures × multiple_run
            _multiple_run = config.mlip_setup.multiple_run or 1
            for i in range(n_sample):
                struct_idx = i // _multiple_run  # which structure file
                run_idx = i % _multiple_run  # which shoot for that structure
                struct_name = (
                    structure_file[struct_idx]
                    if isinstance(structure_file, list)
                    else structure_file
                )
                SparcLog(
                    f"ML-MD Run {i + 1}/{n_sample} | Iter {iter} | Structure: {struct_name} (s{struct_idx}) | Shoot {run_idx + 1}/{_multiple_run}"
                )
                dp_system = get_initial_structure(
                    iter=iter,
                    sample_idx=i,  # used as random seed — unique per run
                    struct_idx=struct_idx,
                    config=config,
                    structure_file=structure_file,
                    parent_dir=parent_dir,
                )

                dp_atoms, dp_calc = setup_DeepPotential(
                    atoms=dp_system, model_path=al_dp_path, model_name=al_dp_model
                )

                dyn_dp = initialize_thermostat(config.mlip_setup, dp_atoms)

                if plumed_config.enabled and iter >= plumed_config.start_iteration:
                    remove_backup_files(file_ext="bck.*")
                    dp_atoms.calc = setup_plumed_calc(
                        dp_calc,
                        dp_atoms,
                        config,
                        iter_structure["dpmd_dir"],
                        use_aimd=False,
                    )

                ExecuteMlpDynamics(
                    system=dp_atoms,
                    dyn=dyn_dp,
                    steps=config.mlip_setup.md_steps,
                    pace=config.mlip_setup.log_frequency,
                    log_filename=f"Iter{iter}_dpmd_s{struct_idx}_r{run_idx}.log",
                    trajfile=config.output.dptraj_file,
                    dir_name=iter_structure["dpmd_dir"],
                    distance_metrics=config.distance_metrics,
                    name=ensemble_name,
                    epot_threshold=getattr(config.mlip_setup, "epot_threshold", None),
                    temp_start=config.mlip_setup.temperature,
                    temp_end=config.mlip_setup.temp_end,
                    restart=config.mlip_setup.restart,
                )
        timer.end_step(_mlmd_step)
        # -----------------------------------------------------------------------
        # Step 4: Query-by-Committee for New Candidates
        # -----------------------------------------------------------------------
        _qbc_step = timer.start_step(iter, "qbc", "02.dpmd")
        candidate_found_is, candidates_file, candidate_idx, latest_models = (
            QueryByCommittee(
                trajfile=iter_structure["dpmd_dir"] / config.output.dptraj_file,
                model_path=iter_structure["train_dir"],
                num_models=config.mlip_setup.num_models,
                min_lim=config.model_dev.f_min_dev,
                max_lim=config.model_dev.f_max_dev,
                dpmd_data_path=iter_structure["dpmd_dir"],
                iteration=iter,
                rmsd_threshold=config.model_dev.rmsd_threshold,
                exclude_hydrogen=config.model_dev.exclude_hydrogen,
                deal_config=config.deal if config.deal.enabled else None,
                structure_file=(
                    config.general.structure_file[0]
                    if isinstance(config.general.structure_file, list)
                    else config.general.structure_file
                ),
                deepmd_input_json=config.mlip_setup.input_file,
            )
        )
        timer.end_step(_qbc_step)

        if candidate_idx < config.min_candidates:
            SparcLog("=" * 80)
            SparcLog(
                f"Candidates found ({candidate_idx}) below threshold ({config.min_candidates}). Stopping AL."
            )
            SparcLog("End of Active Learning Loop")
            SparcLog("=" * 80)
            break

        # Increment iteration counter
        iter += 1
        i_start = 1

        save_progress(
            {
                "state": str(iter_structure["dft_dir"]),
                "iteration": iter,
                "candidate": candidate_idx,
            }
        )

    # ---------------------------------------------------------------------------
    # Active Learning Complete
    # ---------------------------------------------------------------------------
    SparcLog("")
    SparcLog("-" * 80)
    SparcLog("ACTIVE LEARNING WORKFLOW COMPLETED".center(80))
    SparcLog("-" * 80)
    SparcLog(f"Total iterations : {iter - 1}")
    SparcLog(f"Final models saved in : iter_{iter - 1:06d}/01.train")
    SparcLog("=" * 80)
    timer.log_summary()


if __name__ == "__main__":
    main()

# ===================================================================================================#
# END OF FILE
# ===================================================================================================#
