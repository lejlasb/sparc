# neb.py
################################################################
# Nudged Elastic Band (NEB) minimum energy path calculations.
#
# This module is deliberately CALCULATOR-AGNOSTIC: it accepts any
# ASE calculator factory, so the same code path serves DFT (VASP/CP2K)
# reference calculations and trained MLPs (DeepMD, MACE, etc.).
################################################################
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.mep import NEB, NEBTools
from ase.optimize import BFGS, FIRE, LBFGS

################################################################
# Local import
from sparc.src.utils.logger import SparcLog

################################################################

OPTIMIZERS = {
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "FIRE": FIRE,
}


# ===================================================================================================#
def build_band(
    initial,
    final,
    n_images=8,
    interpolation="idpp",
    climb=False,
    spring_constant=0.1,
    allow_shared_calculator=False,
):
    """
    Construct an interpolated NEB band between two endpoint structures.

    Parameters
    ----------
    initial : ase.Atoms
        Relaxed reactant-side endpoint.
    final : ase.Atoms
        Relaxed product-side endpoint.
    n_images : int, optional
        Number of INTERMEDIATE images (endpoints are added on top).
    interpolation : str, optional
        'idpp' (image-dependent pair potential) or 'linear'. IDPP is
        strongly recommended for molecular/reactive systems, since linear
        interpolation routinely produces overlapping atoms.
    climb : bool, optional
        Enable climbing-image NEB (CI-NEB) to converge onto the saddle.
    spring_constant : float, optional
        NEB spring constant in eV/Ang^2.
    allow_shared_calculator : bool, optional
        Only safe for cheap/stateless calculators. Keep False for DFT.

    Returns
    -------
    NEB
        Configured ASE NEB object with images interpolated.
    """
    if n_images < 1:
        raise ValueError("n_images must be >= 1 (number of intermediate images).")

    if len(initial) != len(final):
        raise ValueError(
            f"Endpoint atom counts differ: initial={len(initial)}, final={len(final)}. "
            "NEB requires a one-to-one atom mapping between endpoints."
        )

    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError(
            "Endpoint chemical symbol ORDER differs. NEB maps atoms by index, so the "
            "two endpoint files must list atoms in identical order."
        )

    images = [initial]
    images += [initial.copy() for _ in range(n_images)]
    images += [final]

    neb = NEB(
        images,
        k=spring_constant,
        climb=climb,
        allow_shared_calculator=allow_shared_calculator,
    )

    if interpolation.lower() == "idpp":
        neb.interpolate(method="idpp")
    elif interpolation.lower() == "linear":
        neb.interpolate()
    else:
        raise ValueError(
            f"Unknown interpolation '{interpolation}'. Use 'idpp' or 'linear'."
        )

    return neb


# ===================================================================================================#
def attach_calculators(neb, calculator_factory, workdir=None, include_endpoints=True):
    """
    Attach a SEPARATE calculator instance to every image.

    Each image must own its calculator. Sharing one instance across images
    causes silent result contamination for stateful calculators, and for
    file-based DFT codes (VASP/CP2K) the images would overwrite each other's
    scratch files.

    Parameters
    ----------
    neb : NEB
        Band produced by build_band().
    calculator_factory : callable
        Zero- or one-argument callable returning a FRESH ASE calculator.
        If it accepts an argument, the image index is passed so file-based
        calculators can be given a unique directory.
    workdir : str or Path, optional
        Parent directory for per-image scratch directories.
    include_endpoints : bool, optional
        Endpoints are held fixed during optimisation and so need no forces,
        but their ENERGIES are required to compute barriers. Structures read
        from formats that store energies (.traj) carry a SinglePointCalculator
        already; those read from plain .xyz do not. Attaching a calculator
        here makes the behaviour independent of input format.
    """
    import inspect

    from ase.calculators.calculator import FileIOCalculator

    n_args = len(inspect.signature(calculator_factory).parameters)

    if include_endpoints:
        targets = list(enumerate(neb.images))
    else:
        targets = list(enumerate(neb.images))[1:-1]

    for idx, image in targets:
        # Preserve an existing energy if the structure already carries one.
        # Structures read from .traj arrive with a SinglePointCalculator
        # holding the stored energy; recomputing it would be wasteful and,
        # for a different calculator, inconsistent.
        if idx in (0, len(neb.images) - 1) and image.calc is not None:
            try:
                image.get_potential_energy()
                continue
            except (RuntimeError, AttributeError, NotImplementedError) as exc:
                SparcLog(
                    f"[NEB] endpoint {idx} carries a calculator but no usable "
                    f"energy ({type(exc).__name__}); attaching a fresh one."
                )

        calc = calculator_factory(idx) if n_args >= 1 else calculator_factory()

        # Only file-based calculators (VASP, CP2K, Gaussian, ...) need their
        # own scratch directory. In-memory calculators (EMT, MACE, DeePMD)
        # inherit a `directory` attribute from ASE's base class but never
        # write to it, so creating one would just leave empty folders behind.
        if workdir is not None and isinstance(calc, FileIOCalculator):
            image_dir = Path(workdir) / f"image_{idx:02d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            calc.directory = str(image_dir)

        image.calc = calc

    return neb


# ===================================================================================================#
def run_neb(
    initial_file,
    final_file,
    calculator_factory,
    n_images=8,
    interpolation="idpp",
    climb=True,
    two_stage=True,
    spring_constant=0.1,
    fmax=0.05,
    fmax_loose=0.5,
    steps=500,
    optimizer="BFGS",
    workdir="neb",
    trajfile="neb.traj",
    logfile="neb.log",
    allow_shared_calculator=False,
):
    """
    Run a (climbing-image) NEB calculation between two endpoints.

    The default protocol is two-stage: a loose plain-NEB relaxation to get the
    band into the right basin, followed by a tight CI-NEB relaxation to converge
    the saddle. Switching on the climbing image before the band is roughly
    converged is a common way to converge onto the wrong saddle.

    Parameters
    ----------
    initial_file, final_file : str
        Paths to endpoint structures (any ASE-readable format).
    calculator_factory : callable
        Returns a fresh ASE calculator (see attach_calculators).
    n_images : int
        Number of intermediate images.
    climb : bool
        Run the climbing-image stage.
    two_stage : bool
        Run loose plain NEB before CI-NEB.
    fmax : float
        Force convergence criterion (eV/Ang) for the final stage.
    fmax_loose : float
        Force criterion for the preliminary stage.
    steps : int
        Max optimizer steps per stage.

    Returns
    -------
    dict
        Summary containing forward/reverse barriers, reaction energy,
        saddle image index, and output paths.
    """
    if optimizer.upper() not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Options: {list(OPTIMIZERS)}"
        )
    opt_class = OPTIMIZERS[optimizer.upper()]

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    initial = read(initial_file)
    final = read(final_file)

    SparcLog("=" * 60)
    SparcLog("                  NEB CALCULATION                  ")
    SparcLog("=" * 60)
    SparcLog(f"  Initial structure : {initial_file}")
    SparcLog(f"  Final structure   : {final_file}")
    SparcLog(f"  Intermediate imgs : {n_images}")
    SparcLog(f"  Interpolation     : {interpolation}")
    SparcLog(f"  Climbing image    : {climb}")
    SparcLog(f"  Spring constant   : {spring_constant} eV/Ang^2")
    SparcLog(f"  Optimizer         : {optimizer}")
    SparcLog("=" * 60 + "\n")

    neb = build_band(
        initial,
        final,
        n_images=n_images,
        interpolation=interpolation,
        climb=False,  # never start with climbing on
        spring_constant=spring_constant,
        allow_shared_calculator=allow_shared_calculator,
    )
    attach_calculators(neb, calculator_factory, workdir=workdir)

    traj_path = workdir / trajfile
    log_path = workdir / logfile

    if two_stage and climb:
        SparcLog(f"[NEB] Stage 1/2: plain NEB, fmax = {fmax_loose} eV/Ang")
        opt = opt_class(neb, trajectory=str(traj_path), logfile=str(log_path))
        opt.run(fmax=fmax_loose, steps=steps)

        SparcLog(f"[NEB] Stage 2/2: climbing-image NEB, fmax = {fmax} eV/Ang")
        neb.climb = True
        opt = opt_class(neb, trajectory=str(traj_path), logfile=str(log_path))
        opt.run(fmax=fmax, steps=steps)
    else:
        neb.climb = climb
        SparcLog(f"[NEB] Single-stage relaxation, fmax = {fmax} eV/Ang")
        opt = opt_class(neb, trajectory=str(traj_path), logfile=str(log_path))
        opt.run(fmax=fmax, steps=steps)

    converged = opt.converged()
    if not converged:
        SparcLog("[NEB] WARNING: optimizer did not reach fmax within step limit.")

    summary = analyse_band(neb.images, workdir=workdir)
    summary["converged"] = bool(converged)
    summary["trajectory"] = str(traj_path)

    write(workdir / "neb_final_path.xyz", neb.images)
    summary["path_file"] = str(workdir / "neb_final_path.xyz")

    saddle = neb.images[summary["saddle_index"]]
    write(workdir / "neb_saddle.xyz", saddle)
    summary["saddle_file"] = str(workdir / "neb_saddle.xyz")

    # Viewer-friendly exports (VMD / PyMOL).
    summary["visualization"] = write_visualization(
        neb.images, workdir=workdir, optimizer_traj=traj_path
    )

    SparcLog("=" * 60)
    SparcLog("                   NEB SUMMARY                     ")
    SparcLog("=" * 60)
    SparcLog(
        f"  Forward barrier   : {summary['forward_barrier_eV']:.4f} eV "
        f"({summary['forward_barrier_kcal']:.2f} kcal/mol)"
    )
    SparcLog(
        f"  Reverse barrier   : {summary['reverse_barrier_eV']:.4f} eV "
        f"({summary['reverse_barrier_kcal']:.2f} kcal/mol)"
    )
    SparcLog(
        f"  Reaction energy   : {summary['reaction_energy_eV']:.4f} eV "
        f"({summary['reaction_energy_kcal']:.2f} kcal/mol)"
    )
    SparcLog(
        f"  Saddle image      : {summary['saddle_index']} of {len(neb.images) - 1}"
    )
    SparcLog(f"  Converged         : {converged}")
    SparcLog("=" * 60 + "\n")

    if summary["saddle_index"] in (0, len(neb.images) - 1):
        SparcLog(
            "[NEB] WARNING: energy maximum sits on an ENDPOINT. The band likely does "
            "not bracket a barrier -- check that your endpoints are correct and relaxed."
        )

    return summary


# ===================================================================================================#
def analyse_band(images, workdir=None):
    """
    Extract barriers and reaction energy from a relaxed band.

    Returns
    -------
    dict
        Energies in both eV and kcal/mol.
    """
    EV_TO_KCAL = 23.060548

    energies = np.array([img.get_potential_energy() for img in images])
    rel = energies - energies[0]

    saddle_index = int(np.argmax(energies))
    forward = float(energies[saddle_index] - energies[0])
    reverse = float(energies[saddle_index] - energies[-1])
    reaction = float(energies[-1] - energies[0])

    summary = {
        "energies_eV": energies.tolist(),
        "relative_energies_eV": rel.tolist(),
        "relative_energies_kcal": (rel * EV_TO_KCAL).tolist(),
        "saddle_index": saddle_index,
        "forward_barrier_eV": forward,
        "reverse_barrier_eV": reverse,
        "reaction_energy_eV": reaction,
        "forward_barrier_kcal": forward * EV_TO_KCAL,
        "reverse_barrier_kcal": reverse * EV_TO_KCAL,
        "reaction_energy_kcal": reaction * EV_TO_KCAL,
    }

    if workdir is not None:
        out = Path(workdir) / "neb_energies.dat"
        with open(out, "w") as fh:
            fh.write("# image   E_rel(eV)   E_rel(kcal/mol)\n")
            fh.writelines(
                f"{i:6d}  {e_ev:12.6f}  {e_kcal:14.4f}\n"
                for i, (e_ev, e_kcal) in enumerate(
                    zip(
                        summary["relative_energies_eV"],
                        summary["relative_energies_kcal"],
                    )
                )
            )
        summary["energy_file"] = str(out)

    return summary


# ===================================================================================================#
def plot_band(images, filename="neb_profile.png", workdir=None):
    """
    Write a NEB energy profile plot using ASE's NEBTools fit.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nebtools = NEBTools(images)
    fig = nebtools.plot_band().get_figure()

    out = Path(workdir) / filename if workdir else Path(filename)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    SparcLog(f"[NEB] Energy profile written to {out}")

    return str(out)


# ===================================================================================================#
def write_visualization(images, workdir=None, prefix="neb_path", optimizer_traj=None):
    """
    Write the band in formats that VMD and PyMOL can animate.

    ASE's native .traj is a binary format that external viewers cannot read.
    This writes the same information as multi-frame XYZ and PDB, both of which
    load directly as animations:

        VMD    : vmd neb_path.xyz
        PyMOL  : load neb_path.xyz

    Parameters
    ----------
    images : list of ase.Atoms
        Relaxed band, including endpoints.
    workdir : str or Path, optional
        Output directory.
    prefix : str, optional
        Base filename.
    optimizer_traj : str or Path, optional
        Path to the optimizer .traj file. If given, the full relaxation
        history is also exported as a movie, which is useful for diagnosing
        a band that converged somewhere unexpected.

    Returns
    -------
    dict
        Paths of the files written.
    """
    base = Path(workdir) if workdir else Path(".")
    base.mkdir(parents=True, exist_ok=True)
    written = {}

    # --- final band -------------------------------------------------------
    xyz_path = base / f"{prefix}.xyz"
    write(xyz_path, images, format="extxyz")
    written["xyz"] = str(xyz_path)

    pdb_path = base / f"{prefix}.pdb"
    try:
        write(pdb_path, images, format="proteindatabank")
        written["pdb"] = str(pdb_path)
    except Exception as exc:  # noqa: BLE001
        SparcLog(f"[NEB] PDB export skipped: {exc}")

    # --- reaction-coordinate ordering -------------------------------------
    # Viewers step through frames in file order, which is already the band
    # order, so the animation runs reactant -> saddle -> product.
    SparcLog(f"[NEB] Visualization written: {xyz_path}")
    SparcLog("[NEB]   VMD   : vmd " + str(xyz_path))
    SparcLog("[NEB]   PyMOL : load " + str(xyz_path))

    # --- optional: full optimizer history ---------------------------------
    if optimizer_traj is not None and Path(optimizer_traj).exists():
        try:
            history = read(str(optimizer_traj), index=":")
            n_moving = len(images) - 2
            if n_moving > 0 and len(history) >= n_moving:
                n_steps = len(history) // n_moving
                movie = []
                for step in range(n_steps):
                    block = history[step * n_moving : (step + 1) * n_moving]
                    # Re-attach the fixed endpoints so each frame is a full band.
                    frame = [images[0]] + list(block) + [images[-1]]
                    movie.extend(frame)

                movie_path = base / f"{prefix}_optimization.xyz"
                write(movie_path, movie, format="extxyz")
                written["optimization_movie"] = str(movie_path)
                SparcLog(
                    f"[NEB] Optimization history written: {movie_path} "
                    f"({n_steps} steps x {len(images)} images)"
                )
        except Exception as exc:  # noqa: BLE001
            SparcLog(f"[NEB] Optimization movie skipped: {exc}")

    return written


# ===================================================================================================#
def run_neb_from_config(config):
    """
    Run a NEB calculation driven by a SPARC configuration object.

    This is the entry point used by the main workflow. It bridges the
    configured DFT engine to the calculator-agnostic NEB driver by wrapping
    `dft_calculator` in a factory, so that each image receives its own
    calculator instance.

    Parameters
    ----------
    config : SparcConfig
        Configuration containing a populated `neb` section and a
        `dft_calculator` section specifying the engine.

    Returns
    -------
    dict
        Summary as returned by run_neb().
    """
    from sparc.src.calculator import dft_calculator

    neb_cfg = config.neb

    def calculator_factory(image_index):
        # A fresh calculator per image: file-based engines would otherwise
        # overwrite one another's scratch files, and stateful calculators
        # would carry results between images.
        return dft_calculator(config, print_screen=False)

    SparcLog("")
    SparcLog("=" * 80)
    SparcLog(f"{'NUDGED ELASTIC BAND':^80}")
    SparcLog("=" * 80)
    SparcLog(f"{'DFT Engine':<30} {config.dft_calculator.engine}")
    SparcLog(f"{'Initial structure':<30} {neb_cfg.initial_structure}")
    SparcLog(f"{'Final structure':<30} {neb_cfg.final_structure}")
    SparcLog(f"{'Intermediate images':<30} {neb_cfg.n_images}")
    SparcLog(f"{'Interpolation':<30} {neb_cfg.interpolation}")
    SparcLog(f"{'Climbing image':<30} {neb_cfg.climb}")
    SparcLog(f"{'Optimizer':<30} {neb_cfg.optimizer}")
    SparcLog("=" * 80)
    SparcLog("")

    summary = run_neb(
        initial_file=neb_cfg.initial_structure,
        final_file=neb_cfg.final_structure,
        calculator_factory=calculator_factory,
        n_images=neb_cfg.n_images,
        interpolation=neb_cfg.interpolation,
        climb=neb_cfg.climb,
        two_stage=neb_cfg.two_stage,
        spring_constant=neb_cfg.spring_constant,
        fmax=neb_cfg.fmax,
        fmax_loose=neb_cfg.fmax_loose,
        steps=neb_cfg.steps,
        optimizer=neb_cfg.optimizer,
        workdir=neb_cfg.workdir,
        trajfile=neb_cfg.trajfile,
        logfile=neb_cfg.logfile,
    )

    try:
        images = read(str(summary["path_file"]), index=":")
        plot_band(images, workdir=neb_cfg.workdir)
    except Exception as exc:  # noqa: BLE001
        SparcLog(f"[NEB] Energy profile plot skipped: {exc}")

    return summary
