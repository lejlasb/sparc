#!/usr/bin/python3
# src/utils/read_input.py


"""
Configuration schema and loading for SPARC package.
Provides type-safe configuration with automatic validation.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml

# Local imports
from sparc.src.utils.logger import SparcLog

################################################################
# Custom Exceptions
################################################################


class SparcException(Exception):
    """Base exception for all SPARC-related errors."""

    def __init__(self, message: str, context: dict = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ConfigurationError(SparcException):
    """Raised when configuration is invalid or missing."""

    pass


class ValidationError(SparcException):
    """Raised when data validation fails."""

    pass


################################################################
# Configuration Dataclasses
################################################################


@dataclass
class GeneralConfig:
    """General settings for SPARC workflow."""

    structure_file: Union[str, List[str]]

    def __post_init__(self):
        """Validate structure file(s) exist."""
        if isinstance(self.structure_file, list):
            # Validate each file in list
            for idx, sf in enumerate(self.structure_file):
                if not Path(sf).exists():
                    raise ValidationError(
                        f"Structure file {idx + 1} not found: {sf}",
                        context={"file": sf, "index": idx},
                    )
        else:
            # Validate single file
            if not Path(self.structure_file).exists():
                raise ValidationError(
                    f"Structure file not found: {self.structure_file}",
                    context={"file": self.structure_file},
                )


@dataclass
class DFTCalculatorConfig:
    """DFT calculator configuration."""

    engine: Literal["ORCA", "VASP", "CP2K", "xTB", "QE", "Gaussian"]
    template_file: str
    exe_command: Optional[str] = None

    def __post_init__(self):
        # Validate template file exists
        if not Path(self.template_file).exists():
            SparcLog(f"Warning: Template file not found: {self.template_file}")


@dataclass
class ThermostatConfig:
    """Thermostat configuration for MD simulations."""

    type: Literal["Nose", "Langevin"] = "Nose"
    tdamp: float = 2.0
    friction: Optional[float] = None  # For Langevin only

    def __post_init__(self):
        if self.type == "Langevin" and self.friction is None:
            self.friction = 0.01
            SparcLog(f"Setting default Langevin friction: {self.friction}")
        if self.tdamp <= 0:
            raise ValidationError(
                "Thermostat damping must be positive", context={"tdamp": self.tdamp}
            )


@dataclass
class ForceCorrectionConfig:
    """Bias force correction for PLUMED-biased AIMD trajectories.

    cv_atoms : list of lists of 1-based atom indices, one entry per CV.
        Order must match the ARG= order in the PLUMED input file.
        Example: [[1,4], [1,2]] for two distance CVs.
    cv_force_file : PLUMED DUMPFORCES output filename (inside dft_dir).
    cv_derivs_file : PLUMED DUMPDERIVATIVES output filename (inside dft_dir).

    IMPORTANT: PLUMED STRIDE for DUMPFORCES/DUMPDERIVATIVES must equal
    aimd_setup.log_frequency so that row i in the PLUMED files corresponds
    to frame i in the trajectory.
    """

    enabled: bool = False
    cv_force_file: str = "cv_force"
    cv_derivs_file: str = "cv_derivs"
    cv_atoms: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        if self.enabled:
            if not self.cv_atoms:
                raise ValidationError(
                    "force_correction.cv_atoms must list atom indices for each CV"
                )
            for k, atoms in enumerate(self.cv_atoms):
                if not atoms:
                    raise ValidationError(
                        f"CV {k} in force_correction.cv_atoms is empty"
                    )
                if any(a < 1 for a in atoms):
                    raise ValidationError(
                        f"CV {k} atom indices must be 1-based (all >= 1)"
                    )


@dataclass
class PlumedConfig:
    """PLUMED enhanced sampling configuration."""

    enabled: bool = False
    plumed_file: str = "plumed.dat"
    restart: bool = False
    kT: float = 0.02585
    force_correction: ForceCorrectionConfig = field(
        default_factory=ForceCorrectionConfig
    )

    def __post_init__(self):
        if self.enabled and not Path(self.plumed_file).exists():
            SparcLog(f"Warning: PLUMED file not found: {self.plumed_file}")
        if self.kT <= 0:
            raise ValidationError("kT must be positive")


@dataclass
class AIMDSetupConfig:
    """AIMD simulation setup (ASE-based DFT-MD)."""

    ensemble: Literal["NVT", "NVE", "NPT"] = "NVT"
    thermostat: ThermostatConfig = field(default_factory=ThermostatConfig)
    timestep_fs: float = 1.0
    temperature: float = 300.0
    temp_start: Optional[float] = None
    temp_end: Optional[float] = None
    steps: int = 0
    log_frequency: int = 1
    restart: bool = False
    plumed: PlumedConfig = field(default_factory=PlumedConfig)

    # NPT-specific parameters
    tau_t: Optional[float] = None
    tau_p: Optional[float] = None
    pressure: Optional[float] = None
    compressibility: Optional[float] = None

    def __post_init__(self):
        if self.timestep_fs <= 0 or self.timestep_fs > 5.0:
            raise ValidationError(
                "Timestep must be between 0 and 5 fs",
                context={"timestep_fs": self.timestep_fs},
            )
        if self.temperature <= 0:
            raise ValidationError("Temperature must be positive")
        if self.steps < 0:
            raise ValidationError("Number of steps must be non-negative")

        # Validate NPT-specific parameters
        if self.ensemble == "NPT":
            if self.tau_t is None or self.tau_p is None or self.pressure is None:
                raise ValidationError(
                    "NPT ensemble requires tau_t, tau_p, and pressure"
                )


@dataclass
class UmbrellaSamplingConfig:
    """Umbrella sampling configuration."""

    enabled: bool = False
    config_file: str = "umbrella_sampling.yaml"

    def __post_init__(self):
        if self.enabled and not Path(self.config_file).exists():
            SparcLog(f"Warning: Umbrella sampling config not found: {self.config_file}")


@dataclass
class MLIPPlumedConfig:
    """PLUMED configuration for MLIP MD simulations."""

    enabled: bool = False
    plumed_file: str = "plumed.dat"
    restart: bool = False
    kT: float = 0.02585
    start_iteration: int = 0
    umbrella_sampling: UmbrellaSamplingConfig = field(
        default_factory=UmbrellaSamplingConfig
    )

    def __post_init__(self):
        if self.enabled and not Path(self.plumed_file).exists():
            SparcLog(f"Warning: PLUMED file not found: {self.plumed_file}")
        if self.start_iteration < 0:
            raise ValidationError("plumed.start_iteration must be >= 0")


@dataclass
class ModelDeviationConfig:
    """Model deviation thresholds for active learning."""

    f_min_dev: float = 0.1
    f_max_dev: float = 0.8
    rmsd_threshold: float = 0.05
    exclude_hydrogen: bool = True

    def __post_init__(self):
        if self.f_min_dev < 0 or self.f_max_dev < 0:
            raise ValidationError("Deviation thresholds must be non-negative")
        if self.f_min_dev >= self.f_max_dev:
            raise ValidationError("f_min_dev must be less than f_max_dev")
        if self.rmsd_threshold < 0:
            raise ValidationError("rmsd_threshold must be non-negative")


@dataclass
class DEALConfig:
    """DEAL sparse GP diversity filter — replaces RMSD when enabled.

    Parameters
    ----------
    enabled : bool
        True → DEAL SGP filter. False → existing Kabsch RMSD filter (default).
    threshold : float
        SGP local predictive variance threshold. Frames above threshold are
        selected (novel chemical environment). Not comparable to f_min_dev —
        different units and descriptor space. Start at 0.01, tune from output.
    cutoff : float or None
        FLARE descriptor cutoff in Å. None = auto-read from mlip_setup.input_file
        (DeepMD rcut). Override only if auto-detection fails.
    nmax : int
        Radial basis functions for ACE B2 descriptor. Advanced — rarely change.
    lmax : int
        Angular channels for ACE B2 descriptor. Advanced — rarely change.
    """

    enabled: bool = False
    threshold: float = 0.01
    cutoff: Optional[float] = None
    nmax: int = 8
    lmax: int = 4

    def __post_init__(self):
        if self.threshold <= 0:
            raise ValidationError("deal.threshold must be positive")
        if self.cutoff is not None and self.cutoff <= 0:
            raise ValidationError("deal.cutoff must be positive")


@dataclass
class DistanceMetric:
    """Distance constraint for validation."""

    pair: List[int]
    min_distance: float
    max_distance: float

    def __post_init__(self):
        if len(self.pair) != 2:
            raise ValidationError("Distance pair must contain exactly 2 atom indices")
        if self.min_distance < 0:
            raise ValidationError("Minimum distance must be non-negative")
        if self.max_distance <= self.min_distance:
            raise ValidationError("Maximum distance must be greater than minimum")


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning universal ML models (DeePMD/DPA, MACE)."""

    enabled: bool = False
    model_type: Literal["deepmd", "mace"] = "deepmd"
    pretrained_model: str = "DPA3.pt"  # Pre-trained model path or MACE foundation name
    model_branch: Optional[str] = (
        "Omat24"  # Multi-task model branch (e.g., "Omat24", "Organic_Reactions")
    )
    input_file: Optional[str] = (
        None  # Fine-tune JSON config (if None, uses mlip_setup.input_file)
    )
    num_epochs: int = 100  # Fine-tuning epochs (MACE) or numb_steps override
    learning_rate: float = 0.001  # Starting learning rate
    batch_size: int = 4  # Batch size for fine-tuning
    device: str = "cpu"  # "cpu" or "cuda"


@dataclass
class MLIPSetupConfig:
    """MLIP (DeepMD) training and simulation configuration."""

    training: bool = False
    data_dir: str = "Training_Data"
    input_file: str = "input.json"
    skip_min: int = 0
    skip_max: Optional[int] = None
    train_ratio: float = 0.8
    num_models: int = 2
    MdSimulation: bool = False
    ensemble: Literal["NVT", "NVE", "NPT"] = "NVT"
    thermostat: ThermostatConfig = field(default_factory=ThermostatConfig)
    temperature: float = 300.0
    temp_start: Optional[float] = None
    temp_end: Optional[float] = None
    timestep_fs: float = 1.0
    md_steps: int = 2000
    multiple_run: int = 1
    log_frequency: int = 5
    epot_threshold: Optional[float] = 2.5  # eV
    seed: int = 42
    restart: bool = False  # Resume ML-MD from checkpoint
    restart_exploration: bool = False
    restart_frame: str = "candidates"
    plumed: MLIPPlumedConfig = field(default_factory=MLIPPlumedConfig)

    # NPT-specific parameters
    tau_t: Optional[float] = None
    tau_p: Optional[float] = None
    pressure: Optional[float] = None
    compressibility: Optional[float] = None

    def __post_init__(self):
        if self.num_models < 2:
            raise ValidationError("Number of models must be at least 2")
        if self.md_steps < 0:
            raise ValidationError("MD steps must be non-negative")
        if self.timestep_fs <= 0:
            raise ValidationError("Timestep must be positive")
        if self.temperature <= 0:
            raise ValidationError("Temperature must be positive")

        # Validate NPT-specific parameters
        if self.ensemble == "NPT":
            if self.tau_t is None or self.tau_p is None or self.pressure is None:
                raise ValidationError(
                    "NPT ensemble requires tau_t, tau_p, and pressure"
                )

        # data_dir is created lazily by the data processing step, not here


@dataclass
class NEBConfig:
    """Nudged elastic band configuration."""

    run: bool = False
    initial_structure: Optional[str] = None
    final_structure: Optional[str] = None
    n_images: int = 8
    interpolation: Literal["idpp", "linear"] = "idpp"
    climb: bool = True
    two_stage: bool = True
    spring_constant: float = 0.1
    fmax: float = 0.05
    fmax_loose: float = 0.5
    steps: int = 500
    optimizer: Literal["BFGS", "LBFGS", "FIRE"] = "BFGS"
    workdir: str = "neb"
    trajfile: str = "neb.traj"
    logfile: str = "neb.log"

    def __post_init__(self):
        if not self.run:
            return

        if not self.initial_structure or not self.final_structure:
            raise ValidationError(
                "NEB requires both 'initial_structure' and 'final_structure'"
            )

        for label, path in (
            ("initial_structure", self.initial_structure),
            ("final_structure", self.final_structure),
        ):
            if not Path(path).exists():
                raise ValidationError(
                    f"NEB {label} not found: {path}", context={"file": path}
                )

        if self.n_images < 1:
            raise ValidationError("NEB n_images must be at least 1")
        if self.spring_constant <= 0:
            raise ValidationError("NEB spring_constant must be positive")
        if self.fmax <= 0:
            raise ValidationError("NEB fmax must be positive")
        if self.two_stage and self.fmax_loose <= self.fmax:
            raise ValidationError(
                "NEB fmax_loose must exceed fmax; the first stage is a coarse "
                "relaxation preceding the tight climbing-image stage"
            )


@dataclass
class OutputConfig:
    """Output file settings."""

    log_file: str = "AseMD.log"
    xyz_file: str = "AseTraj.xyz"
    aimdtraj_file: str = "AseMD.traj"
    dptraj_file: str = "dpmd.traj"

    def __post_init__(self):
        # Create output directories
        for file_path in [
            self.log_file,
            self.xyz_file,
            self.aimdtraj_file,
            self.dptraj_file,
        ]:
            parent_dir = Path(file_path).parent
            if parent_dir != Path("."):
                parent_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SparcConfig:
    """Complete SPARC configuration."""

    general: GeneralConfig = field(default_factory=GeneralConfig)
    dft_calculator: DFTCalculatorConfig = field(default_factory=DFTCalculatorConfig)
    aimd_setup: AIMDSetupConfig = field(default_factory=AIMDSetupConfig)
    mlip_setup: MLIPSetupConfig = field(default_factory=MLIPSetupConfig)

    # Flattened active learning fields (no nesting!)
    active_learning: bool = False
    learning_restart: bool = False
    latest_model: str = None
    iteration: int = 10
    min_candidates: int = 1
    model_dev: ModelDeviationConfig = field(default_factory=ModelDeviationConfig)

    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    deal: DEALConfig = field(default_factory=DEALConfig)
    distance_metrics: List[DistanceMetric] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    neb: NEBConfig = field(default_factory=NEBConfig)
    cp2k: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.iteration < 1:
            raise ValidationError("Number of iterations must be at least 1")
        if self.min_candidates < 1:
            raise ValidationError("min_candidates must be at least 1")

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "SparcConfig":
        """Load configuration from YAML file with validation."""
        if not Path(yaml_file).exists():
            raise ConfigurationError(f"Configuration file not found: {yaml_file}")

        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML file: {e}")

        try:
            # Parse general config
            general = GeneralConfig(**data.get("general", {}))

            # Parse DFT calculator
            dft_calc = DFTCalculatorConfig(**data.get("dft_calculator", {}))

            # Parse AIMD setup with thermostat and plumed
            aimd_data = data.get("aimd_setup", {}).copy()

            # Parse thermostat
            thermostat_data = aimd_data.pop("thermostat", {})
            aimd_thermostat = ThermostatConfig(**thermostat_data)

            # Parse PLUMED for AIMD
            plumed_data = aimd_data.pop("plumed", {}).copy()
            fc_data = plumed_data.pop("force_correction", {}).copy()
            fc_config = ForceCorrectionConfig(**fc_data)
            aimd_plumed = PlumedConfig(force_correction=fc_config, **plumed_data)

            aimd_setup = AIMDSetupConfig(
                thermostat=aimd_thermostat, plumed=aimd_plumed, **aimd_data
            )

            # Parse MLIP setup
            mlip_data = data.get("mlip_setup", {}).copy()

            # Parse MLIP thermostat
            mlip_thermostat_data = mlip_data.pop("thermostat", {})
            mlip_thermostat = ThermostatConfig(**mlip_thermostat_data)

            # Parse MLIP PLUMED
            mlip_plumed_data = mlip_data.pop("plumed", {})

            # Parse umbrella sampling within MLIP PLUMED
            umbrella_data = mlip_plumed_data.pop("umbrella_sampling", {})
            umbrella = UmbrellaSamplingConfig(**umbrella_data)

            mlip_plumed = MLIPPlumedConfig(
                umbrella_sampling=umbrella, **mlip_plumed_data
            )

            mlip_setup = MLIPSetupConfig(
                thermostat=mlip_thermostat, plumed=mlip_plumed, **mlip_data
            )

            # Parse fine-tuning config (top-level section)
            finetune_data = data.get("finetune", {})
            finetune = FineTuneConfig(**finetune_data)

            # Parse DEAL config
            deal_data = data.get("deal", {})
            deal = DEALConfig(**deal_data)

            # Parse model_dev directly
            model_dev_data = data.get("model_dev", {})
            model_dev = ModelDeviationConfig(**model_dev_data)

            # Parse distance metrics
            distance_metrics_data = data.get("distance_metrics", [])
            distance_metrics = [DistanceMetric(**dm) for dm in distance_metrics_data]

            # Parse output config
            output = OutputConfig(**data.get("output", {}))

            # Parse NEB config
            neb = NEBConfig(**data.get("neb", {}))

            return cls(
                general=general,
                dft_calculator=dft_calc,
                aimd_setup=aimd_setup,
                mlip_setup=mlip_setup,
                active_learning=data.get("active_learning", False),
                learning_restart=data.get("learning_restart", False),
                latest_model=data.get("latest_model", None),
                iteration=data.get("iteration", 10),
                min_candidates=data.get("min_candidates", 1),
                model_dev=model_dev,
                finetune=finetune,
                deal=deal,
                distance_metrics=distance_metrics,
                output=output,
                neb=neb,
                cp2k=data.get("cp2k", {}),
            )

        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration structure: {e}")
        except Exception as e:
            raise ConfigurationError(f"Configuration parsing failed: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, yaml_file: str):
        """Save configuration to YAML file."""
        with open(yaml_file, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate_environment(self) -> bool:
        """Validate that required files exist."""
        validations = []

        # # Check structure file
        # if not Path(self.general.structure_file).exists():
        #     validations.append(f"Structure file missing: {self.general.structure_file}")
        # Check structure file(s)
        if isinstance(self.general.structure_file, list):
            for sf in self.general.structure_file:
                if not Path(sf).exists():
                    validations.append(f"Structure file missing: {sf}")
        else:
            if not Path(self.general.structure_file).exists():
                validations.append(
                    f"Structure file missing: {self.general.structure_file}"
                )

        # Check template file
        if not Path(self.dft_calculator.template_file).exists():
            validations.append(
                f"Template file missing: {self.dft_calculator.template_file}"
            )

        # Check AIMD PLUMED if enabled
        if self.aimd_setup.plumed.enabled:
            if not Path(self.aimd_setup.plumed.plumed_file).exists():
                validations.append(
                    f"AIMD PLUMED file missing: {self.aimd_setup.plumed.plumed_file}"
                )

        # Check MLIP PLUMED if enabled
        if self.mlip_setup.plumed.enabled:
            if not Path(self.mlip_setup.plumed.plumed_file).exists():
                validations.append(
                    f"MLIP PLUMED file missing: {self.mlip_setup.plumed.plumed_file}"
                )

        if validations:
            for warning in validations:
                SparcLog(f"Warning: {warning}")
            return False

        return True


################################################################
# Configuration Loading Functions
################################################################


def load_config(input_file: str = "input.yaml") -> SparcConfig:
    """
    Load and validate SPARC configuration from YAML file.

    Args:
        input_file: Path to YAML configuration file

    Returns:
        SparcConfig object with validated configuration

    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        config = SparcConfig.from_yaml(input_file)

        # Simple logging like original code
        # SparcLog("========================================================================")
        # SparcLog("  Input Configurations (- PLEASE CHECK SPARC INPUTS CAREFULLY! -)")
        # SparcLog("========================================================================\n")
        # SparcLog(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))
        # Header
        # SparcLog("=" * 80)
        SparcLog(
            "SPARC: Smart Potential with Atomistic Rare Events & Continuous Learning"
        )
        SparcLog("Version: 0.2")
        print("")
        SparcLog("=" * 80)
        SparcLog(f"Working directory: {os.getcwd()}")
        SparcLog("Configuration file: input.yaml")
        SparcLog("")

        # Validate environment
        config.validate_environment()

        return config

    except Exception as e:
        SparcLog(f" Configuration loading failed: {e}")
        raise ConfigurationError(f"Failed to load configuration: {e}")


def get_legacy_dict(config: SparcConfig) -> Dict[str, Any]:
    """
    Convert SparcConfig to legacy dictionary format for backward compatibility.

    Args:
        config: SparcConfig object

    Returns:
        Dictionary in original format
    """
    return config.to_dict()
