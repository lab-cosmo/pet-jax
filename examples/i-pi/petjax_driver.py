"""i-PI PES adapter for pet-jax (``UPETCalculator``).

Loaded by i-PI's Python driver in ``custom`` mode -- this needs **no changes to
i-PI and no changes to the petjax package**:

    i-pi-py_driver -u -a petjax -m custom -P petjax_driver.py \
        -o checkpoint=<ckpt_dir>,template=<structure>

It subclasses i-PI's generic ``ASEDriver`` (``ipi/pes/_ase.py``), which already
handles unit conversion, voigt->3x3 stress, the virial, and "extras". All this
file does is build a ``UPETCalculator`` and hand it over.

Note: do NOT rename this file to ``petjax.py``. i-PI's custom loader registers
the module under its file stem in ``sys.modules``; a stem of ``petjax`` would
shadow the installed ``petjax`` package and break the import below.
"""

from ipi.pes._ase import ASEDriver

__DRIVER_NAME__ = "petjax"
__DRIVER_CLASS__ = "PETJAX_driver"


class PETJAX_driver(ASEDriver):
    """Run a pet-jax ``UPETCalculator`` as an i-PI force client.

    Init parameters (comma-separated after ``-o``):
        :param checkpoint: str, path to a pet-jax checkpoint directory
            (holds ``model.msgpack`` + ``metadata.yaml``).
        :param template: str, ASE-readable structure file. Only its atomic
            numbers and pbc/cell are used to initialise the calculator; i-PI
            streams the live positions and cell every step.
        :param dtype: str, ``"float32"`` (default) or ``"float64"``. fp32 is
            ~2x faster; promote to fp64 for precision-sensitive runs. PET-MAD
            weights ship as fp32, so fp64 promotes the cached params (fp64
            arithmetic, not extra trained precision).
        :param has_stress: bool, default True. Set ``has_stress=false`` for
            non-periodic systems -- the calculator only returns stress under
            PBC, so requesting it for an isolated molecule would error.

    Any further keyword (``has_energy``, ``has_forces``, ``verbose``) is
    forwarded to ``ASEDriver`` / ``Dummy_driver`` unchanged.
    """

    def __init__(self, checkpoint, template, dtype="float32", **kwargs):
        self.checkpoint = checkpoint
        self.dtype = dtype
        super().__init__(template=template, **kwargs)

    def check_parameters(self):
        # Reads the template (sets self.capabilities); leaves ase_calculator None.
        super().check_parameters()

        from petjax import UPETCalculator

        # Tie the calculator's stress flag to the capabilities ASEDriver built
        # from has_stress, so a non-periodic run (has_stress=false) doesn't ask
        # the calculator for a stress it won't produce.
        self.ase_calculator = UPETCalculator.from_checkpoint(
            self.checkpoint,
            default_dtype=self.dtype,
            stress="stress" in self.capabilities,
        )
