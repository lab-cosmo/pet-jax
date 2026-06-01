# Running `pet-jax` with i-PI

[i-PI](https://ipi-code.org) is a universal force engine: it runs the dynamics (MD, PIMD, geometry optimisation, ...) as a server and asks an external *client* for energies and forces over a socket. This example drives a `pet-jax` `UPETCalculator` as that client.

It needs **no code in `pet-jax` and no patch to i-PI**: i-PI ships a generic `ASEDriver` and a `custom` plugin loader, so the entire integration is the ~30-line adapter [`petjax_driver.py`](petjax_driver.py), which just builds a `UPETCalculator` and hands it to i-PI.

Files here:

| file                | role                                                        |
| ------------------- | ----------------------------------------------------------- |
| `petjax_driver.py`  | the adapter (i-PI `ASEDriver` subclass → `UPETCalculator`)  |
| `input.xml`         | i-PI input: 10 steps of NVT on bulk Si over a unix socket   |
| `si-diamond.xyz`    | 8-atom diamond-silicon template (periodic, so stress is on) |

## Prerequisites

**`pet-jax` and i-PI in the same environment.** The client process imports both, so whichever environment you use (pip, uv, conda, ...) needs them together. `pet-jax` isn't on PyPI yet — see the top-level [Installation](../../README.md#installation) — but i-PI is:

```bash
pip install "git+https://github.com/lab-cosmo/pet-jax"   # or a local checkout
pip install ipi
```

**A converted checkpoint.** Conversion needs the extra `convert` dependencies; inference does not:

```bash
pip install "pet-jax[convert] @ git+https://github.com/lab-cosmo/pet-jax"
petjax-convert pet-mad-xs --out pet-mad-xs   # downloads + converts PET-MAD-xs into ./pet-mad-xs
```

`petjax-convert` also accepts a URL or a local `.ckpt`; see the top-level README. Any converted checkpoint directory works below — just point `checkpoint=` at it.

## Run it (de-facto integration test)

The server and client are two processes talking over `/tmp/ipi_petjax`. Run the server in the background, give it a moment to create the socket, then start the client. From **this directory** (`examples/i-pi/`), with both commands using the environment you installed into above:

```bash
# terminal 1 — i-PI server (dynamics)
i-pi input.xml > ipi.log 2>&1 &

sleep 5   # let the server bind the socket

# terminal 2 — pet-jax force client
i-pi-py_driver -u -a petjax -m custom -P petjax_driver.py \
    -o checkpoint=/path/to/pet-mad-xs,template=si-diamond.xyz
```

(Two real terminals work too — just drop the `&` and the redirect from the server command.)

### What success looks like

The run is short and self-checking. Expect, within a few seconds:

1. **The client connects and the run completes.** i-PI logs a socket connection, advances `step` to 10, and then sends `EXIT`; the client prints `Received exit message from i-PI. Bye bye!` and returns 0. The first step is slow (JAX/XLA compile), the rest are fast.

2. **A finite, sane `petjax-si.out`.** One row per step:

   ```bash
   cat petjax-si.out
   ```

   Sanity bar for the 8-atom Si cell with PET-MAD (reference run):
   - `potential` is finite and around **−5.9 eV/atom** (≈ −47 eV total);
   - `temperature` fluctuates around the 300 K setpoint (small cell → large fluctuations are normal);
   - `conserved` stays finite — for this NVT run it sits near **−46.77 eV** and drifts only ~1e-4 eV over the 10 steps;
   - `pressure_md` and `volume` are finite (proves the **virial/stress** path — this is what `pbc` + `has_stress` exercise).

3. **Finite forces.** `petjax-si.forces_0.extxyz` holds one frame per step (11 frames incl. step 0) with finite force vectors.

If all three hold, the full i-PI ↔ pet-jax path — socket protocol, unit conversion, neighbourlist rebuild across changing geometries, forces, and the stress/virial — is working end to end.

### Troubleshooting

- **`i-pi` / `i-pi-py_driver: command not found`:** the environment with i-PI isn't active. Activate it, or call the executables by their full path.
- **`ModuleNotFoundError: petjax` in the client:** `pet-jax` and i-PI are in different environments. Install both into the one that runs `i-pi-py_driver`.
- **Client can't connect / "connection refused":** the server wasn't up yet. Increase the `sleep`, and confirm `<address>` in `input.xml` (`petjax`) matches `-a petjax`.
- **Stale socket after a crash:** `rm -f /tmp/ipi_petjax` before re-running.

## Precision: fp32 (default) → fp64

The adapter defaults to **fp32** (fast; matches the shipped PET-MAD weights). For precision-sensitive runs, promote to fp64 by adding `dtype=float64` to the client's `-o` list — nothing else changes:

```bash
i-pi-py_driver -u -a petjax -m custom -P petjax_driver.py \
    -o checkpoint=/path/to/pet-mad-xs,template=si-diamond.xyz,dtype=float64
```

fp64 is ~2× slower and, since the weights are fp32, buys you fp64 *arithmetic* (accumulation, the autodiff graph), not extra trained precision.

## Non-periodic systems

The calculator only returns stress under PBC. For an isolated molecule, give the client `has_stress=false` (so it doesn't request a stress that won't exist) and use a `template` whose `pbc` is false:

```bash
i-pi-py_driver -u -a petjax -m custom -P petjax_driver.py \
    -o checkpoint=/path/to/pet-mad-xs,template=molecule.xyz,has_stress=false
```

and drop `pressure_md` from the `<properties>` list in your input.

## Other dynamics

`input.xml` is plain i-PI — swap the `<motion>`/`<dynamics>` block for `nve`, geometry optimisation (`<optimize>`), `npt`, replica exchange, or PIMD (`nbeads > 1`) without touching the adapter. The same client serves any of them.
