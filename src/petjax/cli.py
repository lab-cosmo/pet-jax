"""``petjax-convert`` console entry point.

Convert a metatrain PET ``.ckpt`` into pet-jax's ``model.msgpack`` +
``metadata.yaml`` layout. The source can be:

    * a known PET-MAD shortcut (``pet-mad-xs``, ``pet-mad-s``) — fetched from
      Hugging Face;
    * a URL to a ``.ckpt`` file — downloaded then converted;
    * a local path to a ``.ckpt`` file — converted in place.

Idempotent: an existing ``model.msgpack`` in ``--out`` short-circuits the
conversion; the downloaded ``.ckpt`` (if any) is cached under ``--cache``.

The conversion needs ``torch`` + ``metatomic-torch`` + ``metatrain`` (declared
in the ``convert`` extra). Either ``pip install pet-jax[convert] metatrain``
or, for an ephemeral environment, run via uv:

    uv run --with metatrain --with metatomic-torch \\
        petjax-convert pet-mad-xs --out checkpoints/pet-mad-xs
"""

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# -- PET-MAD shortcuts (the publicly available checkpoints on Hugging Face) --

HF_BASE = "https://huggingface.co/lab-cosmo/upet/resolve/main/models"

PET_MAD_SHORTCUTS = {
    "pet-mad-xs": f"{HF_BASE}/pet-mad-xs-v1.5.0.ckpt",
    "pet-mad-s": f"{HF_BASE}/pet-mad-s-v1.5.0.ckpt",
}


def _resolve_source(source):
    """Map the user-supplied source to ``(kind, value)`` where ``kind`` is
    ``"shortcut"``, ``"url"``, or ``"path"``."""
    if source in PET_MAD_SHORTCUTS:
        return "shortcut", PET_MAD_SHORTCUTS[source]
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in ("http", "https"):
        return "url", source
    return "path", Path(source)


def _download(url, dest):
    if dest.exists():
        print(f"[fetch] {dest.name} already cached, skipping download")
        return
    print(f"[fetch] downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"[fetch] saved {dest}")


def convert_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="petjax-convert",
        description=(
            "Convert a metatrain PET .ckpt into pet-jax's Flax msgpack layout. "
            "SOURCE may be a PET-MAD shortcut (pet-mad-xs, pet-mad-s), an http(s) "
            "URL to a .ckpt, or a local path."
        ),
    )
    parser.add_argument(
        "source",
        help="pet-mad-xs / pet-mad-s, an http(s) URL, or a local .ckpt path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: derived from the source's basename)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("checkpoints/.cache"),
        help="where to cache downloaded .ckpt files (ignored for local paths)",
    )
    args = parser.parse_args(argv)

    kind, value = _resolve_source(args.source)

    if kind == "path":
        ckpt = value
        default_out_name = ckpt.stem
        if not ckpt.exists():
            print(f"error: {ckpt} does not exist", file=sys.stderr)
            return 1
    else:
        url = value
        args.cache.mkdir(parents=True, exist_ok=True)
        ckpt_name = Path(urllib.parse.urlparse(url).path).name or "model.ckpt"
        ckpt = args.cache / ckpt_name
        default_out_name = args.source if kind == "shortcut" else Path(ckpt_name).stem
        try:
            _download(url, ckpt)
        except Exception as e:  # noqa: BLE001
            print(f"error: download failed: {e}", file=sys.stderr)
            return 1

    out_dir = args.out or Path(f"checkpoints/{default_out_name}")

    if (out_dir / "model.msgpack").exists():
        print(f"[convert] {out_dir}/model.msgpack already present, done")
        return 0

    from .convert import convert_checkpoint

    convert_checkpoint(ckpt, out_dir)
    print(f"[done] pet-jax checkpoint at {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(convert_main())
