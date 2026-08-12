# QE-derived regression tests

`upstream/` contains official Quantum ESPRESSO inputs selected from QEF/q-e
commit `4a218b993489604a844db92fe85c747bd09b2442`. Only cases supported by
qepy-pw are included.

The active H, Si, and Al fixtures are the unmodified UPF files from the
PseudoDojo `NC SR (ONCVPSP v0.5) / PBE / standard` table. Their official
source URLs, download date, and SHA-256 digests are pinned in `manifest.json`.
The source archive used for the migration had SHA-256
`455e00dac71aa13ade7508fbc863cb52f2333e28d556fe22d374a1889aad6930`.
See the embedded `PP_INFO` blocks and the repository's
`THIRD_PARTY_NOTICES.md` for license and citation information.

`reference/` contains qepy-pw output for every selected input. Regenerate all
references with:

```bash
python -m tests.qe_reference.check
```

Changing a pseudopotential is a reference-set migration: update the pinned
digest, every affected QE input, the saved-wavefunction fixture, and all
Python output snapshots in the same change series.

Run the regression suite with:

```bash
python -m pytest -q
```
