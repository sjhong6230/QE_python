# QE-derived regression tests

`upstream/` contains official Quantum ESPRESSO inputs selected from QEF/q-e
commit `4a218b993489604a844db92fe85c747bd09b2442` and the required
norm-conserving pseudopotentials. Only cases supported by qepy-pw are included.

`reference/` contains qepy-pw output for every selected input. Regenerate all
references with:

```bash
python -m tests.qe_reference.check
```

Run the regression suite with:

```bash
python -m pytest -q
```
