# Test suite structure

The suite is organized by responsibility so a failure points to the smallest
relevant layer:

- `unit/core`: numerical kernels that do not require a complete PW run.
- `unit/io`: input validation, QE formatting, and working-file policies.
- `unit/pw`: PW-specific state construction and spin-polarized behavior.
- `postprocessing`: `bands.x`, `dos.x`, `pp.x`, `projwfc.x`, and plot tools.
- `integration`: native FFT/MPI behavior and comparisons with QE references.
- `qe_reference`: immutable upstream inputs, pseudopotentials, and outputs.

Unit tests should use analytic identities or finite differences where
possible. Integration tests should cover only behavior that cannot be
isolated without a complete calculation; this avoids repeating expensive SCF
runs for properties already established by a numerical-kernel test.
