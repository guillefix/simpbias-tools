# simpbias-tools

`KC_LZ.py` contains the python implementation of the version of Lempel-Ziv (LZ) complexity defined and used in Dingle et al. Input–output maps are strongly biased towards simple outputs (Nature comm, 2018).

To use, in your own script: `from KC_LZ import calc_KC`, and then evaluate the complexity of a string `x` doing `calc_KC(x)`. It is currently desinged for binary strings, but the code would trivially be extended to larger alphabets.

See also the sandbox for experiments in other variants of LZ complexity. In `complexities.py`, there are other complexity measures mostly for Boolean functions expressed as binary strings.
