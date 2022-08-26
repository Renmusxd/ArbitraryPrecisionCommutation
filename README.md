# ArbitraryPrecisionCommutation

## Installation
1. Install rust on your system: https://www.rust-lang.org/learn/get-started
2. Prepare your [python environment](https://docs.python.org/3/tutorial/venv.html) by installing `maturin`, `numpy`, `wheel`, and upgrading `pip`:
   1. `> pip install --upgrade pip`
   2. `> pip install maturin numpy wheel`
3. Clone the repository:
   1. `> git clone git@github.com:Renmusxd/ArbitraryPrecisionCommutation.git`
4. Run `make` in the parent directory
   1. `> make`
5. Install the resulting wheel with pip
   1. `> pip install target/wheels/*`
   2. If multiple versions exist, select the one for your current python version.

See [notebook](https://github.com/Renmusxd/ArbitraryPrecisionCommutation/blob/main/jupyter/Example.ipynb) for example usage.

## Possible Errors:
The arbitrary precision float library I use is called [rug](https://docs.rs/rug/latest/rug/), it uses [GMP](https://gmplib.org/) and [MPFR](https://www.mpfr.org/). These should be on many systems already but if not could cause issues.
