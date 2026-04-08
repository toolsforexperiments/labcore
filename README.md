# labcore

A Python toolkit for acquiring, processing, and analyzing data in a condensed matter / quantum information physics lab.

labcore is designed to complement the [QCodes](https://qcodes.github.io/Qcodes/) ecosystem — it sits alongside QCodes instruments and parameters, adding a flexible sweep framework, structured HDF5 storage, and analysis tools.

**[Get started in 15 minutes →](https://toolsforexperiments.github.io/labcore/first_steps/15_min_guide.html)**

---

## What's inside

- **Sweep framework** — compose parameter sweeps with `+` (sequential), `*` (zip), and `@` (nested) operators; decorate functions with `@recording` to produce structured records automatically
- **Structured data storage** — `DataDict` and `DDH5Writer` for writing and reading HDF5 data files; `find_data`, `load_as_xr`, and `load_as_df` for data discovery and loading
- **Fitting** — lmfit-based fitting framework with built-in fit functions (cosine, exponential, linear, exponentially decaying sine, and more) and xarray integration
- **Analysis base** — a lightweight framework for organizing, saving, and loading analysis artifacts (figures, datasets, parameters)

---

## Installation

labcore is not yet on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/toolsforexperiments/labcore.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/toolsforexperiments/labcore.git
pip install -e labcore/
```

Requires Python ≥ 3.11.

---

## Quick example

```python
from labcore.measurement import sweep_parameter, record_as, recording, dep, independent

# Define what to record
@recording(dep('signal', ['frequency']))
def measure(frequency):
    return {'signal': my_instrument.read(frequency)}

# Run a sweep and save to HDF5
from labcore.measurement.storage import run_and_save_sweep
folder, _ = run_and_save_sweep(
    sweep_parameter('frequency', range(100, 200)) @ measure,
    data_dir='./data',
    name='resonator_scan',
)

# Load the result as xarray
from labcore.data.datadict_storage import load_as_xr
ds = load_as_xr(folder)
```

See the [15-minute guide](https://toolsforexperiments.github.io/labcore/first_steps/15_min_guide.html) for a full walkthrough.

---

## Command-line tools

| Command | Description |
|---|---|
| `autoplot` | Live plotting server for monitoring running measurements |
| `reconstruct-data` | Reconstruct HDF5 files from safe-write temporary data |

---

## Development

```bash
git clone https://github.com/toolsforexperiments/labcore.git
cd labcore
uv sync --group dev
uv run pytest test/ -v
```

---

## License

MIT. See [LICENSE](LICENSE) for details.

## Authors

Wolfgang Pfaff and Marcos Frenkel.