

# Quantum Binary Classification with Qiskit v2.x (SamplerV2/EstimatorV2/IBM Quantum)

A comprehensive example of quantum binary classification using Qiskit v2.0.4+ (primitives V2 API), supporting both local simulators and real IBM Quantum hardware (with Qiskit Runtime).
Includes custom Variational Quantum Classifiers (VQC) with flexible circuits and full compatibility with the latest SamplerV2 and EstimatorV2 APIs.

## Features

* **Fully compatible with Qiskit v2.1.0 and above**
* **Primitives V2 API support:** `SamplerV2` and `EstimatorV2`
* **Run on both local simulators and IBM Quantum hardware**
* **Custom hardware-efficient quantum circuits**
* **End-to-end demo: data generation, training, and evaluation**
* **Examples for both local and cloud/hardware execution**
* **Robust transpilation for real hardware (native gate sets)**
* **Simple, modular design for educational or research use**

---

## Requirements

* Python 3.9+
* Qiskit v2.1.0 or later
  (see [Qiskit documentation](https://qiskit.org/documentation/))
* For IBM Quantum hardware execution:

  * `qiskit-ibm-runtime` (install with `pip install qiskit-ibm-runtime`)
  * An IBM Quantum account (sign up at [quantum.ibm.com](https://quantum.ibm.com))

**Other dependencies:**

* `numpy`
* `scipy`
* `scikit-learn`

---

## Installation

```bash
pip install qiskit==2.1.0 qiskit-aer scikit-learn scipy
# For IBM Quantum hardware:
pip install qiskit-ibm-runtime
```

---

## Usage

Check the "Running with EstimatorV2" section in the script output.

### 1. IBM Quantum Hardware (Qiskit Runtime)

To run on a real IBM Quantum backend:

1. Install the required runtime package:

   ```bash
   pip install qiskit-ibm-runtime
   ```
2. Create an account at [quantum.ibm.com](https://quantum.ibm.com).
3. Save your IBM Quantum Cloud API token:

   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(channel='ibm_cloud', token='YOUR_TOKEN')
   ```
4. Use the included script functions:

   * `run_on_ibm_quantum()`: Full version (SamplerV2, recommended for new hardware)
   * `run_on_ibm_quantum_direct()`: Runs circuits directly with `backend.run()`
   * `run_on_ibm_quantum_simple()`: Simple example (Sampler V1, for debugging)
   * All support `use_simulator=True` for cloud simulators

**Note:**

* Actual quantum hardware may have long queue times!
* Make sure to use `ibm_cloud` channel. The old `ibm_quantum` channel is deprecated after July 1, 2025.
* All circuits are transpiled to native gate sets for hardware.

---

## Circuit Design

* **Feature map:** Custom Z-feature map (or hardware-efficient RY-only layer)
* **Ansatz:** Real Amplitudes (or hardware-efficient RY+CX structure)
* No H gates in hardware-efficient circuits to ensure transpilation on real devices

### Visualize circuits:

```python
visualize_quantum_circuit()
```

---

## File Structure

* Main script: *your\_script.py* (rename as needed)
* All logic is contained in a single file for easy exploration and modification.

---

## Troubleshooting

* If you encounter `"Unable to retrieve instances."` or credential errors:

  * Make sure your IBM Quantum account is saved with `QiskitRuntimeService.save_account(channel='ibm_cloud', token='YOUR_TOKEN')`.
* If gates are not supported on hardware:

  * The script will transpile to the native basis (`rz`, `sx`, `x`, `cx`).
* For long queue times, test first on simulators (`use_simulator=True`).

---

## References

* [Qiskit Documentation](https://qiskit.org/documentation/)
* [Qiskit Runtime (IBM Cloud)](https://docs.quantum.ibm.com/run/systems)
* [Qiskit Tutorials](https://qiskit.org/documentation/tutorials/)

---

## License

This repository is released under the MIT License.

---

## Acknowledgments

* IBM Quantum and Qiskit development team
* Inspired by Qiskit community examples and Qiskit textbook

---

## Author

Amon Koike (A K)
2024-2025

