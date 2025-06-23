# Quantum Binary Classification with Enhanced VQC Implementation

A comprehensive implementation of **Variational Quantum Classifier (VQC)** using Qiskit v2.1.0+ with SamplerV2/EstimatorV2 primitives API. This implementation supports both local simulators and real IBM Quantum hardware with automatic result collection, visualization, and comprehensive file output capabilities.

## 🌟 Features

### Core Quantum Machine Learning
- **Full Qiskit v2.1.0+ compatibility** with SamplerV2/EstimatorV2 primitives
- **Custom VQC implementation** with flexible circuit architectures
- **Hardware-efficient quantum circuits** optimized for NISQ devices
- **Dual execution modes**: Local simulators and IBM Quantum hardware
- **Advanced error handling** with automatic retry mechanisms

### Enhanced Results Management
- **Comprehensive experiment tracking** with metadata collection
- **Automatic file generation**: JSON, CSV, PNG, and text reports
- **Circuit visualization** with automatic saving capabilities
- **Real-time result monitoring** and progress tracking
- **Detailed summary reports** for analysis and documentation

### Advanced Features
- **IBM Quantum Open Plan support** with job mode execution
- **Transpilation optimization** for real quantum hardware
- **Custom feature maps** and parameterized ansatz circuits
- **Robust data preprocessing** with StandardScaler integration
- **Cross-entropy loss optimization** with COBYLA optimizer

---

## 🔬 Theoretical Background

### Variational Quantum Classifier (VQC)

The Variational Quantum Classifier (VQC) is a hybrid quantum-classical algorithm that combines quantum circuits with classical optimization techniques for supervised learning tasks. VQAs (Variational Quantum Algorithms) leverage both classical and quantum computing resources to find approximate solutions to optimization and machine learning problems.

#### Key Components

1. **Feature Map (Data Encoding)**
   - Maps classical data into quantum states through parameterized quantum circuits
   - Mathematical map that embeds classical data into a quantum state using variational quantum circuits whose parameters depend on the input data
   - Implementation: Custom Z-feature maps and hardware-efficient RY rotations

2. **Ansatz (Variational Circuit)**
   - Parameterized quantum circuit analogous to layers in classical neural networks with tunable parameters optimized to minimize objective functions
   - RealAmplitudes circuit consisting of alternating layers of Y rotations and CX entanglements, preparing quantum states with only real amplitudes
   - Implementation: RealAmplitudes with configurable repetitions and custom hardware-efficient designs

3. **Measurement and Classification**
   - Quantum state measurement provides probability distributions
   - Classical post-processing interprets measurements as class predictions
   - Measured bitstrings are interpreted as classifier output with support for binary and multi-class problems

### Quantum Advantage in Machine Learning

Variational quantum classifiers can solve classification problems based on PROMISEBQP-complete problems, implying quantum advantage for classification problems that cannot be classically solved in polynomial time. Quantum kernel-based classifiers are practical techniques for hyper-linear classification of complex data, with potential sub-quadratic runtime complexity.

---

## 📋 Requirements

### Core Dependencies
```bash
# Quantum computing framework
qiskit>=2.1.0
qiskit-aer>=0.13.0

# IBM Quantum hardware access
qiskit-ibm-runtime>=0.19.0

# Scientific computing
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0

# Visualization (optional but recommended)
matplotlib>=3.7.0
```

### System Requirements
- **Python**: 3.9+ (recommended: 3.11+)
- **Memory**: 4GB+ RAM for local simulations
- **Storage**: 1GB+ for result files and circuit diagrams
- **IBM Quantum Account**: For hardware execution ([Register here](https://quantum.ibm.com))

---

## 🚀 Installation

### Quick Setup
```bash
# Install core quantum computing packages
pip install qiskit>=2.1.0 qiskit-aer qiskit-ibm-runtime

# Install scientific computing dependencies
pip install numpy scipy scikit-learn pandas matplotlib

# Verify installation
python -c "import qiskit; print(f'Qiskit version: {qiskit.__version__}')"
```

### IBM Quantum Setup
```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your IBM Quantum credentials (one-time setup)
QiskitRuntimeService.save_account(
    channel='ibm_cloud',  # Use 'ibm_cloud' for new accounts
    token='YOUR_IBM_QUANTUM_TOKEN'
)
```

---

## 💻 Usage

### Basic Local Simulation

```python
# Run local quantum simulation
python main.py

# Or import and use individual functions
from your_script import run_local_simulation, QuantumClassifier

# Execute local simulation with custom parameters
local_result = run_local_simulation(data_size=80, max_iter=25)
```

### IBM Quantum Hardware Execution

```python
# Test backend connectivity
from your_script import test_quantum_backend
success = test_quantum_backend(token='YOUR_TOKEN', channel='ibm_cloud')

# Run on IBM Quantum hardware
from your_script import run_on_ibm_quantum_fixed
quantum_result = run_on_ibm_quantum_fixed(
    token='YOUR_TOKEN',
    use_simulator=False,  # Set to True for cloud simulators
    channel='ibm_cloud'
)
```

### Custom VQC Implementation

```python
from your_script import QuantumClassifier, generate_sample_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate or load your data
X, y = generate_sample_data(n_samples=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train quantum classifier
qc = QuantumClassifier(num_features=2, reps=2)
qc.fit(X_train_scaled, y_train, max_iter=30)

# Evaluate performance
train_accuracy = qc.score(X_train_scaled, y_train)
test_accuracy = qc.score(X_test_scaled, y_test)

print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Testing accuracy: {test_accuracy:.3f}")
```

---

## 🗂️ File Structure and Outputs

### Automatic Result Generation

The implementation automatically generates comprehensive result files:

```
quantum_results/
├── training_results_YYYYMMDD_HHMMSS.json    # Complete experimental data
├── training_summary_YYYYMMDD_HHMMSS.csv     # Summary statistics
├── experiment_summary_YYYYMMDD_HHMMSS.txt   # Human-readable report
└── circuits/                                 # Circuit visualizations
    ├── Feature_Map_YYYYMMDD_HHMMSS.png
    ├── Ansatz_YYYYMMDD_HHMMSS.png
    ├── Complete_Circuit_YYYYMMDD_HHMMSS.png
    └── *.txt files with circuit descriptions
```

### Result File Contents

#### JSON Output
```json
{
  "timestamp": "2025-01-XX",
  "qiskit_version": "2.1.0",
  "experiments": [
    {
      "name": "Local_Simulation",
      "backend": "AerSimulator",
      "training_accuracy": 0.875,
      "testing_accuracy": 0.833,
      "execution_time": 12.34,
      "shots": 1024,
      "success": true
    }
  ],
  "circuit_diagrams": {...},
  "summaries": {...}
}
```

#### CSV Output
| experiment | backend | training_accuracy | testing_accuracy | execution_time | shots | success |
|------------|---------|-------------------|------------------|----------------|-------|---------|
| Local_Simulation | AerSimulator | 0.875 | 0.833 | 12.34 | 1024 | true |
| IBM_Quantum_RealDevice | ibm_brisbane | 0.750 | 0.800 | 45.67 | 256 | true |

### Manual Result Management

```python
from your_script import save_results_to_files, add_experiment_result

# Save current results
output_dir = save_results_to_files()

# Add custom experiment
add_experiment_result("Custom_Experiment", {
    'training_accuracy': 0.95,
    'testing_accuracy': 0.88,
    'backend': 'Custom_Backend',
    'execution_time': 10.0,
    'success': True
})
```

---

## 🔧 Configuration Options

### Circuit Customization

```python
# Hardware-efficient circuit for real quantum devices
qc_hardware = QuantumClassifier(
    num_features=2,
    reps=1,  # Lower complexity for NISQ devices
    backend=sampler,
    actual_backend=quantum_backend
)

# Standard circuit for simulators
qc_standard = QuantumClassifier(
    num_features=4,
    reps=3,  # Higher complexity for simulation
    backend=None  # Defaults to local AerSimulator
)
```

### Backend Selection

```python
# Automatic backend selection
service = QiskitRuntimeService(channel='ibm_cloud')
backends = service.backends(operational=True)

# Filter by capabilities
real_backends = [b for b in backends if not b.configuration().simulator]
sim_backends = [b for b in backends if b.configuration().simulator]

# Select least busy
optimal_backend = min(real_backends, key=lambda b: b.status().pending_jobs)
```

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. IBM Quantum Connection Issues
```
Error: "Unable to retrieve instances"
```
**Solutions:**
- Verify token validity at [IBM Quantum](https://quantum.ibm.com)
- Use correct channel: `'ibm_cloud'` for new accounts
- Check account status and instance availability

#### 2. Circuit Execution Errors
```
Error 9701: Internal quantum system error
```
**Solutions:**
- Reduce dataset size for real quantum devices
- Lower circuit complexity (fewer repetitions)
- Retry execution after brief delay
- Switch to different backend

#### 3. Queue Management
```
Long queue times on quantum hardware
```
**Solutions:**
- Use `backend.status().pending_jobs` to check queue length
- Try different times of day
- Consider cloud simulators for development

#### 4. Memory Issues
```
Local simulation out of memory
```
**Solutions:**
- Reduce `data_size` parameter
- Use fewer qubits in circuit design
- Implement batch processing for large datasets

### Performance Optimization

#### For Real Quantum Hardware
```python
# Optimized settings for NISQ devices
- shots = 256  # Lower shots for faster execution
- reps = 1     # Minimal circuit depth
- max_iter = 3 # Fewer optimization iterations
- use batched processing for multiple data points
```

#### For Local Simulation
```python
# Optimized settings for simulation
- shots = 1024  # Higher shots for accuracy
- reps = 2-3    # Moderate circuit complexity
- max_iter = 20 # More optimization iterations
- larger datasets for comprehensive training
```

---

## 📊 Performance Benchmarks

### Typical Results

| Backend Type | Dataset Size | Training Accuracy | Testing Accuracy | Execution Time |
|--------------|--------------|-------------------|------------------|----------------|
| AerSimulator | 60 samples | 0.85-0.95 | 0.80-0.90 | 10-30 seconds |
| IBM Quantum Real | 8 samples | 0.70-0.85 | 0.65-0.80 | 2-10 minutes |
| IBM Quantum Sim | 20 samples | 0.80-0.90 | 0.75-0.85 | 1-3 minutes |

### Scalability Considerations

- **Local Simulation**: Up to 10-15 qubits practical
- **NISQ Hardware**: 2-5 qubits recommended for VQC
- **Dataset Size**: 50-100 samples for simulation, 8-20 for hardware
- **Circuit Depth**: Keep below 50 for real quantum devices

---

## 📚 References and Further Reading

### Scientific Papers
1. **Universal expressiveness of variational quantum classifiers and quantum kernels for support vector machines** - Nature Communications (2023)
2. **Variational quantum classifiers through the lens of the Hessian** - PLOS One (2021)
3. **Variational quantum approximate support vector machine with inference transfer** - PMC (2023)
4. **Generalization in quantum machine learning from few training data** - Nature Communications (2022)

### Technical Resources
- [Qiskit Machine Learning Documentation](https://qiskit-community.github.io/qiskit-machine-learning/)
- [IBM Quantum Platform](https://quantum.ibm.com)
- [Qiskit Textbook - Quantum Machine Learning](https://qiskit.org/textbook/ch-machine-learning/)
- [Variational Quantum Algorithms Review](https://quantum-journal.org/papers/q-2021-10-11-567/)

### Educational Materials
- [Q-munity VQC Tutorial](https://www.qmunity.tech/tutorials/building-a-variational-quantum-classifier)
- [PennyLane Variational Classifier Demo](https://pennylane.ai/qml/demos/tutorial_variational_classifier)
- [Qiskit VQC Tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02a_training_a_quantum_model_on_a_real_dataset.html)

---

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 standards
2. **Testing**: Add unit tests for new features
3. **Documentation**: Update docstrings and README
4. **Compatibility**: Ensure Qiskit v2.1.0+ compatibility

### Feature Requests
- Multi-class classification support
- Additional ansatz architectures
- Advanced error mitigation techniques
- Performance benchmarking tools

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **IBM Quantum Team** for Qiskit framework and quantum hardware access
- **Qiskit Community** for machine learning extensions and tutorials
- **Research Community** for theoretical foundations of quantum machine learning
- **NISQ Era Contributors** for practical quantum algorithm development

---

## 👨‍💻 Author

**Amon Koike (A K)**  
*Quantum Software Developer*  
*2024-2025*

---

## 🚀 Quick Start Example

```python
# Set your IBM Quantum token
YOUR_IBM_TOKEN = "your_token_here"

# Run on quantum hardware (Open Plan compatible)
print("Attempt 1: Standard quantum execution...")
qc_ibm = run_on_ibm_quantum_fixed(
    token=YOUR_IBM_TOKEN,
    use_simulator=False,  # Use real quantum device
    channel='ibm_cloud'   # New platform channel
)
```

---

*For the latest updates and detailed API documentation, visit our [GitHub repository](https://github.com/your-repo/quantum-vqc-implementation).*