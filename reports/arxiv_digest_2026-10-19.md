# arXiv Daily Digest - 2026-10-19

**Search Period:** Last 7 days  
**Papers Found:** 6

## Summary

This digest covers:
- Serial vs. parallel QNN architectures (expressivity, trainability)
- Fourier analysis of parameterized quantum circuits
- Dynamical Lie algebra (DLA) and QFIM rank theory
- Barren plateaus, overparameterization, near-zero initialization
- Data re-uploading / trainable frequency feature maps
- VQE and Hamiltonian learning

---

## Papers


### [Expressivity of IQP Feature-Map Quantum Kernels via Reproducing Kernel Hilbert Spaces](http://arxiv.org/abs/2610.23692v1)
**Authors:** Valeria Repetto, Stefano Melchionna, Romina D'Aurizio  
**Published:** 2026-10-16  
**Updated:** 2026-10-16  
**Categories:** quant-ph, cs.LG  

**Abstract:** Quantum kernels are among the most promising methods in quantum machine learning, providing a natural way to embed data into high-dimensional Hilbert spaces via quantum circuits and capturing non-linear relationships in data. Their learning behaviour is governed by the function class they represent, which is formally characterized by the structure of the associated Reproducing Kernel Hilbert Space...

[View on arXiv](http://arxiv.org/abs/2610.23692v1) | [PDF](https://arxiv.org/pdf/2610.23692v1)

---

### [Task-aligned spectral inductive biases in parameterized quantum circuits](http://arxiv.org/abs/2610.23834v1)
**Authors:** Felix Frohnert, Adrián Pérez-Salinas, Sebastian Lange et al.  
**Published:** 2026-10-16  
**Updated:** 2026-10-16  
**Categories:** quant-ph, cs.LG  

**Abstract:** Parameterized quantum circuits are widely investigated in quantum machine learning, yet how their architecture can provide task-aligned inductive biases remains poorly understood. We study data reuploading quantum circuits with two distinct inputs: a signal that varies between samples and a fixed, task-defined spectral query that specifies how the signal should be probed. We show that encoding the...

[View on arXiv](http://arxiv.org/abs/2610.23834v1) | [PDF](https://arxiv.org/pdf/2610.23834v1)

---

### [A Tutorial on Quantum Amplitude Embedding with Data Re-Uploading](http://arxiv.org/abs/2610.17836v1)
**Authors:** Hyojun Ahn, Joongheon Kim  
**Published:** 2026-10-13  
**Updated:** 2026-10-13  
**Categories:** quant-ph, cs.AI, cs.LG  

**Abstract:** Amplitude embedding lets a variational quantum model read many classical features with few qubits, but the model output is then only a quadratic form in the normalized input. Data re-uploading, which encodes the input repeatedly between trainable blocks, can be used to enrich such a model. Because every additional upload of an amplitude embedding repeats a costly state preparation, the gain that a...

[View on arXiv](http://arxiv.org/abs/2610.17836v1) | [PDF](https://arxiv.org/pdf/2610.17836v1)

---

### [QGrAT: Quantum Grade-Aware Transformer for Jet Tagging](http://arxiv.org/abs/2610.17351v1)
**Authors:** Pallavi Bhardwaj, Michael Spannowsky  
**Published:** 2026-10-12  
**Updated:** 2026-10-12  
**Categories:** hep-ph, quant-ph  

**Abstract:** Lorentz symmetry and the three-prong structure of hadronic top decays provide physical guidance for designing compact quantum jet taggers. We introduce QGrAT (Quantum Grade-Aware Transformer), a quantum transformer that incorporates these structures into its interactions between constituent qubits. Geometric features constructed from the constituent four-momenta are organised by the grades of the ...

[View on arXiv](http://arxiv.org/abs/2610.17351v1) | [PDF](https://arxiv.org/pdf/2610.17351v1)

---

### [Infinite magic and the emergence of the continuum](http://arxiv.org/abs/2610.20790v1)
**Authors:** Leo Shaposhnik, Lorenzo Leone, Lennart Bittel et al.  
**Published:** 2026-10-14  
**Updated:** 2026-10-14  
**Categories:** quant-ph, hep-th, math-ph  

**Abstract:** Which features of a quantum state are needed for a relativistic quantum field theory to emerge from a lattice system? We address this question by establishing a quantitative relationship between the type classification of hyperfinite von Neumann algebras and the quantum-information-theoretic resource of nonstabilizerness, also called magic. Our results establish two complementary parts of this rel...

[View on arXiv](http://arxiv.org/abs/2610.20790v1) | [PDF](https://arxiv.org/pdf/2610.20790v1)

---

### [Classification of quantum states through conserved operators](http://arxiv.org/abs/2610.20988v1)
**Authors:** Federico Petrovich, R. Rossignoli, M. Cerezo  
**Published:** 2026-10-14  
**Updated:** 2026-10-14  
**Categories:** quant-ph  

**Abstract:** Determining whether quantum states can be interconverted by a prescribed set of transformations is a fundamental problem in quantum information. Here we develop a general classification framework based on conserved operators (COs), i.e., not necessarily hermitian matrices having the state of interest as an eigenstate. We first show that COs belonging to the algebra generating the allowed transform...

[View on arXiv](http://arxiv.org/abs/2610.20988v1) | [PDF](https://arxiv.org/pdf/2610.20988v1)

---

---

## Search Configuration

**Queries:**
- ti:"quantum circuit" AND (ti:fourier OR ti:frequency OR ti:spectral OR abs:expressivity)
- (ti:"barren plateau" OR ti:"loss landscape" OR ti:"near-zero initialization") AND quantum
- (ti:"dynamical Lie" OR ti:"Lie algebra" OR ti:"quantum Fisher" OR ti:overparameterization) AND quantum
- (ti:"data re-uploading" OR ti:"data encoding" OR ti:"feature map") AND (quantum OR qubit)
- (ti:"variational quantum" OR ti:"quantum neural network" OR ti:"parameterized quantum") AND (machine learning OR trainability OR expressivity)
- (ti:"variational quantum eigensolver" OR ti:VQE OR ti:"transverse field Ising") AND (barren OR landscape OR layer)

**Tracked Authors:** Maria Schuld, Zoe Holmes, Marco Cerezo, Martin Larocca, Elies Gil-Fuster, Adrian Perez-Salinas, Johannes Jakob Meyer, Frederic Sauvage, Lennart Bittel, Michael Spannowsky, Vishal S. Ngairangbam, Hela Mhiri, Jonas Landmann

**Categories:** quant-ph, cs.LG, cs.AI, stat.ML
**Lookback Period:** 7 days
