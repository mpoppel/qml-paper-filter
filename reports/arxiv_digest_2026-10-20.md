# arXiv Daily Digest - 2026-10-20

**Search Period:** Last 7 days  
**Papers Found:** 7

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

### [Margin-Based Generalization for Quantum Machine Learning with Classical Data Encoding](http://arxiv.org/abs/2610.24426v1)
**Authors:** Ju-Young Ryu, Tak Hur, Daniel K. Park  
**Published:** 2026-10-17  
**Updated:** 2026-10-17  
**Categories:** quant-ph, cs.LG  

**Abstract:** Understanding generalization is essential for assessing the practical viability of quantum machine learning. In many applications, quantum models operate on classical data that must first be encoded into quantum states, making the encoding part of the effective hypothesis class. We develop a margin-based generalization framework for this setting, covering both fixed and trainable quantum data enco...

[View on arXiv](http://arxiv.org/abs/2610.24426v1) | [PDF](https://arxiv.org/pdf/2610.24426v1)

---

### [Physics-anchored residual quantum neural networks for cloud-cover parameterization](http://arxiv.org/abs/2610.26279v1)
**Authors:** Zhuo Wang, Ce Yu, Bin Yang  
**Published:** 2026-10-19  
**Updated:** 2026-10-19  
**Categories:** quant-ph, cs.AI  

**Abstract:** Compact quantum neural networks have been explored for cloud-cover parameterization, but their predictive value depends on how physical knowledge is incorporated and which classical models provide the reference. We investigate an offline residual formulation that retains a fitted \XR\ cloud-cover estimate and learns a correction to its logit in transformed target space. Using released ICON/DYAMOND...

[View on arXiv](http://arxiv.org/abs/2610.26279v1) | [PDF](https://arxiv.org/pdf/2610.26279v1)

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

### [Magneto-oscillations, nonlinearity, and nonreciprocity of Coulomb drag in quantum circuits](http://arxiv.org/abs/2608.02812v2)
**Authors:** Alex Levchenko, Mingyang Zheng, Dominique Laroche  
**Published:** 2026-08-03  
**Updated:** 2026-10-19  
**Categories:** cond-mat.mes-hall, quant-ph  

**Abstract:** We consider the problem of Coulomb drag in interactively coupled quantum circuits built of adiabatic constrictions: quantum point contacts and short quantum-wire channels. The interplay of spatial confinement and magnetic field leads to a rich oscillatory response of the drag current as a function of gate voltage and magnetic field: drag peaks track the depopulation of magnetoelectric subbands, ar...

[View on arXiv](http://arxiv.org/abs/2608.02812v2) | [PDF](https://arxiv.org/pdf/2608.02812v2)

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
