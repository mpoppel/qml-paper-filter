# arXiv Daily Digest - 2026-06-14

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


### [An LLM System for Autonomous Variational Quantum Circuit Design](http://arxiv.org/abs/2606.13380v1)
**Authors:** Kenya Sakka, Wataru Mizukami, Kosuke Mitarai  
**Published:** 2026-06-11  
**Updated:** 2026-06-11  
**Categories:** quant-ph, cs.AI  

**Abstract:** The design of high performing quantum circuits remains largely dependent on human expertise. We introduce an autonomous agentic framework that employs large language models (LLMs) to conduct iterative quantum circuit designs under explicit design constraints. Our system integrates seven components: Exploration, Generation, Discussion, Validation, Storage, Evaluation, and Review. These components f...

[View on arXiv](http://arxiv.org/abs/2606.13380v1) | [PDF](https://arxiv.org/pdf/2606.13380v1)

---

### [Representation-Induced Symmetry Trapping in Adaptive Variational Quantum Simulations of Multi-Reference Topologies](http://arxiv.org/abs/2606.13387v1)
**Authors:** Hermawan Kresno Dipojono  
**Published:** 2026-06-11  
**Updated:** 2026-06-11  
**Categories:** quant-ph, physics.chem-ph  

**Abstract:** Evaluating the trainability of adaptive quantum chemistry algorithms under multi-reference static correlation requires understanding how representation topologies intertwine with molecular geometry. We systematically expose a deep physical dependence on point-group symmetry by evaluating a spin-conserved SUSD operator pool across highly stretched configurations (2 x Re) of asymmetric LiH, symmetri...

[View on arXiv](http://arxiv.org/abs/2606.13387v1) | [PDF](https://arxiv.org/pdf/2606.13387v1)

---

### [JGRA: Jacobian Geometry Robustness Assessment in NISQ Noise-Aware Quantum Neural Networks](http://arxiv.org/abs/2606.09964v2)
**Authors:** Gianluca Scanu, Luca Barletta, Stefano Rini  
**Published:** 2026-06-08  
**Updated:** 2026-06-10  
**Categories:** quant-ph, cs.LG  

**Abstract:** The NISQ era places stringent constraints on quantum computation, where noise and decoherence fundamentally limit performance. In classical deep learning, model robustness and resilience to perturbations are well studied: deep neural networks (DNNs) maintain high performance despite pruning, noise injection, and structural perturbations due to inherent redundancy in their representations. A centra...

[View on arXiv](http://arxiv.org/abs/2606.09964v2) | [PDF](https://arxiv.org/pdf/2606.09964v2)

---

### [Generalized two-qubit Hamiltonian for Projective Quantum Feature Maps](http://arxiv.org/abs/2606.13641v1)
**Authors:** Rafael Simões do Carmo, Edson Amaro Junior, Felipe Fanchini  
**Published:** 2026-06-11  
**Updated:** 2026-06-11  
**Categories:** quant-ph  

**Abstract:** Projected quantum feature maps provide a strategy for using quantum processors as feature generators for classical machine-learning models. Building on counterdiabatic Ising-glass and one-dimensional Heisenberg PQFMs, we introduce a generalized two-qubit Hamiltonian-based PQFM that provides a unified way to encode classical features through local Pauli fields and pairwise two-qubit Pauli interacti...

[View on arXiv](http://arxiv.org/abs/2606.13641v1) | [PDF](https://arxiv.org/pdf/2606.13641v1)

---

### [Generating function and Bloch representation for quantum Fisher tensor](http://arxiv.org/abs/2603.04615v2)
**Authors:** Felipe P. Abreu, Wei Chen  
**Published:** 2026-03-04  
**Updated:** 2026-06-09  
**Categories:** quant-ph  

**Abstract:** The Uhlmann relative amplitude between two density matrices is shown to be a generating function, through which the quantum Fisher tensor that contains both the quantum Fisher information matrix and the mean Uhlmann curvature can be obtained via differentiation over system parameters. In the pure state limit, our generating function recovers that of the quantum geometric tensor proposed by Hetényi...

[View on arXiv](http://arxiv.org/abs/2603.04615v2) | [PDF](https://arxiv.org/pdf/2603.04615v2)

---

### [Visual-to-Code Authoring, Tensor-Network Debugging, and Quantum-Circuit Inspection Tools in Python](http://arxiv.org/abs/2606.08760v1)
**Authors:** Alejandro Mata Ali  
**Published:** 2026-06-07  
**Updated:** 2026-06-07  
**Categories:** quant-ph, physics.comp-ph  

**Abstract:** Tensor networks and quantum circuits are structural objects whose meaning depends on connectivity, indices, contraction order, gate placement, measurements, and related design choices. They are often easier to reason about visually than as code, yet in Python they are frequently constructed, transformed, and checked through backend-specific objects or compact symbolic expressions. This can make st...

[View on arXiv](http://arxiv.org/abs/2606.08760v1) | [PDF](https://arxiv.org/pdf/2606.08760v1)

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

**Tracked Authors:** Maria Schuld, Zoe Holmes, Marco Cerezo, Martin Larocca, Elies Gil-Fuster, Adrian Perez-Salinas, Johannes Jakob Meyer, Frederic Sauvage, Lennart Bittel

**Categories:** quant-ph, cs.LG, cs.AI, stat.ML
**Lookback Period:** 7 days
