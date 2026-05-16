# arXiv Daily Digest - 2026-05-16

**Search Period:** Last 7 days  
**Papers Found:** 4

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


### [Scalable Quantum Machine Learning via Multi-layer Fully-Connected Variational Quantum Circuits](http://arxiv.org/abs/2602.16623v2)
**Authors:** Howard Su, Chen-Yu Liu, Samuel Yen-Chi Chen et al.  
**Published:** 2026-02-18  
**Updated:** 2026-05-10  
**Categories:** quant-ph  

**Abstract:** Variational Quantum Circuits (VQC) are promising models for quantum machine learning, but standard monolithic architectures face an expressivity--trainability dilemma: small circuits can be under-parameterized, while larger circuits are difficult to simulate and optimize. We propose Multi-Layer Fully-Connected Variational Quantum Circuits (FC-VQC), a modular framework that decomposes high-dimensio...

[View on arXiv](http://arxiv.org/abs/2602.16623v2) | [PDF](https://arxiv.org/pdf/2602.16623v2)

---

### [Stopping Reliability in Adaptive Krylov-Shadow Quantum Fisher Information Estimation](http://arxiv.org/abs/2605.14338v1)
**Authors:** Erjie Liu, Yangshuai Wang  
**Published:** 2026-05-14  
**Updated:** 2026-05-14  
**Categories:** quant-ph  

**Abstract:** Adaptive quantum Fisher information (QFI) estimation requires a stopping rule that distinguishes accuracy from apparent numerical stability. For Krylov-shadow QFI estimators, finite Krylov order $K$ produces truncation bias, while finite sample budget $M$ produces finite-$M$ sampling-side error. We show that a width-only empirical stopping rule, based on interval width and local Krylov stability, ...

[View on arXiv](http://arxiv.org/abs/2605.14338v1) | [PDF](https://arxiv.org/pdf/2605.14338v1)

---

### [Algorithmic Advantage on a Gate-Based Photonic Quantum Neural Network](http://arxiv.org/abs/2605.10801v1)
**Authors:** Solomon McKiernan, Luca Sapienza  
**Published:** 2026-05-11  
**Updated:** 2026-05-11  
**Categories:** quant-ph, physics.optics  

**Abstract:** We report on a gate-based variational quantum classifier implemented with single photons and probabilistic gates, to emulate the standard quantum circuit model framework. We evaluate the expressive power of two deployable quantum neural networks (QNNs) by computing their effective dimension, a capacity measure grounded in a proven generalization-error bound, and compare them with classical artific...

[View on arXiv](http://arxiv.org/abs/2605.10801v1) | [PDF](https://arxiv.org/pdf/2605.10801v1)

---

### [Symmetry-Protected Basin Localization in Variational Quantum Eigensolvers](http://arxiv.org/abs/2605.09909v1)
**Authors:** Yangshuai Wang  
**Published:** 2026-05-11  
**Updated:** 2026-05-11  
**Categories:** quant-ph  

**Abstract:** Variational quantum eigensolvers fail before optimization begins when strong correlation splits the molecular energy landscape into competing basins and the initial state selects a non-ground-state basin. We introduce a geometry-conditioned preconditioner $\mathcal{P}_{\mathrm{eq}}:\mathbf{R}\mapsto\boldsymbolθ_0$ constrained by the $SE(3)$ covariance of the molecular Hamiltonian, so that nuclear ...

[View on arXiv](http://arxiv.org/abs/2605.09909v1) | [PDF](https://arxiv.org/pdf/2605.09909v1)

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
