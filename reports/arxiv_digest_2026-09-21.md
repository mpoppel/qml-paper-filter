# arXiv Daily Digest - 2026-09-21

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


### [A quantum representation of $π$ fragmentation functions through variational quantum circuits](http://arxiv.org/abs/2609.20557v1)
**Authors:** David F. Rentería-Estrada, Roger J. Hernández-Pinto, Germán Rodrigo et al.  
**Published:** 2026-09-17  
**Updated:** 2026-09-17  
**Categories:** hep-ph, quant-ph  

**Abstract:** We present a variational quantum-circuit model for fragmentation functions (FFs). Isospin and charge-conjugation symmetries are imposed to construct an independent six-flavor basis describing charged and neutral pion production, while physics-inspired Ansätze, including logarithmic feature maps and mass thresholds, encode the relevant kinematics. This quantum architecture substantially reduces the...

[View on arXiv](http://arxiv.org/abs/2609.20557v1) | [PDF](https://arxiv.org/pdf/2609.20557v1)

---

### [Hybrid Variational Quantum Circuits for Multivariate Regression and High-Dimensional Data Reconstruction](http://arxiv.org/abs/2609.17358v1)
**Authors:** Koffi Ognandon Ayena, Frédéric Holweck, Serge Iovleff et al.  
**Published:** 2026-09-15  
**Updated:** 2026-09-15  
**Categories:** cs.LG, stat.ML  

**Abstract:** Variational quantum circuits (VQCs) are parameterized quantum circuits optimized classically. We propose a hybrid variational quantum circuit (HVQC) extending VQCs with a classical affine post-measurement layer, enabling vector-valued regression without the linear overhead of independent scalar circuits. Theoretically, we show that elementary one-and two-qubit circuits can approximate quadratic fu...

[View on arXiv](http://arxiv.org/abs/2609.17358v1) | [PDF](https://arxiv.org/pdf/2609.17358v1)

---

### [From Trainability Diagnostics to Optimization Claims: Boundaries and Controls in Variational Quantum Optimization](http://arxiv.org/abs/2609.21243v1)
**Authors:** Pilsung Kang  
**Published:** 2026-09-18  
**Updated:** 2026-09-18  
**Categories:** quant-ph, cs.LG  

**Abstract:** Barren plateau diagnostics characterize whether gradient signal remains available for training, but surviving signal need not translate into successful optimization. We study this trainability--optimization gap at the level of optimizer steps. Treating coefficient-weighted Hamiltonian-term gradients as task-like components, we introduce step-level diagnostics and derive an exact bridge between sig...

[View on arXiv](http://arxiv.org/abs/2609.21243v1) | [PDF](https://arxiv.org/pdf/2609.21243v1)

---

### [Information limits of photonic lantern wavefront sensing: a Fisher- and quantum-Fisher-information framework and its relation to Fourier-filtering sensitivity limits](http://arxiv.org/abs/2607.29342v2)
**Authors:** Kalaga Madhav  
**Published:** 2026-07-31  
**Updated:** 2026-09-16  
**Categories:** astro-ph.IM, physics.optics, quant-ph  

**Abstract:** The photonic lantern is an all-photonic wavefront sensor native to single-mode-fibre-fed instruments, but its performance is almost always quoted through a specific reconstruction algorithm, obscuring how much wavefront information the device itself encodes. We develop, from first principles, the Fisher-information and Cramer-Rao theory of the photonic-lantern wavefront sensor, benchmark it agains...

[View on arXiv](http://arxiv.org/abs/2607.29342v2) | [PDF](https://arxiv.org/pdf/2607.29342v2)

---

### [On the convergence of the variational quantum eigensolver and quantum optimal control](http://arxiv.org/abs/2509.05295v3)
**Authors:** Marco Wiedmann, Daniel Burgarth, Gunther Dirr et al.  
**Published:** 2025-09-05  
**Updated:** 2026-09-14  
**Categories:** quant-ph, math.OC  

**Abstract:** When do variational quantum algorithms converge to a globally optimal solution? Despite extensive work, this question remains largely open. We develop a convergence theory for the variational quantum eigensolver (VQE), using quantum control landscape terminology to prove a sufficient criterion guaranteeing convergence to a Hamiltonian's ground state for almost all initial parameters. Specifically,...

[View on arXiv](http://arxiv.org/abs/2509.05295v3) | [PDF](https://arxiv.org/pdf/2509.05295v3)

---

### [The Pauli Probability Spectrum Carries the Pure-State Quantum Fisher Metric](http://arxiv.org/abs/2608.21437v3)
**Authors:** E. A. Ramirez Trino, M. A. Rajabpour  
**Published:** 2026-08-17  
**Updated:** 2026-09-17  
**Categories:** quant-ph, cond-mat.stat-mech, math-ph  

**Abstract:** How a quantum state responds to small parameter changes underlies state distinguishability, many-body response, and metrological sensitivity. The Pauli probability spectrum provides a natural description of many-body quantum states, but it is not evident how much of this response survives in the corresponding classical distribution. Here we show that, for pure states, the complete labeled Pauli di...

[View on arXiv](http://arxiv.org/abs/2608.21437v3) | [PDF](https://arxiv.org/pdf/2608.21437v3)

---

### [Variational Quantum Transformer Architecture for Synthetic Language Generation](http://arxiv.org/abs/2609.18565v1)
**Authors:** Julian Hager, Michael Kölle, Gerhard Stenzel et al.  
**Published:** 2026-09-16  
**Updated:** 2026-09-16  
**Categories:** quant-ph, cs.CL, cs.LG  

**Abstract:** We propose a compact NISQ-compatible quantum transformer architecture for synthetic QNLP sequence modelling. The model preserves the autoregressive next-token interface of a classical transformer, but replaces attention and feed-forward sublayers with variational quantum encoder blocks, connector circuits, decoder blocks and a direct two-qubit measurement readout. Token contexts are angle-encoded ...

[View on arXiv](http://arxiv.org/abs/2609.18565v1) | [PDF](https://arxiv.org/pdf/2609.18565v1)

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
