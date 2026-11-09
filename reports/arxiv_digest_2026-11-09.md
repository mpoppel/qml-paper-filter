# arXiv Daily Digest - 2026-11-09

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


### [Data informed Ansatz Selection for Variational Quantum Circuits](http://arxiv.org/abs/2611.05079v1)
**Authors:** Manav Seksaria, Advith Desu, Anil Prabhakar  
**Published:** 2026-11-03  
**Updated:** 2026-11-03  
**Categories:** quant-ph, cs.LG  

**Abstract:** Choosing a variational quantum circuit (VQC) usually means searching over ansatze and depths, and training each candidate. We give three rules that fix these choices from the data and the gate set alone, before any training. First, a circuit without entangling gates has a rank-one Fourier coefficient tensor on every subset of qubits. A singular value decomposition of a dataset's Fourier coefficien...

[View on arXiv](http://arxiv.org/abs/2611.05079v1) | [PDF](https://arxiv.org/pdf/2611.05079v1)

---

### [What Does the Effective Dimension of a Hybrid Quantum Neural Network Actually Measure?](http://arxiv.org/abs/2610.14070v2)
**Authors:** Md Masum Mia, Amith Khandakar, Mohamed Arselene Ayari  
**Published:** 2026-10-07  
**Updated:** 2026-11-04  
**Categories:** quant-ph, cs.AI  

**Abstract:** Hybrid quantum neural networks (QNNs) pass features from a classical backbone into a variational quantum circuit, and many studies log a scalar called the effective dimension as a sign of capacity or model quality. The effective dimension is usually defined as the participation ratio (PR) of the quantum Fisher information matrix (QFIM), but studies in practice log a PR of cheap parameter sensitivi...

[View on arXiv](http://arxiv.org/abs/2610.14070v2) | [PDF](https://arxiv.org/pdf/2610.14070v2)

---

### [Band structure of two-dimensional graphene and of one-dimensional nanoribbons via variational quantum algorithms: from energies to wavefunction quality](http://arxiv.org/abs/2611.04975v1)
**Authors:** Raúl Medrano Millán, Rodrigo Gil-Merino y Rubio, Juan Borge et al.  
**Published:** 2026-11-03  
**Updated:** 2026-11-03  
**Categories:** cond-mat.mtrl-sci, quant-ph  

**Abstract:** Simulating quantum many-body systems classically encounters exponential bottlenecks, which quantum devices circumvent by natively encoding wavefunctions. In the Noisy Intermediate-Scale Quantum (NISQ) era, however, hardware noise constrains simulation accuracy. As a stepping stone toward strongly correlated materials, we compute the band structure of 2D monolayer graphene and 1D nanoribbons (zigza...

[View on arXiv](http://arxiv.org/abs/2611.04975v1) | [PDF](https://arxiv.org/pdf/2611.04975v1)

---

### [Exploring topological signatures of quantum Fisher information scaling in graph states](http://arxiv.org/abs/2611.08593v1)
**Authors:** Majid Hassani, Patrick Emonts, Jordi Tura  
**Published:** 2026-11-05  
**Updated:** 2026-11-05  
**Categories:** quant-ph  

**Abstract:** Graph states are a versatile resource for quantum metrology, and their quantum Fisher information (QFI) has a closed-form expression in terms of the underlying graph: it counts pairs of vertices with identical neighborhoods. It remains unclear, however, which global structural properties of a graph determine whether its QFI exceeds the standard quantum limit (SQL). Here we ask whether topological ...

[View on arXiv](http://arxiv.org/abs/2611.08593v1) | [PDF](https://arxiv.org/pdf/2611.08593v1)

---

### [Higher Order Component Attribution via Learned Surrogates in High Energy Physics](http://arxiv.org/abs/2611.08656v1)
**Authors:** Lauri Laatu, Aritra Bal, Benedikt Maier et al.  
**Published:** 2026-11-05  
**Updated:** 2026-11-05  
**Categories:** cs.LG, hep-ex, hep-ph  

**Abstract:** Experimental High Energy Physics (HEP) has adopted the transformer architecture for identification of jets, particularly the state-of-the-art Particle Transformer (ParT) which injects explicit information about particle correlations into the attention mechanism. While able to achieve state-of-the-art results in accuracy, interpretability to its effectiveness has been limited. In this paper we prop...

[View on arXiv](http://arxiv.org/abs/2611.08656v1) | [PDF](https://arxiv.org/pdf/2611.08656v1)

---

### [Polynomial-time classical simulation of boson and fermion sampling under thermal loss](http://arxiv.org/abs/2611.08444v1)
**Authors:** Armando Angrisani, Hela Mhiri  
**Published:** 2026-11-05  
**Updated:** 2026-11-05  
**Categories:** quant-ph, cs.CC, math-ph, physics.optics  

**Abstract:** Sampling experiments are a leading approach to demonstrating quantum computational advantage, yet their robustness to experimental noise is not completely understood. We develop a common framework for classical simulation of boson and fermion sampling under thermal loss, including pure loss. For Fock-state boson sampling, Gaussian boson sampling with fixed uniform squeezing, and fermion sampling w...

[View on arXiv](http://arxiv.org/abs/2611.08444v1) | [PDF](https://arxiv.org/pdf/2611.08444v1)

---

### [Dissipation accelerates quantum and classical simulation of open-system dynamics](http://arxiv.org/abs/2609.40174v2)
**Authors:** Armando Angrisani, Ricard Puig, Yanting Teng et al.  
**Published:** 2026-09-30  
**Updated:** 2026-11-05  
**Categories:** quant-ph, cond-mat.stat-mech  

**Abstract:** Simulating open quantum systems reveals how environmental coupling shapes relaxation, excitation transport, and the dynamics of quantum correlations. On quantum hardware, dissipative channels add operations and might seem to increase cost. However, we show that a broad class of Pauli noise, including depolarization, can ease quantum and classical simulation by exponentially suppressing high-weight...

[View on arXiv](http://arxiv.org/abs/2609.40174v2) | [PDF](https://arxiv.org/pdf/2609.40174v2)

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
