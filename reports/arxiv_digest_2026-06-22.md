# arXiv Daily Digest - 2026-06-22

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


### [Separation of Statistical Complexity and Trainability in Variational Quantum Circuits](http://arxiv.org/abs/2606.18580v1)
**Authors:** Suman Mandal, Maximillian Daughtry, Eduardo R. Mucciolo  
**Published:** 2026-06-17  
**Updated:** 2026-06-17  
**Categories:** quant-ph  

**Abstract:** Variational quantum algorithms (VQAs) are among the leading approaches for near-term quantum computing, yet their performance can degrade in barren plateau regimes characterized by vanishing gradients. A widely held intuition is that increasing circuit expressivity, often associated with random-state behavior, leads to a loss of trainability. Existing results show that sufficiently random circuits...

[View on arXiv](http://arxiv.org/abs/2606.18580v1) | [PDF](https://arxiv.org/pdf/2606.18580v1)

---

### [Latent-Conditioned Parameterized Quantum Circuits as Universal Approximators for Distributions over Quantum States](http://arxiv.org/abs/2605.28690v3)
**Authors:** Quoc Hoan Tran, Koki Chinzei, Yasuhiro Endo et al.  
**Published:** 2026-05-27  
**Updated:** 2026-06-17  
**Categories:** quant-ph, cs.LG  

**Abstract:** Many applications in quantum simulation, quantum chemistry, and quantum machine learning require not a single quantum state but an ensemble of states characterizing the heterogeneity of a target system. Preparing such ensembles state-by-state is prohibitive in both variational and fault-tolerant settings, thereby motivating a generative modeling approach. We introduce latent-conditioned parameteri...

[View on arXiv](http://arxiv.org/abs/2605.28690v3) | [PDF](https://arxiv.org/pdf/2605.28690v3)

---

### [Exponentially many initializations to avoid barren plateaus](http://arxiv.org/abs/2606.18515v1)
**Authors:** Ankit Kulshrestha, Ricard Puig, Diego García-Martín et al.  
**Published:** 2026-06-16  
**Updated:** 2026-06-16  
**Categories:** quant-ph, cs.LG, stat.ML  

**Abstract:** Barren plateaus are stated as an average-case phenomenon: pick an ansatz, initialize it naively, and concentration follows. This has led to the common view that a potential cure for barren plateaus is simply to initialize the parameters more carefully. Here we show that the situation is subtler. We introduce a first-moment framework that gives a simple operator-level diagnostic for when an initial...

[View on arXiv](http://arxiv.org/abs/2606.18515v1) | [PDF](https://arxiv.org/pdf/2606.18515v1)

---

### [Exploiting More Than Symmetry in Variational Quantum Machine Learning](http://arxiv.org/abs/2606.20316v1)
**Authors:** Markus Baumann, Claudia Linnhoff-Popien  
**Published:** 2026-06-18  
**Updated:** 2026-06-18  
**Categories:** quant-ph  

**Abstract:** The success of variational quantum learning models crucially depends on choosing parametrizations that reflect the structure of the problem at hand. Symmetries provide one of the clearest such structures: whenever transformations of the input leave the desired outcome unchanged, this invariance should be built into the model rather than discovered during training. However, imposing a symmetry does...

[View on arXiv](http://arxiv.org/abs/2606.20316v1) | [PDF](https://arxiv.org/pdf/2606.20316v1)

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
