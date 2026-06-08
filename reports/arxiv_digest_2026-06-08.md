# arXiv Daily Digest - 2026-06-08

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


### [Machine-Learning Prediction of Quantum Fisher Information from Collective Spin and Spectral Features](http://arxiv.org/abs/2606.02986v1)
**Authors:** Yusef Maleki, Luis D. Zambrano Palma  
**Published:** 2026-06-02  
**Updated:** 2026-06-02  
**Categories:** quant-ph  

**Abstract:** Quantum Fisher information (QFI) is a fundamental quantifier in quantum metrology, determining the ultimate precision achievable in parameter-estimation protocols through the quantum Cramér-Rao bound. However, direct evaluation of the QFI generally requires detailed knowledge of the density matrix, making it increasingly demanding as the Hilbert-space dimension grows. In this work, we investigate ...

[View on arXiv](http://arxiv.org/abs/2606.02986v1) | [PDF](https://arxiv.org/pdf/2606.02986v1)

---

### [Latent-Conditioned Parameterized Quantum Circuits as Universal Approximators for Distributions over Quantum States](http://arxiv.org/abs/2605.28690v2)
**Authors:** Quoc Hoan Tran, Koki Chinzei, Yasuhiro Endo et al.  
**Published:** 2026-05-27  
**Updated:** 2026-06-01  
**Categories:** quant-ph, cs.LG  

**Abstract:** Many applications in quantum simulation, quantum chemistry, and quantum machine learning require not a single quantum state but an ensemble of states characterizing the heterogeneity of a target system. Preparing such ensembles state-by-state is prohibitive in both variational and fault-tolerant settings, motivating a generative-modeling approach. We introduce latent-conditioned parameterized quan...

[View on arXiv](http://arxiv.org/abs/2605.28690v2) | [PDF](https://arxiv.org/pdf/2605.28690v2)

---

### [Fundamental Limits of Non-Hermitian Sensing from Quantum Fisher Information](http://arxiv.org/abs/2603.10614v2)
**Authors:** Jan Wiersig, Stefan Rotter  
**Published:** 2026-03-11  
**Updated:** 2026-06-03  
**Categories:** quant-ph, physics.optics  

**Abstract:** Exceptional points (EPs) exhibit strongly enhanced spectral responses and are therefore promising candidates for sensing applications. Whether these non-Hermitian degeneracies provide a genuine advantage in the quantum regime has been the subject of ongoing debate. Here, we address this issue within a scattering-matrix formalism for sensing with coherent light, which allows the quantum Fisher info...

[View on arXiv](http://arxiv.org/abs/2603.10614v2) | [PDF](https://arxiv.org/pdf/2603.10614v2)

---

### [Scalable On-Hardware Training of Quantum Neural Networks and Application to Clinical Data Imputation](http://arxiv.org/abs/2606.03517v1)
**Authors:** Natansh Mathur, Panagiotis Kl. Barkoutsos, Masako Yamada et al.  
**Published:** 2026-06-02  
**Updated:** 2026-06-02  
**Categories:** quant-ph, cs.AI, cs.LG  

**Abstract:** Training quantum neural networks (QNNs) on quantum hardware is currently bottlenecked by the cost of gradient estimation: standard parameter-shift methods require a number of circuit evaluations that grows quadratically with the number of trainable parameters, making hardware-based optimisation impractical beyond small system sizes. In this work, we introduce a training framework that reduces this...

[View on arXiv](http://arxiv.org/abs/2606.03517v1) | [PDF](https://arxiv.org/pdf/2606.03517v1)

---

### [Game, Set, Quantum: Parameterized Quantum Circuit for Correlated Equilibrium in Bayesian Games](http://arxiv.org/abs/2606.03109v1)
**Authors:** Param Pathak, Vidhi Oad, Nouhaila Innan et al.  
**Published:** 2026-06-02  
**Updated:** 2026-06-02  
**Categories:** quant-ph  

**Abstract:** Strategic decision-making among many agents under incomplete information is central to economics, security, and multi-agent artificial intelligence (AI). Computing equilibria in such settings is challenging because the joint type-action space grows exponentially with the number of players. In binary-type, binary-action Bayesian games, an explicit representation over type-action profiles requires O...

[View on arXiv](http://arxiv.org/abs/2606.03109v1) | [PDF](https://arxiv.org/pdf/2606.03109v1)

---

### [Double-bracket quantum algorithms for thermal state preparation](http://arxiv.org/abs/2606.05947v1)
**Authors:** Andrew Wright, Reyhaneh Aghaei Saem, Supanut Thanasilp et al.  
**Published:** 2026-06-04  
**Updated:** 2026-06-04  
**Categories:** quant-ph  

**Abstract:** We propose quantum algorithms for preparing thermal states via the simulation of the thermofield double states. The key idea is to leverage double-bracket quantum algorithms to implement imaginary-time evolution on thermofield double states, whose reduced state realizes the Gibbs state. Our method, termed double-bracket thermofield double (DB-TFD), introduces two variants. The first, the vanilla D...

[View on arXiv](http://arxiv.org/abs/2606.05947v1) | [PDF](https://arxiv.org/pdf/2606.05947v1)

---

### [Resource-efficient energy-based operator selection in fermionic ADAPT-VQE via exact Hamiltonian transformation](http://arxiv.org/abs/2606.04786v1)
**Authors:** Emanuele Rossi, Erik Rosendahl Kjellgren, Artur F. Izmaylov et al.  
**Published:** 2026-06-03  
**Updated:** 2026-06-03  
**Categories:** quant-ph, physics.chem-ph, physics.comp-ph  

**Abstract:** The energy-based approach to operator selection in ADAPT-VQE relies on reconstructing the one-parameter energy landscape for each operator in the pool. In fermionic implementations, the cost of reconstructing this energy landscape often becomes a bottleneck. We address this issue through an exact Hamiltonian transformation that reformulates the one-parameter energy landscape according to a generat...

[View on arXiv](http://arxiv.org/abs/2606.04786v1) | [PDF](https://arxiv.org/pdf/2606.04786v1)

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
