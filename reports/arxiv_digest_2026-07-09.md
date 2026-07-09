# arXiv Daily Digest - 2026-07-09

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


### [Krylov-Lie Algebras for Variational Quantum Algorithms: Geometric, Depth-Aware Insights into Expressivity and Trainability](http://arxiv.org/abs/2607.02626v2)
**Authors:** Anžej Margeta-Cacace  
**Published:** 2026-07-02  
**Updated:** 2026-07-07  
**Categories:** quant-ph, math-ph  

**Abstract:** Variational quantum algorithms (VQAs) are a leading approach to near-term quantum computation, but their utility is limited by barren plateaus and other pathologies in their loss landscapes. Existing landscape theories based on dynamical Lie algebras, Jordan-algebraic Wishart systems, approximate t-designs, and Haar-random circuits are foundational, but they often neglect the finite-depth geometry...

[View on arXiv](http://arxiv.org/abs/2607.02626v2) | [PDF](https://arxiv.org/pdf/2607.02626v2)

---

### [How Hard Is Quantum Advantage? A Cloud Microphysics Stress Test for Variational Quantum Models](http://arxiv.org/abs/2607.04915v1)
**Authors:** Felix Herbort, Ellen Sarauer, Daniel Ohl de Mello et al.  
**Published:** 2026-07-06  
**Updated:** 2026-07-06  
**Categories:** quant-ph  

**Abstract:** Quantum machine learning (QML) could have the potential to leverage advantages of quantum over classical computing but still lacks strong evidence of actual improvements and scalability, partly due to phenomena such as barren plateaus. In this paper, we employ a hybrid quantum neural network (QNN) on a dataset on cloud microphysics, containing processes for phase transitions of water in the atmosp...

[View on arXiv](http://arxiv.org/abs/2607.04915v1) | [PDF](https://arxiv.org/pdf/2607.04915v1)

---

### [A Semantic Framework for Reproducible Variational Quantum Algorithm Execution Records](http://arxiv.org/abs/2607.03982v1)
**Authors:** Silvie Illésová, Martin Beseda  
**Published:** 2026-07-04  
**Updated:** 2026-07-04  
**Categories:** quant-ph  

**Abstract:** Variational quantum algorithms are hybrid quantum-classical workflows whose results depend on many interacting choices, including the ansatz, Hamiltonian, optimizer, backend, shot count, noise model, mitigation method, random seed, stopping criteria, and software versions. In current practice, this information is often scattered across code, configuration files, logs, backend metadata, and paper d...

[View on arXiv](http://arxiv.org/abs/2607.03982v1) | [PDF](https://arxiv.org/pdf/2607.03982v1)

---

### [Nested-Loop Trajectory-Informed Variational Quantum Solver for Interior-Point OPF](http://arxiv.org/abs/2607.03361v1)
**Authors:** Farshad Amani, Amin Kargarian  
**Published:** 2026-07-03  
**Updated:** 2026-07-03  
**Categories:** quant-ph, eess.SY, math.OC  

**Abstract:** Optimal power flow (OPF) solved by an interior-point method (IPM) requires repeatedly solving Newton linear systems. When variational quantum linear solvers (VQLS) are used, each IPM iteration involves an additional nested inner variational optimization loop, which can significantly slow the overall quantum-assisted IPM convergence. To address this challenge, this paper proposes a dual-level train...

[View on arXiv](http://arxiv.org/abs/2607.03361v1) | [PDF](https://arxiv.org/pdf/2607.03361v1)

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
