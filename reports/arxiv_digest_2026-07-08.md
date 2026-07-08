# arXiv Daily Digest - 2026-07-08

**Search Period:** Last 7 days  
**Papers Found:** 8

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


### [Beyond the Expressivity-Trainability Paradox: A Dynamical Lie Algebra Perspective on Navigating Barren Plateaus in Quantum Machine Learning](http://arxiv.org/abs/2606.31536v3)
**Authors:** Kung-Ming Lan, Edward Huang  
**Published:** 2026-06-30  
**Updated:** 2026-07-02  
**Categories:** cs.LG, quant-ph  

**Abstract:** As Quantum Machine Learning (QML) transitions toward practical implementation, the field faces a critical architectural bottleneck that challenges the fundamental assumptions of classical statistical learning theory. In classical deep learning, increasing model capacity typically risks overfitting. However, this study advances a counter-intuitive paradigm: unstructured contemporary QML architectur...

[View on arXiv](http://arxiv.org/abs/2606.31536v3) | [PDF](https://arxiv.org/pdf/2606.31536v3)

---

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

### [The Dynamical Lie Algebra of QAOA-MaxCut on the Complete Graph](http://arxiv.org/abs/2607.00945v1)
**Authors:** Jonathan Allcock, Pei Yuan, Shengyu Zhang  
**Published:** 2026-07-01  
**Updated:** 2026-07-01  
**Categories:** quant-ph  

**Abstract:** We give an analytical expression for the dynamical Lie algebra corresponding to the QAOA-MaxCut problem on complete graphs, and show that the variance of the associated loss function scales linearly in the number of qubits. This solves an open problem from [ASYZ26] and confirms that such systems do not exhibit barren plateaus. The proof is based on projecting the dynamical Lie algebra generators o...

[View on arXiv](http://arxiv.org/abs/2607.00945v1) | [PDF](https://arxiv.org/pdf/2607.00945v1)

---

### [A Semantic Framework for Reproducible Variational Quantum Algorithm Execution Records](http://arxiv.org/abs/2607.03982v1)
**Authors:** Silvie Illésová, Martin Beseda  
**Published:** 2026-07-04  
**Updated:** 2026-07-04  
**Categories:** quant-ph  

**Abstract:** Variational quantum algorithms are hybrid quantum-classical workflows whose results depend on many interacting choices, including the ansatz, Hamiltonian, optimizer, backend, shot count, noise model, mitigation method, random seed, stopping criteria, and software versions. In current practice, this information is often scattered across code, configuration files, logs, backend metadata, and paper d...

[View on arXiv](http://arxiv.org/abs/2607.03982v1) | [PDF](https://arxiv.org/pdf/2607.03982v1)

---

### [Comparing the Performance of Leading VQE Algorithms for Computing Ground-State Energies of Amino Acids](http://arxiv.org/abs/2607.02620v1)
**Authors:** Sanskriti Shindadkar, Clyde Villacrusis, Jasper Andrews et al.  
**Published:** 2026-07-02  
**Updated:** 2026-07-02  
**Categories:** quant-ph, cs.ET  

**Abstract:** Simulating molecules is a major application of quantum computing, with the potential to overcome exponential scaling constraints of classical computation. Researchers use different methods in order to evaluate the readiness of NISQ computers in order to test current simulation capabilities. We present an integrated repository with reproducible benchmarks of over 10 different ansatzes from publishe...

[View on arXiv](http://arxiv.org/abs/2607.02620v1) | [PDF](https://arxiv.org/pdf/2607.02620v1)

---

### [Quantum circuit design via dynamic Pauli constraints](http://arxiv.org/abs/2605.22744v2)
**Authors:** James R. Wootton, Merlin Incerti-Medici, Daniel Bultrini et al.  
**Published:** 2026-05-21  
**Updated:** 2026-07-01  
**Categories:** quant-ph  

**Abstract:** We introduce the Motte model, a software-oriented model of quantum computation motivated by the practical constraints of near-term quantum hardware. In this model, gates are specified by constraints expressed in terms of Pauli observables, with each disjoint layer of gates accompanied by a pairwise or k-local quantum state tomography of the device. We prove that the model is equivalent to the coup...

[View on arXiv](http://arxiv.org/abs/2605.22744v2) | [PDF](https://arxiv.org/pdf/2605.22744v2)

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
