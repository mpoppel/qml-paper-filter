# arXiv Daily Digest - 2026-04-28

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


### [The effect of the number of parameters and the number of local feature patches on loss landscapes in distributed quantum neural networks](http://arxiv.org/abs/2504.19239v2)
**Authors:** Yoshiaki Kawase  
**Published:** 2025-04-27  
**Updated:** 2026-04-22  
**Categories:** quant-ph, cs.LG  

**Abstract:** Quantum neural networks hold promise for tackling computationally challenging tasks that are intractable for classical computers. However, their practical application is hindered by significant optimization challenges, arising from complex loss landscapes characterized by barren plateaus and numerous local minima. These problems become more severe as the number of parameters or qubits increases, h...

[View on arXiv](http://arxiv.org/abs/2504.19239v2) | [PDF](https://arxiv.org/pdf/2504.19239v2)

---

### [H-EFT-VA: An Effective-Field-Theory Variational Ansatz with Provable Barren Plateau Avoidance](http://arxiv.org/abs/2601.10479v2)
**Authors:** Eyad I. B Hamid  
**Published:** 2026-01-15  
**Updated:** 2026-04-23  
**Categories:** quant-ph, cs.LG, math-ph  

**Abstract:** Variational Quantum Algorithms (VQAs) are critically threatened by the Barren Plateau (BP) phenomenon. In this work, we introduce the H-EFT Variational Ansatz (H-EFT-VA), an architecture inspired by Effective Field Theory (EFT). By enforcing a hierarchical "UV-cutoff" on initialization, we theoretically restrict the circuit's state exploration, preventing the formation of approximate unitary 2-des...

[View on arXiv](http://arxiv.org/abs/2601.10479v2) | [PDF](https://arxiv.org/pdf/2601.10479v2)

---

### [Fixed-Reservoir vs Variational Quantum Architectures for Chaotic Dynamics: Benchmarking QRC and QPINN on the Lorenz System](http://arxiv.org/abs/2604.23743v1)
**Authors:** Tushar Pandey  
**Published:** 2026-04-26  
**Updated:** 2026-04-26  
**Categories:** quant-ph, cs.LG  

**Abstract:** Deploying quantum machine learning on NISQ devices requires architectures where training overhead does not negate computational advantages. We systematically compare two quantum approaches for chaotic time-series prediction on the Lorenz system: a variational Quantum Physics-Informed Neural Network (QPINN) and a Quantum Reservoir Computing (QRC) framework utilizing a fixed transverse-field Ising H...

[View on arXiv](http://arxiv.org/abs/2604.23743v1) | [PDF](https://arxiv.org/pdf/2604.23743v1)

---

### [Ansätz Expressivity and Optimization in Variational Quantum Simulations of Transverse-field Ising Model Across System Sizes](http://arxiv.org/abs/2604.20961v1)
**Authors:** Ashutosh P. Tripathi, Nilmani Mathur, Vikram Tripathi  
**Published:** 2026-04-22  
**Updated:** 2026-04-22  
**Categories:** quant-ph, cond-mat.stat-mech, hep-lat  

**Abstract:** We explore the application of the Variational Quantum Eigensolver (VQE) to investigate the ground state properties, particularly the entanglement entropy, of the Transverse Field Ising Model (TFIM) in one, two, and three dimensions, considering systems of up to 27 spins. By benchmarking VQE results against exact diagonalization and analyzing the entanglement properties across different system size...

[View on arXiv](http://arxiv.org/abs/2604.20961v1) | [PDF](https://arxiv.org/pdf/2604.20961v1)

---

### [A four-player potential game for barren-plateau-aware quantum ansatz design](http://arxiv.org/abs/2604.21955v1)
**Authors:** Rubén Darío Guerrero  
**Published:** 2026-04-23  
**Updated:** 2026-04-23  
**Categories:** quant-ph, cs.MA  

**Abstract:** We cast the design of parameterized quantum circuits as a four-player potential game whose state is a circuit directed acyclic graph (DAG) and whose players encode trainability, non-stabilizerness, task performance, and hardware cost. Per-player restricted action sets factorize the move space into append, remove, retype, and rewire operations; a block-coordinate $\varepsilon$-Nash residual $δ_\tex...

[View on arXiv](http://arxiv.org/abs/2604.21955v1) | [PDF](https://arxiv.org/pdf/2604.21955v1)

---

### [QuanForge: A Mutation Testing Framework for Quantum Neural Networks](http://arxiv.org/abs/2604.20706v1)
**Authors:** Minqi Shao, Shangzhou Xia, Jianjun Zhao  
**Published:** 2026-04-22  
**Updated:** 2026-04-22  
**Categories:** cs.SE, cs.AI  

**Abstract:** With the growing synergy between deep learning and quantum computing, Quantum Neural Networks (QNNs) have emerged as a promising paradigm by leveraging quantum parallelism and entanglement. However, testing QNNs remains underexplored due to their complex quantum dynamics and limited interpretability. Developing a mutation testing technique for QNNs is promising while requires addressing stochastic...

[View on arXiv](http://arxiv.org/abs/2604.20706v1) | [PDF](https://arxiv.org/pdf/2604.20706v1)

---

### [Quantum-Enhanced Recurrent Neural Networks via Variational Quantum Gating for Battery State of Health Prediction](http://arxiv.org/abs/2604.20438v1)
**Authors:** Yin Xu, Qinglin Liu, Li Gao et al.  
**Published:** 2026-04-22  
**Updated:** 2026-04-22  
**Categories:** quant-ph  

**Abstract:** Accurate state-of-health (SOH) estimation for lithium-ion batteries remains a challenging problem due to complex electrochemical degradation mechanisms and long-range temporal dependencies. In this work, we propose a quantum-enhanced recurrent framework, termed QLSTM, in which variational quantum circuits are directly embedded into the gating mechanisms of long short-term memory networks. By repla...

[View on arXiv](http://arxiv.org/abs/2604.20438v1) | [PDF](https://arxiv.org/pdf/2604.20438v1)

---

### [Coherent-State Propagation: A Computational Framework for Simulating Bosonic Quantum Systems](http://arxiv.org/abs/2604.19625v1)
**Authors:** Nikita Guseynov, Zoë Holmes, Armando Angrisani  
**Published:** 2026-04-21  
**Updated:** 2026-04-21  
**Categories:** quant-ph, cs.CC  

**Abstract:** We introduce coherent-state propagation, a computational framework for simulating bosonic systems. We focus on bosonic circuits composed of displaced linear optics augmented by Kerr nonlinearities, a universal model of bosonic quantum computation that is also physically motivated by driven Bose-Hubbard dynamics. The method works in the Schrödinger picture representing the evolving state as a spars...

[View on arXiv](http://arxiv.org/abs/2604.19625v1) | [PDF](https://arxiv.org/pdf/2604.19625v1)

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
