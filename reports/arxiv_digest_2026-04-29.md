# arXiv Daily Digest - 2026-04-29

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


### [Beyond Single Trajectories: Optimal Control and Jordan-Lie Algebra in Hybrid Quantum Walks for Combinatorial Optimization](http://arxiv.org/abs/2604.25760v1)
**Authors:** Tianen Chen, Yun Shang  
**Published:** 2026-04-28  
**Updated:** 2026-04-28  
**Categories:** quant-ph  

**Abstract:** The Quantum Approximate Optimization Algorithm (QAOA) follows a single, fixed evolution path, overlooking the potential computational advantage of coherently superposing multiple trajectories. Here we overcome this limitation with a hybrid quantum walk (HQW) ansatz that super poses multiple Hamiltonian-driven paths coherently within each circuit layer via a dynamical coin operator. QAOA emerges as...

[View on arXiv](http://arxiv.org/abs/2604.25760v1) | [PDF](https://arxiv.org/pdf/2604.25760v1)

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

### [One Coordinate at a Time: Convergence Guarantees for Rotosolve in Variational Quantum Algorithms](http://arxiv.org/abs/2604.25613v1)
**Authors:** Sayantan Pramanik, M Girish Chandra  
**Published:** 2026-04-28  
**Updated:** 2026-04-28  
**Categories:** quant-ph  

**Abstract:** In this paper, we resolve an open question in the field of optimization algorithms for training parametrized quantum circuits: Does the popular Rotosolve algorithm converge? Until now, interpolation-based coordinate descent methods such as Rotosolve have mostly been treated as heuristics, lacking any formal convergence guarantees. We rigorously analyze Rotosolve, and show that it converges to $\va...

[View on arXiv](http://arxiv.org/abs/2604.25613v1) | [PDF](https://arxiv.org/pdf/2604.25613v1)

---

### [Iterative Quantum Feature Maps](http://arxiv.org/abs/2506.19461v4)
**Authors:** Nasa Matsumoto, Quoc Hoan Tran, Koki Chinzei et al.  
**Published:** 2025-06-24  
**Updated:** 2026-04-28  
**Categories:** quant-ph, cs.AI, stat.ML  

**Abstract:** Quantum machine learning models that leverage quantum circuits as quantum feature maps (QFMs) are recognized for their enhanced expressive power in learning tasks. Such models have demonstrated rigorous end-to-end quantum speedups for specific families of classification problems. However, deploying deep QFMs on real quantum hardware remains challenging due to circuit noise and hardware constraints...

[View on arXiv](http://arxiv.org/abs/2506.19461v4) | [PDF](https://arxiv.org/pdf/2506.19461v4)

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
