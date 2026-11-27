# arXiv Daily Digest - 2026-11-27

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


### [How to find expressible and trainable parameterized quantum circuits?](http://arxiv.org/abs/2603.14451v2)
**Authors:** Peter Röseler, Dennis Willsch, Kristel Michielsen  
**Published:** 2026-03-15  
**Updated:** 2026-11-23  
**Categories:** quant-ph, cs.LG  

**Abstract:** Whether parameterized quantum circuits (PQCs) can be systematically constructed to be both trainable and expressive remains an open question. Highly expressive PQCs often exhibit barren plateaus, while several trainable alternatives admit efficient classical simulation. We prove that deciding whether a PQC family exhibits a barren plateau is undecidable in general. However, we derive a practical f...

[View on arXiv](http://arxiv.org/abs/2603.14451v2) | [PDF](https://arxiv.org/pdf/2603.14451v2)

---

### [Observable-Level Expressivity in Fixed-Readout Quantum Neural Networks](http://arxiv.org/abs/2611.30175v1)
**Authors:** Iqra Bashir, Yuanfeng Wang, Mingda Huang et al.  
**Published:** 2026-11-23  
**Updated:** 2026-11-23  
**Categories:** quant-ph  

**Abstract:** A central question in quantum machine learning is which functions of an input state a model can represent when its measurement is fixed by hardware or by experimental design. We address this question for quantum neural networks with a fixed readout observable, where the trainable pre-measurement dynamics determine the effective observable in the Heisenberg picture and the scalar output is the meas...

[View on arXiv](http://arxiv.org/abs/2611.30175v1) | [PDF](https://arxiv.org/pdf/2611.30175v1)

---

### [TQ-HQNet: A Trainability-Aware and Qubit-Efficient Hybrid Quantum Neural Network for Cross-Domain Learning on Heterogeneous Data](http://arxiv.org/abs/2611.31908v1)
**Authors:** Abdur Rasool, Saba Aslam, Hui Chen et al.  
**Published:** 2026-11-24  
**Updated:** 2026-11-24  
**Categories:** cs.LG  

**Abstract:** Hybrid quantum neural networks are a promising direction for noisy intermediate-scale quantum computing, but their practical effectiveness is often limited by poor trainability, rigid circuit design, and weak robustness across heterogeneous real-world tasks. This paper proposes TQ-HQNet, a Trainability-Aware and Qubit-Efficient Hybrid Quantum Neural Network for cross-domain learning on heterogeneo...

[View on arXiv](http://arxiv.org/abs/2611.31908v1) | [PDF](https://arxiv.org/pdf/2611.31908v1)

---

### [Search-for-Gradient: Search-Based Restarts for Optimisation in Structured Flat Landscapes of Variational Quantum Algorithms](http://arxiv.org/abs/2611.34291v1)
**Authors:** Subham Das, Bikash K. Behera, Andrés Camilo Granda Arango et al.  
**Published:** 2026-11-25  
**Updated:** 2026-11-25  
**Categories:** quant-ph  

**Abstract:** Variational Quantum Algorithms (VQAs) are a promising class of hybrid quantum-classical algorithms that involve parameterised quantum circuits and classical optimisation. VQAs have found applications in simulation, optimisation, and machine learning. However, their scalability is hindered by flat landscapes-regions in the parameter space where gradients vanish exponentially small. In such regimes,...

[View on arXiv](http://arxiv.org/abs/2611.34291v1) | [PDF](https://arxiv.org/pdf/2611.34291v1)

---

### [A Frequency-Learning Interpolation-Based Coordinate Descent Method to Optimize Parameterized Quantum Circuits on Quantum Hardware](http://arxiv.org/abs/2611.29019v1)
**Authors:** Jack Mandell, Adam Leicester, Zixu Wang et al.  
**Published:** 2026-11-21  
**Updated:** 2026-11-21  
**Categories:** quant-ph  

**Abstract:** Structure-aware optimizers have been the focus of recent study for the optimization of parameters in variational quantum circuits. Many of these optimizers follow a coordinate-based paradigm, making single parameter updates by using classical approximations to the cost function generated from quantum sampling. However, the performance of structure-aware optimizers on real quantum hardware at large...

[View on arXiv](http://arxiv.org/abs/2611.29019v1) | [PDF](https://arxiv.org/pdf/2611.29019v1)

---

### [Simultaneous computation of multiple excited states via double bracket flows](http://arxiv.org/abs/2611.34871v1)
**Authors:** Reyhaneh Aghaei Saem, Mehrad Sahebi, Zoë Holmes et al.  
**Published:** 2026-11-25  
**Updated:** 2026-11-25  
**Categories:** quant-ph  

**Abstract:** Efficient computation of excited states is essential for studying strongly correlated many-body systems and molecular electronic structure. In this work, we propose a framework for simultaneously simulating multiple excited states based on dynamical flows. The main idea is to exploit a common double bracket flow that drives different pure initial states toward different target excited states witho...

[View on arXiv](http://arxiv.org/abs/2611.34871v1) | [PDF](https://arxiv.org/pdf/2611.34871v1)

---

### [A machine learning architecture for scaling of holographic quantum circuits](http://arxiv.org/abs/2611.31867v1)
**Authors:** Cliff Sun, Michael L. Wall  
**Published:** 2026-11-24  
**Updated:** 2026-11-24  
**Categories:** quant-ph  

**Abstract:** We describe an architecture and workflow for learning and generalizing the syntax and parameters of holographic quantum circuits (HQCs), which are natural quantum circuit analogs of classical tensor networks. An HQC uses mid-circuit measurements and resets on a small subset of "physical" qubits whose measurement outcomes become correlated through coupling with a register of persistent "ancilla" qu...

[View on arXiv](http://arxiv.org/abs/2611.31867v1) | [PDF](https://arxiv.org/pdf/2611.31867v1)

---

### [Dirac Propagation for Simulating Fermionic Observables under Particle-Number-Conserving Unitaries](http://arxiv.org/abs/2611.32902v1)
**Authors:** G Arnav Naidu, Matteo D'Anna, Adrián Pérez-Salinas et al.  
**Published:** 2026-11-24  
**Updated:** 2026-11-24  
**Categories:** quant-ph, cond-mat.str-el  

**Abstract:** Simulating the dynamics of interacting fermions remains a central computational challenge across physics and chemistry. We introduce Dirac propagation, a classical method that simulates local observables in fermionic systems subject to particle-number-conserving unitaries. We show that, in the operator expansion, strings with weights above the total particle number can be discarded without incurri...

[View on arXiv](http://arxiv.org/abs/2611.32902v1) | [PDF](https://arxiv.org/pdf/2611.32902v1)

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
