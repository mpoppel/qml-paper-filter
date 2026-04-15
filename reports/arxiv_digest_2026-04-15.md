# arXiv Daily Digest - 2026-04-15

**Search Period:** Last 7 days  
**Papers Found:** 14

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


### [Q-LINK: Quantum Layerwise Information Residual Network via a Messenger Qubit for Barren Plateaus Mitigation](http://arxiv.org/abs/2604.11831v1)
**Authors:** Zhehao Yi, Rahul Bhadani  
**Published:** 2026-04-11  
**Updated:** 2026-04-11  
**Categories:** quant-ph  

**Abstract:** In hybrid classical-quantum computing, variational quantum algorithms (VQAs) have emerged as a promising approach in the Noisy Intermediate-Scale Quantum (NISQ) era; however, their performance is often hindered by barren plateaus, where gradients vanish exponentially, rendering optimization ineffective. In this work, we introduce a residual-inspired quantum circuit architecture that incorporates a...

[View on arXiv](http://arxiv.org/abs/2604.11831v1) | [PDF](https://arxiv.org/pdf/2604.11831v1)

---

### [Mitigating Barren Plateaus in Variational Quantum Circuits through PDE-Constrained Loss Functions](http://arxiv.org/abs/2604.09957v1)
**Authors:** Prasad Nimantha Madusanka Ukwatta Hewage, Midhun Chakkravarthy, Ruvan Kumara Abeysekara  
**Published:** 2026-04-10  
**Updated:** 2026-04-10  
**Categories:** quant-ph  

**Abstract:** The barren plateau phenomenon; where cost function gradients vanish exponentially with system size; remains a fundamental obstacle to training variational quantum circuits (VQCs) at scale. We demonstrate, both theoretically and numerically, that embedding partial differential equation (PDE) constraints into the VQC loss function provides a natural and effective mitigation mechanism against barren ...

[View on arXiv](http://arxiv.org/abs/2604.09957v1) | [PDF](https://arxiv.org/pdf/2604.09957v1)

---

### [Large Language Models Can Help Mitigate Barren Plateaus in Quantum Neural Networks](http://arxiv.org/abs/2502.13166v3)
**Authors:** Jun Zhuang, Chaowen Guan  
**Published:** 2025-02-17  
**Updated:** 2026-04-12  
**Categories:** quant-ph, cs.AI, cs.CL, cs.LG  

**Abstract:** In the era of noisy intermediate-scale quantum (NISQ) computing, Quantum Neural Networks (QNNs) have emerged as a promising approach for various applications, yet their training is often hindered by barren plateaus (BPs), where gradient variance vanishes exponentially as the qubit size increases. Most initialization-based mitigation strategies rely heavily on pre-designed static parameter distribu...

[View on arXiv](http://arxiv.org/abs/2502.13166v3) | [PDF](https://arxiv.org/pdf/2502.13166v3)

---

### [Frustration-Induced Expressibility Limitations in Variational Quantum Algorithms](http://arxiv.org/abs/2604.11688v1)
**Authors:** Sandip Maiti  
**Published:** 2026-04-13  
**Updated:** 2026-04-13  
**Categories:** quant-ph  

**Abstract:** Geometric frustration, arising from competing interactions that prevent simultaneous energy minimization, presents a fundamental challenge for variational quantum algorithms applied to quantum many-body systems. We investigate the transverse-field Ising model on a square lattice with frustrated diagonal coupling and show that geometric frustration leads to strongly inhomogeneous correlations that ...

[View on arXiv](http://arxiv.org/abs/2604.11688v1) | [PDF](https://arxiv.org/pdf/2604.11688v1)

---

### [Adaptive H-EFT-VA: A Provably Safe Trajectory Through the Trainability-Expressibility Landscape of Variational Quantum Algorithms](http://arxiv.org/abs/2604.10607v1)
**Authors:** Eyad I. B. Hamid  
**Published:** 2026-04-12  
**Updated:** 2026-04-12  
**Categories:** quant-ph, cs.LG, hep-th  

**Abstract:** H-EFT-VA established a physics-informed solution to the Barren Plateau (BP) problem via a hierarchical EFT UV-cutoff, guaranteeing gradient variance in Omega(1/poly(N)). However, localization restricts the ansatz to a polynomial subspace, creating a reference-state gap for states distant from |0>^N. We introduce Adaptive H-EFT-VA (A-H-EFT) to navigate the trainability-expressibility tradeoff by ex...

[View on arXiv](http://arxiv.org/abs/2604.10607v1) | [PDF](https://arxiv.org/pdf/2604.10607v1)

---

### [A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing](http://arxiv.org/abs/2604.07909v1)
**Authors:** Zhirao Wang, Junxiang Huang, Runyu Ye et al.  
**Published:** 2026-04-09  
**Updated:** 2026-04-09  
**Categories:** quant-ph  

**Abstract:** Variational quantum algorithms (VQAs) have established themselves as a central computational paradigm in the Noisy Intermediate-Scale Quantum (NISQ) era. By coupling parameterized quantum circuits (PQCs) with classical optimization, they operate effectively under strict hardware limitations. However, as quantum architectures transition toward early fault-tolerant (EFT) and ultimate fault-tolerant ...

[View on arXiv](http://arxiv.org/abs/2604.07909v1) | [PDF](https://arxiv.org/pdf/2604.07909v1)

---

### [QNAS: A Neural Architecture Search Framework for Accurate and Efficient Quantum Neural Networks](http://arxiv.org/abs/2604.07013v1)
**Authors:** Kooshan Maleki, Alberto Marchisio, Muhammad Shafique  
**Published:** 2026-04-08  
**Updated:** 2026-04-08  
**Categories:** quant-ph, cs.LG  

**Abstract:** Designing quantum neural networks (QNNs) that are both accurate and deployable on NISQ hardware is challenging. Handcrafted ansatze must balance expressivity, trainability, and resource use, while limited qubits often necessitate circuit cutting. Existing quantum architecture search methods primarily optimize accuracy while only heuristically controlling quantum and mostly ignore the exponential o...

[View on arXiv](http://arxiv.org/abs/2604.07013v1) | [PDF](https://arxiv.org/pdf/2604.07013v1)

---

### [Variational Quantum Physics-Informed Neural Networks for Hydrological PDE-Constrained Learning with Inherent Uncertainty Quantification](http://arxiv.org/abs/2604.09374v2)
**Authors:** Prasad Nimantha Madusanka Ukwatta Hewage, Midhun Chakkravarthy, Ruvan Kumara Abeysekara  
**Published:** 2026-04-10  
**Updated:** 2026-04-14  
**Categories:** quant-ph, cs.LG  

**Abstract:** We propose a Hybrid Quantum-Classical Physics-Informed Neural Network (HQC-PINN) that integrates parameterized variational quantum circuits into the PINN framework for hydrological PDE-constrained learning. Our architecture encodes multi-source remote sensing features into quantum states via trainable angle encoding, processes them through a hardware-efficient variational ansatz with entangling la...

[View on arXiv](http://arxiv.org/abs/2604.09374v2) | [PDF](https://arxiv.org/pdf/2604.09374v2)

---

### [Geodesics of Quantum Feature Maps on the Space of Quantum Operators](http://arxiv.org/abs/2509.02795v5)
**Authors:** Andrew Vlasic  
**Published:** 2025-09-02  
**Updated:** 2026-04-13  
**Categories:** quant-ph  

**Abstract:** Recent advancements in the discipline of quantum algorithms have displayed the importance of the geometry of quantum operators. Given this thrust, this paper develops a rigorous geometric framework to analyze how the Riemannian structure of data, under the manifold hypothesis, influences the subspace of quantum gates induced by quantum feature maps. While numerous encoding schemes have been propos...

[View on arXiv](http://arxiv.org/abs/2509.02795v5) | [PDF](https://arxiv.org/pdf/2509.02795v5)

---

### [Leggett-Garg Inequality Violations Bound Quantum Fisher Information](http://arxiv.org/abs/2604.09772v1)
**Authors:** Nick Abboud, Yuntao Guan, Barry Bradlyn et al.  
**Published:** 2026-04-10  
**Updated:** 2026-04-10  
**Categories:** quant-ph, cond-mat.stat-mech, hep-th  

**Abstract:** We prove that a violation of a Leggett-Garg inequality for bounded observables in stationary pure states and thermal states yields a rigorous lower bound on the quantum Fisher information. This turns a qualitative foundations test of realism in quantum systems into a quantitative witness of useful quantum sensitivity and, in the collective setting, into a lower bound on multipartite entanglement d...

[View on arXiv](http://arxiv.org/abs/2604.09772v1) | [PDF](https://arxiv.org/pdf/2604.09772v1)

---

### [Path Integral Approach to Quantum Fisher Information](http://arxiv.org/abs/2604.12763v1)
**Authors:** Francis J. Headley, Mahdi RouhbakhshNabati, Henry Harper-Gardner et al.  
**Published:** 2026-04-14  
**Updated:** 2026-04-14  
**Categories:** quant-ph, cond-mat.stat-mech, hep-th  

**Abstract:** We present a real-time path-integral formulation of the quantum Fisher information for dynamical parameter estimation. For pure states undergoing unitary evolution, we show that the quantum Fisher information can be expressed as a connected symmetrized covariance of a time-integrated action deformation, equivalently as an integrated insertion of $\partial_λS$ in the propagator. This reformulation ...

[View on arXiv](http://arxiv.org/abs/2604.12763v1) | [PDF](https://arxiv.org/pdf/2604.12763v1)

---

### [Hybrid Quantum--Classical k-Means Clustering via Quantum Feature Maps](http://arxiv.org/abs/2604.07873v1)
**Authors:** Syed M. Abdullah, Alisha Baba, Muhammad Siddique et al.  
**Published:** 2026-04-09  
**Updated:** 2026-04-09  
**Categories:** quant-ph  

**Abstract:** Clustering is one of the most fundamental tasks in machine learning, and the k-means clustering algorithm is perhaps one of the most widely used clustering algorithms. However, it suffers from several limitations, such as sensitivity to centroid initialization, difficulty capturing non-linear structure, and poor performance in high-dimensional spaces. Recent work has proposed improved initializati...

[View on arXiv](http://arxiv.org/abs/2604.07873v1) | [PDF](https://arxiv.org/pdf/2604.07873v1)

---

### [A Bundle Isomorphism Relating Complex Velocity to Quantum Fisher Operators](http://arxiv.org/abs/2604.12187v1)
**Authors:** Jorge Meza-Domínguez  
**Published:** 2026-04-14  
**Updated:** 2026-04-14  
**Categories:** quant-ph, cs.IT, gr-qc, math-ph, math.QA  

**Abstract:** We show that averaging matter dynamics over stochastic gravitational fluctuations gives rise to a complex velocity field \(η_μ = π_μ - i u_μ\) living as a section of the pullback bundle \(E = π_{2}^{*}(T^{*}M)\to \mathcal{C}\times M\). We prove that \(η_μ\) is isomorphic, via the Schrödinger representation, to the symmetric logarithmic derivative (SLD) operator \(L_μ\) on the Hilbert space \(\math...

[View on arXiv](http://arxiv.org/abs/2604.12187v1) | [PDF](https://arxiv.org/pdf/2604.12187v1)

---

### [Battery health prognosis using Physics-informed neural network with Quantum Feature mapping](http://arxiv.org/abs/2604.10362v1)
**Authors:** Muhammad Imran Hossain, Md Fazley Rafy, Sarika Khushlani Solanki et al.  
**Published:** 2026-04-11  
**Updated:** 2026-04-11  
**Categories:** cs.LG  

**Abstract:** Accurate battery health prognosis using State of Health (SOH) estimation is essential for the reliability of multi-scale battery energy storage, yet existing methods are limited in generalizability across diverse battery chemistries and operating conditions. The inability of standard neural networks to capture the complex, high-dimensional physics of battery degradation is a major contributor to t...

[View on arXiv](http://arxiv.org/abs/2604.10362v1) | [PDF](https://arxiv.org/pdf/2604.10362v1)

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
