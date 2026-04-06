# arXiv Daily Digest - 2026-04-06

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


### [LieTrunc-QNN: Lie Algebra Truncation and Quantum Expressivity Phase Transition from LiePrune to Provably Stable Quantum Neural Networks](http://arxiv.org/abs/2604.02697v1)
**Authors:** Haijian Shao, Dalong Zhao, Xing Deng et al.  
**Published:** 2026-04-03  
**Updated:** 2026-04-03  
**Categories:** cs.LG  

**Abstract:** Quantum Machine Learning (QML) is fundamentally limited by two challenges: barren plateaus (exponentially vanishing gradients) and the fragility of parameterized quantum circuits under noise. Despite extensive empirical studies, a unified theoretical framework remains lacking.   We introduce LieTrunc-QNN, an algebraic-geometric framework that characterizes trainability via Lie-generated dynamics. ...

[View on arXiv](http://arxiv.org/abs/2604.02697v1) | [PDF](https://arxiv.org/pdf/2604.02697v1)

---

### [Classical shadows with arbitrary group representations](http://arxiv.org/abs/2604.01429v1)
**Authors:** Maxwell West, Frederic Sauvage, Aniruddha Sen et al.  
**Published:** 2026-04-01  
**Updated:** 2026-04-01  
**Categories:** quant-ph  

**Abstract:** Classical shadows (CS) has recently emerged as an important framework to efficiently predict properties of an unknown quantum state. A common strategy in CS protocols is to parametrize the basis in which one measures the state by a random group action; many examples of this have been proposed and studied on a case-by-case basis. In this work, we present a unified theory that allows us to simultane...

[View on arXiv](http://arxiv.org/abs/2604.01429v1) | [PDF](https://arxiv.org/pdf/2604.01429v1)

---

### [Quantum machine learning for the quantum lattice Boltzmann method: Trainability of variational quantum circuits for the nonlinear collision operator across multiple time steps](http://arxiv.org/abs/2604.00620v1)
**Authors:** Antonio David Bastida Zamora, Ljubomir Budinski, Pierre Sagaut et al.  
**Published:** 2026-04-01  
**Updated:** 2026-04-01  
**Categories:** quant-ph, physics.flu-dyn  

**Abstract:** This study investigates the application of quantum machine learning (QML) to approximate the nonlinear component of the collision operator within the quantum lattice Boltzmann method (QLBM). To achieve this, we train a variational quantum circuit (VQC) to construct an operator $U$. When applied to the post-linear-collision quantum state $\ket{Ψ_i}$, this operator yields a final state $\ket{Ψ_f} = ...

[View on arXiv](http://arxiv.org/abs/2604.00620v1) | [PDF](https://arxiv.org/pdf/2604.00620v1)

---

### [Geodesics of Quantum Feature Maps on the Space of Quantum Operators](http://arxiv.org/abs/2509.02795v4)
**Authors:** Andrew Vlasic  
**Published:** 2025-09-02  
**Updated:** 2026-03-31  
**Categories:** quant-ph  

**Abstract:** Recent advancements in the discipline of quantum algorithms have displayed the importance of the geometry of quantum operators. Given this thrust, this paper develops a rigorous geometric framework to analyze how the Riemannian structure of data, under the manifold hypothesis, influences the subspace of quantum gates induced by quantum feature maps. While numerous encoding schemes have been propos...

[View on arXiv](http://arxiv.org/abs/2509.02795v4) | [PDF](https://arxiv.org/pdf/2509.02795v4)

---

### [Codimension-controlled universality of quantum Fisher information singularities at topological band-touching defects](http://arxiv.org/abs/2604.01515v1)
**Authors:** C. A. S. Almeida  
**Published:** 2026-04-02  
**Updated:** 2026-04-02  
**Categories:** quant-ph, cond-mat.mes-hall  

**Abstract:** Topological phase transitions in generic multiband systems are mediated by band-touching defects whose codimension -- the number of momentum directions along which the gap closes linearly -- varies across universality classes. Although singular behavior of fidelity susceptibilities and quantum Fisher information (QFI) has been computed for specific models, no unifying principle connecting these re...

[View on arXiv](http://arxiv.org/abs/2604.01515v1) | [PDF](https://arxiv.org/pdf/2604.01515v1)

---

### [Calculating the quantum Fisher information via the truncated Wigner method](http://arxiv.org/abs/2603.29196v1)
**Authors:** Thakur G. M. Hiranandani, Joseph J. Hope, Simon A. Haine  
**Published:** 2026-03-31  
**Updated:** 2026-03-31  
**Categories:** quant-ph  

**Abstract:** In this work, we propose new methods of parameter estimation using stochastic sampling quantum phase-space simulations. We show that it is possible to compute the quantum Fisher information (QFI) from semiclassical stochastic samples using the Truncated Wigner Approximation (TWA). This method extends the class of quantum systems whose fundamental sensitivity limit can be computed efficiently to an...

[View on arXiv](http://arxiv.org/abs/2603.29196v1) | [PDF](https://arxiv.org/pdf/2603.29196v1)

---

### [Quantum Fisher information in many-photon states from shift current shot noise](http://arxiv.org/abs/2603.29188v1)
**Authors:** Evgenii Barts, Takahiro Morimoto, Naoto Nagaosa  
**Published:** 2026-03-31  
**Updated:** 2026-03-31  
**Categories:** cond-mat.mes-hall, quant-ph  

**Abstract:** Quantum Fisher information (QFI) sets the ultimate precision of optical phase measurements and reveals multiphoton entanglement, but it is not accessible with conventional photodetection. We theoretically predict that a photodetector utilizing the shot noise of the quantum-geometric shift current of exciton polaritons can directly measure the QFI of nonclassical light. By solving the Lindblad equa...

[View on arXiv](http://arxiv.org/abs/2603.29188v1) | [PDF](https://arxiv.org/pdf/2603.29188v1)

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
