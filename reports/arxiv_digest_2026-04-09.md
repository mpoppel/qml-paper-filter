# arXiv Daily Digest - 2026-04-09

**Search Period:** Last 7 days  
**Papers Found:** 9

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

### [Mitigating the barren plateau problem in linear optics](http://arxiv.org/abs/2510.02430v2)
**Authors:** Matthew D. Horner  
**Published:** 2025-10-02  
**Updated:** 2026-04-07  
**Categories:** quant-ph  

**Abstract:** We prove the existence of barren plateaus in variational quantum algorithms using linear optics with either bosonic or fermionic particles and demonstrate that fermionic linear optics is less susceptible to the barren plateau problem. We use this to motivate a new photonic device, the dual-valued phase shifter, that is a non-linear phase shifter with two distinct eigenvalues. This component result...

[View on arXiv](http://arxiv.org/abs/2510.02430v2) | [PDF](https://arxiv.org/pdf/2510.02430v2)

---

### [QNAS: A Neural Architecture Search Framework for Accurate and Efficient Quantum Neural Networks](http://arxiv.org/abs/2604.07013v1)
**Authors:** Kooshan Maleki, Alberto Marchisio, Muhammad Shafique  
**Published:** 2026-04-08  
**Updated:** 2026-04-08  
**Categories:** quant-ph, cs.LG  

**Abstract:** Designing quantum neural networks (QNNs) that are both accurate and deployable on NISQ hardware is challenging. Handcrafted ansatze must balance expressivity, trainability, and resource use, while limited qubits often necessitate circuit cutting. Existing quantum architecture search methods primarily optimize accuracy while only heuristically controlling quantum and mostly ignore the exponential o...

[View on arXiv](http://arxiv.org/abs/2604.07013v1) | [PDF](https://arxiv.org/pdf/2604.07013v1)

---

### [Hybrid Fourier Neural Operator for Surrogate Modeling of Laser Processing with a Quantum-Circuit Mixer](http://arxiv.org/abs/2604.04828v1)
**Authors:** Mateusz Papierz, Asel Sagingalieva, Alix Benoit et al.  
**Published:** 2026-04-06  
**Updated:** 2026-04-06  
**Categories:** quant-ph, cs.CE, cs.LG, physics.comp-ph  

**Abstract:** Data-driven surrogates can replace expensive multiphysics solvers for parametric PDEs, yet building compact, accurate neural operators for three-dimensional problems remains challenging: in Fourier Neural Operators, dense mode-wise spectral channel mixing scales linearly with the number of retained Fourier modes, inflating parameter counts and limiting real-time deployability. We introduce HQ-LP-F...

[View on arXiv](http://arxiv.org/abs/2604.04828v1) | [PDF](https://arxiv.org/pdf/2604.04828v1)

---

### [Quantum Fisher information matrix via its classical counterpart from random measurements](http://arxiv.org/abs/2509.08196v4)
**Authors:** Jianfeng Lu, Kecen Sha  
**Published:** 2025-09-10  
**Updated:** 2026-04-08  
**Categories:** quant-ph, math-ph, math.OC  

**Abstract:** Preconditioning with the quantum Fisher information matrix (QFIM) is a popular approach in quantum variational algorithms. Yet the QFIM is costly to obtain directly, usually requiring more state preparation than its classical counterpart: the classical Fisher information matrix (CFIM). It is known that averaging the classical Fisher information matrix over Haar-random measurement bases yields $\ma...

[View on arXiv](http://arxiv.org/abs/2509.08196v4) | [PDF](https://arxiv.org/pdf/2509.08196v4)

---

### [Quantum Fisher Information for Entropy of Gibbs States](http://arxiv.org/abs/2603.16456v3)
**Authors:** Francis J. Headley  
**Published:** 2026-03-17  
**Updated:** 2026-04-07  
**Categories:** quant-ph, cond-mat.stat-mech  

**Abstract:** We derive the quantum Fisher information for entropy estimation in a Gibbs state and show that it equals the inverse of the heat capacity, which is dual to the temperature Fisher information given by the heat capacity divided by the square of the temperature. Their product is independent of the Hamiltonian and depends only on the temperature, leading to a metrological uncertainty relation between ...

[View on arXiv](http://arxiv.org/abs/2603.16456v3) | [PDF](https://arxiv.org/pdf/2603.16456v3)

---

### [Recurrent Quantum Feature Maps for Reservoir Computing](http://arxiv.org/abs/2604.03469v1)
**Authors:** Utkarsh Singh, Aaron Z. Goldberg, Christoph Simon et al.  
**Published:** 2026-04-03  
**Updated:** 2026-04-03  
**Categories:** quant-ph, cs.LG  

**Abstract:** Reservoir computing promises a fast method for handling large amounts of temporal data. This hinges on constructing a good reservoir--a dynamical system capable of transforming inputs into a high-dimensional representation while remembering properties of earlier data. In this work, we introduce a reservoir based on recurrent quantum feature maps where a fixed quantum circuit is reused to encode bo...

[View on arXiv](http://arxiv.org/abs/2604.03469v1) | [PDF](https://arxiv.org/pdf/2604.03469v1)

---

### [Shot-Based Quantum Encoding: A Data-Loading Paradigm for Quantum Neural Networks](http://arxiv.org/abs/2604.06135v1)
**Authors:** Basil Kyriacou, Viktoria Patapovich, Maniraman Periyasamy et al.  
**Published:** 2026-04-07  
**Updated:** 2026-04-07  
**Categories:** quant-ph, cs.AI, cs.LG  

**Abstract:** Efficient data loading remains a bottleneck for near-term quantum machine-learning. Existing schemes (angle, amplitude, and basis encoding) either underuse the exponential Hilbert-space capacity or require circuit depths that exceed the coherence budgets of noisy intermediate-scale quantum hardware. We introduce Shot-Based Quantum Encoding (SBQE), a data embedding strategy that distributes the har...

[View on arXiv](http://arxiv.org/abs/2604.06135v1) | [PDF](https://arxiv.org/pdf/2604.06135v1)

---

### [Efficient Learning of Structured Quantum Circuits via Pauli Dimensionality and Sparsity](http://arxiv.org/abs/2510.00168v2)
**Authors:** Sabee Grewal, Daniel Liang  
**Published:** 2025-09-30  
**Updated:** 2026-04-04  
**Categories:** quant-ph, cs.DS  

**Abstract:** We study the problem of efficiently learning an unknown $n$-qubit unitary channel in diamond distance given query access. We present a general framework showing that if Pauli operators remain low-complexity under conjugation by a unitary, then the unitary can be learned efficiently. This framework yields polynomial-time algorithms for a wide range of circuit classes, including $O(\log \log n)$-dep...

[View on arXiv](http://arxiv.org/abs/2510.00168v2) | [PDF](https://arxiv.org/pdf/2510.00168v2)

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
