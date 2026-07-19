# arXiv Daily Digest - 2026-07-19

**Search Period:** Last 7 days  
**Papers Found:** 13

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


### [Overcoming Fourier Locking in Quantum Data Re-uploading Classifiers via Spectral Homotopy](http://arxiv.org/abs/2607.11013v1)
**Authors:** Spencer Topel  
**Published:** 2026-07-13  
**Updated:** 2026-07-13  
**Categories:** quant-ph, cs.LG  

**Abstract:** Data re-uploading parameterized quantum circuits (DRU-PQCs) are universal function approximators, yet their expressivity produces oscillatory, non-convex loss landscapes that resist gradient-based optimization. We show that the primary optimization bottleneck in DRU-PQCs is not insufficient capacity but a structural failure mode we term Fourier locking (FL): because encoding weights and entangling...

[View on arXiv](http://arxiv.org/abs/2607.11013v1) | [PDF](https://arxiv.org/pdf/2607.11013v1)

---

### [An Agentic Formalization for Certified Quantum Neural Network Design](http://arxiv.org/abs/2607.12981v1)
**Authors:** Mingrui Jing, Lei Zhang, Yusheng Zhao et al.  
**Published:** 2026-07-14  
**Updated:** 2026-07-14  
**Categories:** quant-ph  

**Abstract:** A central model in quantum machine learning is the quantum neural network (QNN), whose design requires balancing expressivity and trainability. Technically, expressivity is studied through circuit-function analysis, such as quantum signal processing, while trainability is analyzed using dynamical-Lie-algebra (DLA) methods. To support certified QNN design, we formalize these major components of QNN...

[View on arXiv](http://arxiv.org/abs/2607.12981v1) | [PDF](https://arxiv.org/pdf/2607.12981v1)

---

### [Expressibility and trainability of a two-dimensional pairwise quantum-circuit ansatz](http://arxiv.org/abs/2607.12996v1)
**Authors:** Shuai Zhang, Wei Liu, Ji-Chong Yang  
**Published:** 2026-07-14  
**Updated:** 2026-07-14  
**Categories:** quant-ph  

**Abstract:** Parameterized quantum circuits~(PQCs) constitute a central building block of variational quantum algorithms~(VQAs) and quantum machine learning~(QML) methods. Existing ansatz designs often adopt hardware-agnostic or simplified 1D chain/ring entanglement patterns. However, as quantum hardware continues to develop, native 2D connectivity patterns, such as planar superconducting-qubit architectures, ...

[View on arXiv](http://arxiv.org/abs/2607.12996v1) | [PDF](https://arxiv.org/pdf/2607.12996v1)

---

### [Defeating Barren Plateaus with Task-Aligned Symmetry](http://arxiv.org/abs/2607.12100v1)
**Authors:** Ruipeng Xing, Yanan Li, Zhen Shang et al.  
**Published:** 2026-07-13  
**Updated:** 2026-07-13  
**Categories:** quant-ph  

**Abstract:** Barren plateaus -- the exponential vanishing of gradients -- are a fundamental obstacle to training scalable quantum neural networks. Whether they arise in quantum recurrent neural networks (QRNNs), a natural architecture for sequential data, remains a pressing question. Here we show that the decisive ingredient for trainability in QRNNs is not the recurrent circuit topology per se, but enforcing ...

[View on arXiv](http://arxiv.org/abs/2607.12100v1) | [PDF](https://arxiv.org/pdf/2607.12100v1)

---

### [HarmQ: Harmonic Backdoor Attacks Against Quantum Neural Networks](http://arxiv.org/abs/2607.12055v1)
**Authors:** Junrui Zhang, Zemin Chen, Chunsheng Xin et al.  
**Published:** 2026-07-13  
**Updated:** 2026-07-13  
**Categories:** quant-ph  

**Abstract:** Quantum Neural Networks (QNNs) have emerged as a promising paradigm for quantum machine learning in the Noisy Intermediate-Scale Quantum (NISQ) era, leveraging quantum phenomena such as superposition and entanglement to process information in exponentially large Hilbert spaces. However, QNNs inherit critical security vulnerabilities from classical neural networks, particularly susceptibility to ba...

[View on arXiv](http://arxiv.org/abs/2607.12055v1) | [PDF](https://arxiv.org/pdf/2607.12055v1)

---

### [An architectural capacity ceiling, not a barren plateau: why a fixed-encoding variational quantum circuit cannot fit the Lorenz-63 attractor](http://arxiv.org/abs/2604.23743v2)
**Authors:** Tushar Pandey  
**Published:** 2026-04-26  
**Updated:** 2026-07-15  
**Categories:** quant-ph, cs.LG  

**Abstract:** Variational quantum circuits train poorly on chaotic forecasting, usually blamed on barren plateaus (exponentially vanishing gradients). Using an exactly simulable four-qubit variational quantum physics-informed circuit fit to Lorenz-63, we show the barren-plateau explanation fails: the failure is an architectural capacity ceiling fixed by the circuit time-encoding, not its trainable depth. Four m...

[View on arXiv](http://arxiv.org/abs/2604.23743v2) | [PDF](https://arxiv.org/pdf/2604.23743v2)

---

### [Benchmarking loss functions for trainable quantum feature maps](http://arxiv.org/abs/2607.12487v1)
**Authors:** Nguyen Dinh Quyen, Vu Tuan Hai, Quoc Chuong Nguyen et al.  
**Published:** 2026-07-14  
**Updated:** 2026-07-14  
**Categories:** quant-ph  

**Abstract:** Many quantum machine learning models employ quantum feature maps to encode classical data into quantum states. While fixed feature maps often lack sufficient expressivity for complex nonlinear classification tasks, trainable quantum feature maps (TQFMs) enable adaptive quantum kernels with enhanced learning capability. Different loss functions can induce distinct optimization dynamics, yet their e...

[View on arXiv](http://arxiv.org/abs/2607.12487v1) | [PDF](https://arxiv.org/pdf/2607.12487v1)

---

### [Input-Aware Dynamic Backdoor Attack Against Quantum Neural Networks](http://arxiv.org/abs/2607.11843v1)
**Authors:** Junrui Zhang, Zemin Chen, Lusi Li et al.  
**Published:** 2026-07-13  
**Updated:** 2026-07-13  
**Categories:** quant-ph, cs.LG  

**Abstract:** Quantum Neural Networks (QNNs) are a promising framework for quantum machine learning on near-term quantum devices, but their security risks remain insufficiently understood. Studies have shown that QNNs are vulnerable to backdoor attacks, yet existing quantum backdoors mostly rely on a fixed trigger shared by all poisoned inputs. This fixed-trigger design is a major weakness because many defenses...

[View on arXiv](http://arxiv.org/abs/2607.11843v1) | [PDF](https://arxiv.org/pdf/2607.11843v1)

---

### [Quantum Topological Data Encoding](http://arxiv.org/abs/2607.13847v1)
**Authors:** Adam Wesołowski, Dimitrios Thanos, Daniel Leykam et al.  
**Published:** 2026-07-15  
**Updated:** 2026-07-15  
**Categories:** quant-ph, cs.LG  

**Abstract:** Many datasets encountered across a wide range of domains possess rich geometric and topological structure that is difficult to capture using conventional vector-based representations. Quantum machine learning offers the possibility of processing high-dimensional data in Hilbert spaces, but its practical success depends critically on how classical data is encoded into quantum states. We introduce \...

[View on arXiv](http://arxiv.org/abs/2607.13847v1) | [PDF](https://arxiv.org/pdf/2607.13847v1)

---

### [Ansätz Expressivity and Optimization in Variational Quantum Simulations of Transverse-field Ising Model Across System Sizes](http://arxiv.org/abs/2604.20961v2)
**Authors:** Ashutosh P. Tripathi, Nilmani Mathur, Vikram Tripathi  
**Published:** 2026-04-22  
**Updated:** 2026-07-13  
**Categories:** quant-ph, cond-mat.stat-mech, hep-lat  

**Abstract:** We explore the application of the Variational Quantum Eigensolver (VQE) to investigate the ground state properties, particularly the entanglement entropy, of the Transverse Field Ising Model (TFIM) in one, two, and three dimensions, considering systems of up to 27 spins. By benchmarking VQE results against exact diagonalization and analyzing the entanglement properties across different system size...

[View on arXiv](http://arxiv.org/abs/2604.20961v2) | [PDF](https://arxiv.org/pdf/2604.20961v2)

---

### [Lie-Algebraic Subspace Quantization for Zero-Shot Quantum Learning and Barren-Plateau Mitigation](http://arxiv.org/abs/2607.11174v1)
**Authors:** Yuhan Yao, Yoshihiko Hasegawa  
**Published:** 2026-07-13  
**Updated:** 2026-07-13  
**Categories:** quant-ph  

**Abstract:** The barren plateau phenomenon severely limits the scalability of parameterized quantum circuits (PQCs). We present an analytical framework for zero-shot classical-to-quantum parameter transfer and manifold-based model merging without quantum-side optimization. Our parameter transfer map converts classical neural-network weights into low-dimensional quantum evolutions using Stiefel subspace selecti...

[View on arXiv](http://arxiv.org/abs/2607.11174v1) | [PDF](https://arxiv.org/pdf/2607.11174v1)

---

### [Near-Optimal Mode Scaling for Finite-Dimensional Boson Sampling via Lie-Algebraic Leakage Bounds](http://arxiv.org/abs/2607.11708v1)
**Authors:** Chon-Fai Kam, En-Jui Kuo  
**Published:** 2026-07-13  
**Updated:** 2026-07-13  
**Categories:** quant-ph, math-ph  

**Abstract:** Boson sampling demonstrates quantum advantage through the interference of indistinguishable particles, with output probabilities governed by matrix permanents. Realizing it on deterministic, matter-based platforms requires encoding the bosonic modes in finite-dimensional local Hilbert spaces, which introduces a leakage channel absent in linear optics: multi-particle bunching beyond the local trunc...

[View on arXiv](http://arxiv.org/abs/2607.11708v1) | [PDF](https://arxiv.org/pdf/2607.11708v1)

---

### [A Lie-algebraic approach to non-Markovian quantum dynamics](http://arxiv.org/abs/2607.13865v1)
**Authors:** Haijin Ding, Stephen S. -T. Yau, Zhiwen Zhang  
**Published:** 2026-07-15  
**Updated:** 2026-07-15  
**Categories:** quant-ph  

**Abstract:** In this paper, we study the non-Markovian quantum dynamics in quantum computations from the perspective of a Lie algebraic approach based on numerical analysis. By vectorizing the density matrix of quantum states, the non-Markovian evolutions can be represented with high-dimensional linear time-varying equations, where the time-varying parameters arise from the non-Markovian interactions between t...

[View on arXiv](http://arxiv.org/abs/2607.13865v1) | [PDF](https://arxiv.org/pdf/2607.13865v1)

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
