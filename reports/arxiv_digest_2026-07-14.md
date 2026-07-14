# arXiv Daily Digest - 2026-07-14

**Search Period:** Last 7 days  
**Papers Found:** 5

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

### [Grokking and epoch-wise double descent in quantum neural networks](http://arxiv.org/abs/2607.08350v1)
**Authors:** Daniel Pranjić, Marco Roth, Christian Tutschku  
**Published:** 2026-07-09  
**Updated:** 2026-07-09  
**Categories:** quant-ph, physics.data-an  

**Abstract:** Grokking, the delayed transition from memorization to generalization, is a fundamental phenomenon in gradient-based learning, yet its dynamics within variational quantum machine learning (QML) remain largely unexamined. In this work, we report the empirical observation of both the grokking transition and epoch-wise double descent in a two-qubit quantum neural network (QNN) under a complete paramet...

[View on arXiv](http://arxiv.org/abs/2607.08350v1) | [PDF](https://arxiv.org/pdf/2607.08350v1)

---

### [Input-Aware Dynamic Backdoor Attack Against Quantum Neural Networks](http://arxiv.org/abs/2607.11843v1)
**Authors:** Junrui Zhang, Zemin Chen, Lusi Li et al.  
**Published:** 2026-07-13  
**Updated:** 2026-07-13  
**Categories:** quant-ph, cs.LG  

**Abstract:** Quantum Neural Networks (QNNs) are a promising framework for quantum machine learning on near-term quantum devices, but their security risks remain insufficiently understood. Studies have shown that QNNs are vulnerable to backdoor attacks, yet existing quantum backdoors mostly rely on a fixed trigger shared by all poisoned inputs. This fixed-trigger design is a major weakness because many defenses...

[View on arXiv](http://arxiv.org/abs/2607.11843v1) | [PDF](https://arxiv.org/pdf/2607.11843v1)

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
