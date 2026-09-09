# arXiv Daily Digest - 2026-09-09

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


### [A Representation-Theoretic Framework for Characterizing Barren Plateaus](http://arxiv.org/abs/2609.04462v1)
**Authors:** Pedro Alcântara, Leandro Morais, Rafael Chaves  
**Published:** 2026-09-03  
**Updated:** 2026-09-03  
**Categories:** quant-ph  

**Abstract:** The scalability of variational quantum algorithms is fundamentally limited by the barren plateau effect, where the cost-function variance vanishes with system size, rendering optimization impractical. Recent Lie-algebraic approaches for deep parameterized have enabled a unified analytical understanding of this challenge but require either the initial state or the measurement observable to belong t...

[View on arXiv](http://arxiv.org/abs/2609.04462v1) | [PDF](https://arxiv.org/pdf/2609.04462v1)

---

### [Hierarchical Fourier Approximation for Variational Quantum Distribution Learning](http://arxiv.org/abs/2609.06307v1)
**Authors:** Taha Hoseinpour Asli, Sajjad Hashemian, Ebrahim Ardeshir-Larijani  
**Published:** 2026-09-05  
**Updated:** 2026-09-05  
**Categories:** quant-ph, cs.LG  

**Abstract:** We study variational quantum distribution learning through a hierarchy of Walsh--Fourier approximations on the Boolean cube. At each level, a selected set of target Fourier coefficients defines a spectral truncation, which is projected onto the probability simplex and used as the target of a quantum circuit Born machine. Parameters learned at one level initialize the next through a warm-start map....

[View on arXiv](http://arxiv.org/abs/2609.06307v1) | [PDF](https://arxiv.org/pdf/2609.06307v1)

---

### [From Bits to Qubits: The Theory and Practice of Quantum Data Encoding](http://arxiv.org/abs/2609.08058v1)
**Authors:** Xiao-Ming Zhang, Arthur G. Rattew, Bujiao Wu et al.  
**Published:** 2026-09-07  
**Updated:** 2026-09-07  
**Categories:** quant-ph, physics.comp-ph  

**Abstract:** Encoding classical data into quantum systems is a foundational step in the execution of nearly all quantum algorithms, and a critical bottleneck in realizing practical quantum advantage. This review provides a comprehensive account of the concepts, algorithms, and practical considerations associated with quantum data encoding. We trace the development from its early conceptual foundations to recen...

[View on arXiv](http://arxiv.org/abs/2609.08058v1) | [PDF](https://arxiv.org/pdf/2609.08058v1)

---

### [Dynamical Crossover of the Quantum Fisher Information in the Spin-Boson Model](http://arxiv.org/abs/2609.07674v1)
**Authors:** D. Parlato, G. Di Bello, F. Pavan et al.  
**Published:** 2026-09-07  
**Updated:** 2026-09-07  
**Categories:** quant-ph  

**Abstract:** We investigate the dynamical quantum Fisher information of a two-level system coupled to a bosonic environment, focusing on the estimation of the qubit gap. We combine analytical calculations with numerically controlled matrix-product-state simulations. In the exactly solvable pure-dephasing Ohmic regime at zero temperature, the long-time quantum Fisher information displays a coupling-dependent al...

[View on arXiv](http://arxiv.org/abs/2609.07674v1) | [PDF](https://arxiv.org/pdf/2609.07674v1)

---

### [Impact of Data Loss in Postprocessing on Training and Inference of Quantum Neural Networks](http://arxiv.org/abs/2609.05060v1)
**Authors:** Soraya V. Panambalom, Edoardo Altamura, Nick Chancellor et al.  
**Published:** 2026-09-04  
**Updated:** 2026-09-04  
**Categories:** quant-ph, cs.ET, cs.LG, physics.comp-ph  

**Abstract:** As quantum hardware scales to larger devices, the classical software layers that interface with it must evolve in step. Postprocessing routines developed and tested primarily in simulator settings can encode assumptions that no longer hold on utility-scale devices, leading to data loss that can be difficult to detect from high-level model outputs alone. We present a case study of \texttt{SamplerQN...

[View on arXiv](http://arxiv.org/abs/2609.05060v1) | [PDF](https://arxiv.org/pdf/2609.05060v1)

---

### [Tunable topological enhancement of covariant quantum Fisher information via non-Bloch skin effect in non-Hermitian SSH lattices](http://arxiv.org/abs/2609.03421v1)
**Authors:** Qi-Cheng Wu, Yan-Hui Zhou, Tong Liu et al.  
**Published:** 2026-09-03  
**Updated:** 2026-09-03  
**Categories:** quant-ph  

**Abstract:** The covariant quantum Fisher information (CQFI) has recently been established as the ultimate precision benchmark for pseudo-Hermitian sensors [Phys. Rev. Lett. 136, 080802 (2026)], yet existing analyses are limited to single-mode systems. Here we extend the CQFI formalism to multi-mode non-Hermitian Su-Schrieffer-Heeger lattices and reveal tunable topological enhancement enabled by the non-Hermit...

[View on arXiv](http://arxiv.org/abs/2609.03421v1) | [PDF](https://arxiv.org/pdf/2609.03421v1)

---

### [Improved quantum circuits for division](http://arxiv.org/abs/2603.18110v2)
**Authors:** Priyanka Mukhopadhyay, Alexandru Gheorghiu, Hari Krovi  
**Published:** 2026-03-18  
**Updated:** 2026-09-03  
**Categories:** quant-ph  

**Abstract:** Arithmetic operations are an important component of many quantum algorithms. Optimizing quantum circuits for these operations therefore leads to more efficient implementations of the corresponding algorithms. In this paper, we develop new fault-tolerant quantum circuits for various integer division algorithms (both reversible and non-reversible). These circuits, when implemented in the Clifford+T ...

[View on arXiv](http://arxiv.org/abs/2603.18110v2) | [PDF](https://arxiv.org/pdf/2603.18110v2)

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
