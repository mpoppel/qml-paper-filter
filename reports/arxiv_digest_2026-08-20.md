# arXiv Daily Digest - 2026-08-20

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


### [From coherence to mixedness: a driver of barren plateaus in variational quantum algorithms](http://arxiv.org/abs/2511.22350v5)
**Authors:** Xiang Zhou  
**Published:** 2025-11-27  
**Updated:** 2026-08-15  
**Categories:** quant-ph, math-ph  

**Abstract:** Variational quantum algorithms (VQAs) are a leading approach for near-term quantum advantage. However, their training is often hindered by barren plateaus (BPs). We present a framework based on observational entropy. The framework separates the coherent part of a quantum state from its incoherent part. We define the coherence fraction $η$ as the ratio of coherent to total contribution. This quanti...

[View on arXiv](http://arxiv.org/abs/2511.22350v5) | [PDF](https://arxiv.org/pdf/2511.22350v5)

---

### [Qudit-ADAPT-VQE: an adaptive variational algorithm with counterdiabatic-inspired improvements for qudits](http://arxiv.org/abs/2608.14981v1)
**Authors:** Joaquín Molina, Herbert Díaz-Moraga, Dardo Goyeneche et al.  
**Published:** 2026-08-15  
**Updated:** 2026-08-15  
**Categories:** quant-ph  

**Abstract:** Variational quantum algorithms based on qudits have attracted significant attention in recent years. However, as in their qubit-based counterparts, challenges such as barren plateaus and the design of efficient ansatz remain major obstacles. In this work, we propose to address these issues through a qudit implementation of the ADAPT-VQE algorithm, which constructs the ansatz iteratively. Specifica...

[View on arXiv](http://arxiv.org/abs/2608.14981v1) | [PDF](https://arxiv.org/pdf/2608.14981v1)

---

### [Lie-Algebraic Classical Simulation of Bosonic Systems Beyond Gaussian Dynamics](http://arxiv.org/abs/2608.17094v1)
**Authors:** Adelina Bärligea, Timothy Heightman, Jakob S. Kottmann et al.  
**Published:** 2026-08-17  
**Updated:** 2026-08-17  
**Categories:** quant-ph, math-ph  

**Abstract:** Classical simulability is ultimately determined by both the dynamics of a quantum system and the observables being evaluated. Lie-algebraic simulation exploits the latter to make exact polynomial-time classical simulations by propagating observables through low-dimensional invariant operator spaces. However, its conventional formulation in terms of polynomial-dimensional dynamical Lie algebras doe...

[View on arXiv](http://arxiv.org/abs/2608.17094v1) | [PDF](https://arxiv.org/pdf/2608.17094v1)

---

### [Ambient unitaries don't enable shallow group designs](http://arxiv.org/abs/2608.13528v1)
**Authors:** Maxwell West, M. Cerezo, Martin Larocca  
**Published:** 2026-08-13  
**Updated:** 2026-08-13  
**Categories:** quant-ph  

**Abstract:** Characterising the efficiency with which designs over various subsets of the unitary group may be constructed is an important goal of quantum information theory. While it is now known that approximate unitary designs can be realised in depth logarithmic in the system size, it has recently been shown that ensembles of local nearest-neighbour sublinear-depth one-dimensional circuits over the matchga...

[View on arXiv](http://arxiv.org/abs/2608.13528v1) | [PDF](https://arxiv.org/pdf/2608.13528v1)

---

### [Noise Resilience of Quantum Support Vector Machine with Selected Feature Maps](http://arxiv.org/abs/2608.17495v1)
**Authors:** Muhammad Ahsan Shakeel, Saad Muzammil, Danyal Tayyub et al.  
**Published:** 2026-08-18  
**Updated:** 2026-08-18  
**Categories:** quant-ph  

**Abstract:** Gate-level noise degrades the classification accuracy of Quantum Support Vector Machines (QSVMs) on Noisy Intermediate-Scale Quantum (NISQ) hardware, and the degree of degradation depends on how classical data is encoded into quantum states. We tested Z, ZZ, a Pauli, and an amplitude-inspired feature maps under depolarizing, bit-flip, and phase-flip noise channels in $52$ controlled experiments wi...

[View on arXiv](http://arxiv.org/abs/2608.17495v1) | [PDF](https://arxiv.org/pdf/2608.17495v1)

---

### [Unbiased Hamiltonian Simulation by Reversing Trotter Error Dynamics](http://arxiv.org/abs/2606.29741v2)
**Authors:** Keisuke Murota, Yuta Kikuchi, Enrico Rinaldi et al.  
**Published:** 2026-06-29  
**Updated:** 2026-08-17  
**Categories:** quant-ph  

**Abstract:** Owing to their simplicity and low overhead, Suzuki-Trotter formulas remain the de facto Hamiltonian simulation methods on current quantum computing platforms. Systematic Trotter errors, however, will quickly become limiting when scaling to larger problems and aiming for higher accuracy. We present a mechanism that removes the systematic error of any $k$-th order Suzuki-Trotter simulation, at the c...

[View on arXiv](http://arxiv.org/abs/2606.29741v2) | [PDF](https://arxiv.org/pdf/2606.29741v2)

---

### [Automating Variational Quantum Sensing through Reinforcement-Learned Circuit Structures](http://arxiv.org/abs/2608.17582v1)
**Authors:** Jie Liu, Xin Wang  
**Published:** 2026-08-18  
**Updated:** 2026-08-18  
**Categories:** quant-ph, cond-mat.dis-nn  

**Abstract:** Variational quantum sensing offers a promising route to high-precision parameter estimation, but its performance depends strongly on the circuit architectures used for probe preparation and measurement. Existing approaches typically optimize continuous parameters within predefined ansätze, restricting the accessible design space and limiting adaptation to sensing tasks and hardware constraints. He...

[View on arXiv](http://arxiv.org/abs/2608.17582v1) | [PDF](https://arxiv.org/pdf/2608.17582v1)

---

### [Parallelizable Exact Synthesis of Quantum Circuits via Semi-Tensor Product](http://arxiv.org/abs/2607.24195v2)
**Authors:** Chenjian Li, Dingchao Gao, Xiangzhen Zhou et al.  
**Published:** 2026-07-27  
**Updated:** 2026-08-16  
**Categories:** quant-ph, cs.ET  

**Abstract:** Exact synthesis is a key infrastructure in quantum circuit synthesis and optimization, which provides optimal implementations of small circuit shards and is widely used as a circuit re-synthesis optimization kernel. However, existing quantum exact synthesis methods suffer from encoding overhead, memory bottlenecks, and poor parallel scalability. In this work, we introduce a parallel exact synthesi...

[View on arXiv](http://arxiv.org/abs/2607.24195v2) | [PDF](https://arxiv.org/pdf/2607.24195v2)

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
