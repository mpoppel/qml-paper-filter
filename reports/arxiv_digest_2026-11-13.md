# arXiv Daily Digest - 2026-11-13

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


### [Invariance can prevent overfitting and shortcut learning in Quantum Neural Networks](http://arxiv.org/abs/2611.14368v1)
**Authors:** Giuseppe D'Ambruoso, Giovanni Gramegna, Domenico Pomarico et al.  
**Published:** 2026-11-10  
**Updated:** 2026-11-10  
**Categories:** quant-ph  

**Abstract:** Many image classification tasks have labels that are unchanged by rotations and reflections. Equivariant quantum neural networks incorporate such symmetries by design, but the specific benefit of symmetry is difficult to isolate. Equivariant quantum circuits are usually compared with architectures that also differ in gates, structure or size. Here, we investigate the role of invariance in a contro...

[View on arXiv](http://arxiv.org/abs/2611.14368v1) | [PDF](https://arxiv.org/pdf/2611.14368v1)

---

### [ShardQ: Hardware-Aware Circuit Cutting and Recomposition for Quantum Data Encoding](http://arxiv.org/abs/2511.05492v4)
**Authors:** Ziqing Guo, Jan Balewski, Steven Rayan et al.  
**Published:** 2025-11-07  
**Updated:** 2026-11-07  
**Categories:** quant-ph  

**Abstract:** Quantum data encoding (QDE) loads classical data into quantum states and is a core primitive for quantum machine learning and quantum algorithms, yet executing QDE circuits on connectivity-limited noisy quantum processing units (QPUs) requires a compilation-and-reconstruction pipeline that partitions the circuit across the hardware and recombines the fragment outputs. Circuit cutting and knitting ...

[View on arXiv](http://arxiv.org/abs/2511.05492v4) | [PDF](https://arxiv.org/pdf/2511.05492v4)

---

### [A Unified Benchmarking Study of Variational Quantum Eigensolver Ansatz for Electronic Structure Problems](http://arxiv.org/abs/2611.15365v1)
**Authors:** Julie Maria Raju, Abhoy Kole, Julen Larrucea et al.  
**Published:** 2026-11-11  
**Updated:** 2026-11-11  
**Categories:** quant-ph  

**Abstract:** Electronic structure calculations become computationally expensive on classical computers as electron correlation increases. Quantum computing offers a promising alternative through hybrid quantum-classical Variational Quantum Eigensolver (VQE) on Noisy-Intermediate Scale Quantum (NISQ) devices. The choice of ansatz plays a pivotal role in the performance and scalability of the VQE, yet selecting ...

[View on arXiv](http://arxiv.org/abs/2611.15365v1) | [PDF](https://arxiv.org/pdf/2611.15365v1)

---

### [Delta-Corrected Shot Reduction for Bypassing Optimization Plateaus in Variational Quantum Eigensolvers](http://arxiv.org/abs/2611.10524v1)
**Authors:** Jamie Evander Ghozaly, Pekik Nurwantoro, Gagus Ketut Sunnardianto et al.  
**Published:** 2026-11-07  
**Updated:** 2026-11-07  
**Categories:** quant-ph, physics.chem-ph  

**Abstract:** The variational quantum eigensolver (VQE) remains a cornerstone hybrid quantum-classical algorithm for quantum simulation, yet its scalability is fundamentally constrained by heavy measurement overheads. While strategies such as variance-preserved shot reduction (VPSR) mitigates this cost, they suffer from convergence instability and optimization stagnation caused by finite-sample bias in limited ...

[View on arXiv](http://arxiv.org/abs/2611.10524v1) | [PDF](https://arxiv.org/pdf/2611.10524v1)

---

### [Alchemically Transferable Parameters for Variational Quantum Eigensolvers](http://arxiv.org/abs/2611.11445v1)
**Authors:** Jonas Motyl, Jakob S. Kottmann  
**Published:** 2026-11-08  
**Updated:** 2026-11-08  
**Categories:** quant-ph, cs.LG, physics.chem-ph  

**Abstract:** Machine-learned circuit parameters could remove one of the main bottlenecks of variational quantum eigensolvers: the classical optimization loop, with its typical barren plateaus, local minima, and costly energy and gradient evaluations, is replaced by a direct prediction. Prior work showed that machine-learning models paired with a structured circuit design can transfer across system size and str...

[View on arXiv](http://arxiv.org/abs/2611.11445v1) | [PDF](https://arxiv.org/pdf/2611.11445v1)

---

### [Phase Interference and Transverse Catalysts Govern the Noise Fragility of Long-Range Variational Quantum Circuits](http://arxiv.org/abs/2611.16115v1)
**Authors:** Longhui Shen, Fangzhou Jin, Xiaoxia Yi et al.  
**Published:** 2026-11-11  
**Updated:** 2026-11-11  
**Categories:** quant-ph  

**Abstract:** Long-range entangling gates shorten variational state preparation, but they also accelerate operator spreading across the register, linking nonlocality to a presumed noise fragility. Here we show that the single-fault fragility $F$ of a variational circuit is governed by how fault phases interfere on the target Hamiltonian, not by how far faults spread. For Heisenberg chains we prove that two faul...

[View on arXiv](http://arxiv.org/abs/2611.16115v1) | [PDF](https://arxiv.org/pdf/2611.16115v1)

---

### [Multi-head Quantum Neural Networks: Replacing Transformer Blocks with NISQ Modules inside a 30-Billion-Parameter LLM](http://arxiv.org/abs/2611.15250v1)
**Authors:** Maforikan Amoussou, Borja Aizpurua, Sukhbinder Singh et al.  
**Published:** 2026-11-11  
**Updated:** 2026-11-11  
**Categories:** quant-ph, cs.AI, cs.LG  

**Abstract:** We introduce the multi-head quantum neural network (QNN), a quantum module compatible with noisy intermediate-scale quantum (NISQ) devices that can replace transformer blocks inside pre-trained large language models (LLMs) with little loss in accuracy and with far fewer parameters than the blocks it replaces. Encoding a model's $D$ hidden neurons into a qubit register is the obstacle: amplitude en...

[View on arXiv](http://arxiv.org/abs/2611.15250v1) | [PDF](https://arxiv.org/pdf/2611.15250v1)

---

### [Training quantum neural networks one job at a time](http://arxiv.org/abs/2611.16365v1)
**Authors:** Muhammad Faryad, Rabab Alomairy  
**Published:** 2026-11-12  
**Updated:** 2026-11-12  
**Categories:** quant-ph  

**Abstract:** When a quantum processor is reached through a cloud queue, a variational algorithm pays for every round trip, and a single round trip can carry many circuits. The classical optimizer therefore decides how much is learned per job, not only per shot. We study this for a quantum neural network classifier. We compare optimizers that are given the same number of jobs and the same shots per job, and tha...

[View on arXiv](http://arxiv.org/abs/2611.16365v1) | [PDF](https://arxiv.org/pdf/2611.16365v1)

---

### [Exact Quantum Circuit Optimization is co-NQP-hard](http://arxiv.org/abs/2510.16420v4)
**Authors:** Adam Husted Kjelstrøm, Andreas Pavlogiannis, Jaco van de Pol  
**Published:** 2025-10-18  
**Updated:** 2026-11-07  
**Categories:** quant-ph, cs.CC  

**Abstract:** As quantum computing resources remain scarce and error rates high, minimizing the resource consumption of quantum circuits is essential for achieving practical quantum advantage. Here we consider the natural problem of, given a circuit $C$, computing a circuit $C'$ that behaves equivalently on a desired subspace, and that minimizes a quantum resource type, expressed as the count or depth of (i) al...

[View on arXiv](http://arxiv.org/abs/2510.16420v4) | [PDF](https://arxiv.org/pdf/2510.16420v4)

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
