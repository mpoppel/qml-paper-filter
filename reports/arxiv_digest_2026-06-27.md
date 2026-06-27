# arXiv Daily Digest - 2026-06-27

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


### [Challenges in Barren Plateau Mitigation with Dynamic Parameterized Quantum Circuits](http://arxiv.org/abs/2606.23751v1)
**Authors:** Sumeet Shirgure, Efekan Kökcü, Siyuan Niu  
**Published:** 2026-06-22  
**Updated:** 2026-06-22  
**Categories:** quant-ph  

**Abstract:** Variational quantum algorithms (VQAs) are a promising paradigm for quantum advantage, yet their trainability is severely hampered by barren plateaus (BPs). Several works have proposed using dynamic parameterized quantum circuits (DPQCs) which intersperse unitary layers with parameterized CPTP maps (e.g. engineered dissipation, feedforward gadgets, or periodic resets), as a potential route around B...

[View on arXiv](http://arxiv.org/abs/2606.23751v1) | [PDF](https://arxiv.org/pdf/2606.23751v1)

---

### [The Cost of Removing Tunability in Quantum Data Re-Uploading](http://arxiv.org/abs/2606.25598v1)
**Authors:** Anthony Yuezhang Liu, Lirandë Pira  
**Published:** 2026-06-24  
**Updated:** 2026-06-24  
**Categories:** quant-ph  

**Abstract:** Fixed encoding data re-uploading quantum circuits provide a striking example of universality emerging from a highly constrained architecture. However, universality alone is insufficient for assessing the theoretical and practical value of fixed and tunable upload circuits. The resource cost of removing tunability remains poorly understood. In this work, we establish quantitative depth-error scalin...

[View on arXiv](http://arxiv.org/abs/2606.25598v1) | [PDF](https://arxiv.org/pdf/2606.25598v1)

---

### [Beyond Single Trajectories: Optimal Control and Jordan-Lie Algebra in Hybrid Quantum Walks for Combinatorial Optimization](http://arxiv.org/abs/2604.25760v2)
**Authors:** Tianen Chen, Yun Shang  
**Published:** 2026-04-28  
**Updated:** 2026-06-25  
**Categories:** quant-ph  

**Abstract:** The Quantum Approximate Optimization Algorithm (QAOA) follows a single, fixed evolution path, overlooking the potential computational advantage of coherently superposing multiple trajectories. Here we overcome this limitation with a hybrid quantum walk (HQW) ansatz that super poses multiple Hamiltonian-driven paths coherently within each circuit layer via a dynamical coin operator. QAOA emerges as...

[View on arXiv](http://arxiv.org/abs/2604.25760v2) | [PDF](https://arxiv.org/pdf/2604.25760v2)

---

### [Mitigating Measurement-Induced Training Instability in Hybrid Quantum Neural Networks for Protein Classification](http://arxiv.org/abs/2606.22551v1)
**Authors:** Milton Mondal, Sushovan Chanda, Mohamad Mahdi Alawieh et al.  
**Published:** 2026-06-21  
**Updated:** 2026-06-21  
**Categories:** cs.LG, cs.CV  

**Abstract:** Hybrid Quantum Neural Network (QNN) classifiers produce logits as expectation values of quantum measurement operators. For standard Pauli measurements, these outputs are intrinsically bounded to the interval [-1,1]. When such bounded logits are used directly with the cross-entropy loss applied to softmax-normalized logits for multi-class classification, the loss function operates in a regime of we...

[View on arXiv](http://arxiv.org/abs/2606.22551v1) | [PDF](https://arxiv.org/pdf/2606.22551v1)

---

### [Particle-preserving fermionic shadows with mode-independent sample complexity](http://arxiv.org/abs/2606.27254v1)
**Authors:** Maxwell West, M. Cerezo, Martin Larocca  
**Published:** 2026-06-25  
**Updated:** 2026-06-25  
**Categories:** quant-ph  

**Abstract:** We consider the problem of learning expectation values of particle-preserving operators with respect to an unknown $η$-particle $n$-mode fermionic state via classical shadows. Our main application is to estimating overlaps with arbitrary Slater determinant states: While it is known that such overlaps can, in the average case, be learnt to a fixed additive precision with a constant number of sample...

[View on arXiv](http://arxiv.org/abs/2606.27254v1) | [PDF](https://arxiv.org/pdf/2606.27254v1)

---

### [Recursive QLSTM with Dynamic Variational Quantum Circuit Adaptation](http://arxiv.org/abs/2606.24932v1)
**Authors:** Samuel Yen-Chi Chen, Yifeng Peng, Jiun-Cheng Jiang et al.  
**Published:** 2026-06-22  
**Updated:** 2026-06-22  
**Categories:** quant-ph, cs.AI, cs.ET, cs.LG, cs.NE  

**Abstract:** Recent advances in quantum computing and machine learning have motivated the development of quantum models for sequential data processing. In this paper, we propose a Recursive Quantum Long Short-Term Memory model, or Recursive QLSTM, which extends QLSTM through metacore-based recursive constructions. We numerically test the model under different input sequence lengths, metacore designs, and recur...

[View on arXiv](http://arxiv.org/abs/2606.24932v1) | [PDF](https://arxiv.org/pdf/2606.24932v1)

---

### [Exact log-depth preparation of highly entangled matrix product states](http://arxiv.org/abs/2606.24475v1)
**Authors:** Keisuke Murota, Frédéric Sauvage, Marco Ballarin et al.  
**Published:** 2026-06-23  
**Updated:** 2026-06-23  
**Categories:** quant-ph  

**Abstract:** Preparing matrix product states (MPS) on a quantum device is a key subroutine in many quantum algorithms. The most competitive methods, based on the renormalisation group, prepare translationally invariant MPS of size $L$ and bond dimension $χ$, up to an error $\varepsilon$, in circuit depth $\tilde O(χ^{4}\log(L/\varepsilon))$ or $\tilde O(χ^{6}\log\log(L/\varepsilon))$. We improve multiple aspec...

[View on arXiv](http://arxiv.org/abs/2606.24475v1) | [PDF](https://arxiv.org/pdf/2606.24475v1)

---

### [Resource-efficient energy-based operator selection in fermionic ADAPT-VQE via exact Hamiltonian transformation](http://arxiv.org/abs/2606.04786v2)
**Authors:** Emanuele Rossi, Erik Rosendahl Kjellgren, Artur F. Izmaylov et al.  
**Published:** 2026-06-03  
**Updated:** 2026-06-20  
**Categories:** quant-ph, physics.chem-ph, physics.comp-ph  

**Abstract:** The energy-based approach to operator selection in ADAPT-VQE relies on reconstructing the one-parameter energy landscape for each operator in the pool. In fermionic implementations, the cost of reconstructing this energy landscape often becomes a bottleneck. We address this issue through an exact Hamiltonian transformation that reformulates the one-parameter energy landscape according to a generat...

[View on arXiv](http://arxiv.org/abs/2606.04786v2) | [PDF](https://arxiv.org/pdf/2606.04786v2)

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
