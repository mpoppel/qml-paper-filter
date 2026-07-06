# arXiv Daily Digest - 2026-07-06

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


### [Beyond the Expressivity-Trainability Paradox: A Dynamical Lie Algebra Perspective on Navigating Barren Plateaus in Quantum Machine Learning](http://arxiv.org/abs/2606.31536v3)
**Authors:** Kung-Ming Lan, Edward Huang  
**Published:** 2026-06-30  
**Updated:** 2026-07-02  
**Categories:** cs.LG, quant-ph  

**Abstract:** As Quantum Machine Learning (QML) transitions toward practical implementation, the field faces a critical architectural bottleneck that challenges the fundamental assumptions of classical statistical learning theory. In classical deep learning, increasing model capacity typically risks overfitting. However, this study advances a counter-intuitive paradigm: unstructured contemporary QML architectur...

[View on arXiv](http://arxiv.org/abs/2606.31536v3) | [PDF](https://arxiv.org/pdf/2606.31536v3)

---

### [Overcoming Barren Plateaus in Variational Quantum Circuits using a Two-Step Least Squares Approach](http://arxiv.org/abs/2601.18060v4)
**Authors:** Francis Boabang, Samuel Asante Gyamerah  
**Published:** 2026-01-26  
**Updated:** 2026-06-30  
**Categories:** quant-ph, cs.IT  

**Abstract:** Variational Quantum Algorithms are a vital part of quantum computing. It is a blend of quantum and classical methods for tackling tough problems in machine learning, chemistry, and combinatorial optimization. Yet as these algorithms scale up, they cannot escape the barren-plateau phenomenon. As systems grow, gradients can vanish so quickly that training deep or randomly initialized circuits become...

[View on arXiv](http://arxiv.org/abs/2601.18060v4) | [PDF](https://arxiv.org/pdf/2601.18060v4)

---

### [The Dynamical Lie Algebra of QAOA-MaxCut on the Complete Graph](http://arxiv.org/abs/2607.00945v1)
**Authors:** Jonathan Allcock, Pei Yuan, Shengyu Zhang  
**Published:** 2026-07-01  
**Updated:** 2026-07-01  
**Categories:** quant-ph  

**Abstract:** We give an analytical expression for the dynamical Lie algebra corresponding to the QAOA-MaxCut problem on complete graphs, and show that the variance of the associated loss function scales linearly in the number of qubits. This solves an open problem from [ASYZ26] and confirms that such systems do not exhibit barren plateaus. The proof is based on projecting the dynamical Lie algebra generators o...

[View on arXiv](http://arxiv.org/abs/2607.00945v1) | [PDF](https://arxiv.org/pdf/2607.00945v1)

---

### [Quantum machine learning models for graphs](http://arxiv.org/abs/2607.00698v1)
**Authors:** Frédéric Sauvage, Pranav Kalidindi, Frederic Rapp et al.  
**Published:** 2026-07-01  
**Updated:** 2026-07-01  
**Categories:** quant-ph  

**Abstract:** Geometric Machine Learning (GML) successes have been achieved through the thorough study and design of new equivariant neural networks. In comparison, geometric quantum machine learning (GQML) models lack such a detailed understanding and, despite already several proposals, a unifying perspective on their design remains elusive. In this work, we focus on GQML models for graph problems that showcas...

[View on arXiv](http://arxiv.org/abs/2607.00698v1) | [PDF](https://arxiv.org/pdf/2607.00698v1)

---

### [Quantum circuit design via dynamic Pauli constraints](http://arxiv.org/abs/2605.22744v2)
**Authors:** James R. Wootton, Merlin Incerti-Medici, Daniel Bultrini et al.  
**Published:** 2026-05-21  
**Updated:** 2026-07-01  
**Categories:** quant-ph  

**Abstract:** We introduce the Motte model, a software-oriented model of quantum computation motivated by the practical constraints of near-term quantum hardware. In this model, gates are specified by constraints expressed in terms of Pauli observables, with each disjoint layer of gates accompanied by a pairwise or k-local quantum state tomography of the device. We prove that the model is equivalent to the coup...

[View on arXiv](http://arxiv.org/abs/2605.22744v2) | [PDF](https://arxiv.org/pdf/2605.22744v2)

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
