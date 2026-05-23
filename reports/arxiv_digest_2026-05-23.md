# arXiv Daily Digest - 2026-05-23

**Search Period:** Last 7 days  
**Papers Found:** 4

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


### [Discovering Data Encoding Strategies for Quantum-Classical Neural Networks Using Monte Carlo Tree Search](http://arxiv.org/abs/2605.18540v1)
**Authors:** Lena Tokuhiro, Amine Bentellis, Jeanette Miriam Lorenz  
**Published:** 2026-05-18  
**Updated:** 2026-05-18  
**Categories:** quant-ph  

**Abstract:** Quantum machine learning (QML) has attracted considerable research interest, yet whether it offers practical benefits over classical approaches remains an open question. The choice of data encoding significantly influences QML performance, but why certain encodings outperform others remains poorly understood. We employ Monte Carlo Tree Search (MCTS) to discover optimal data encoding circuits for a...

[View on arXiv](http://arxiv.org/abs/2605.18540v1) | [PDF](https://arxiv.org/pdf/2605.18540v1)

---

### [Lie-algebraic incompleteness of symmetry-adapted VQE for non-Abelian molecular point groups](http://arxiv.org/abs/2603.21009v2)
**Authors:** Leon D. da Silva, Marcelo P. Santos  
**Published:** 2026-03-22  
**Updated:** 2026-05-19  
**Categories:** quant-ph, math.RT  

**Abstract:** Symmetry-adapted variational quantum eigensolvers (VQE) based on the Unitary Coupled-Cluster ansatz (SymUCCSD) effectively reduce the parameter count for Abelian molecular point groups. For non-Abelian groups, they systematically fail, without a theoretical explanation. In this work, we prove that the Abelian-subgroup restriction induces a spurious splitting of multidimensional irreducible represe...

[View on arXiv](http://arxiv.org/abs/2603.21009v2) | [PDF](https://arxiv.org/pdf/2603.21009v2)

---

### [Precision and Privacy in Distributed Quantum Sensing: A Quantum Fisher Information Duality](http://arxiv.org/abs/2605.20765v1)
**Authors:** Farhad Farokhi  
**Published:** 2026-05-20  
**Updated:** 2026-05-20  
**Categories:** quant-ph, cs.CR, cs.IT  

**Abstract:** We establish a quantum Fisher information (QFI) duality for distributed quantum sensor networks with local phase encoding. For any $N$-qubit probe state, where $N$ denotes the number of sensors, $F_Q(\boldsymbol{w}^\top \boldsymbolθ) + F_Q(\boldsymbol{v}^\top \boldsymbolθ) \leq N$ for all unit orthogonal sensing directions $\boldsymbol{w}$ and $\boldsymbol{v}$, with equality for all equatorial sta...

[View on arXiv](http://arxiv.org/abs/2605.20765v1) | [PDF](https://arxiv.org/pdf/2605.20765v1)

---

### [Quantum circuit design via dynamic Pauli constraints](http://arxiv.org/abs/2605.22744v1)
**Authors:** James R. Wootton, Merlin Incerti-Medici, Daniel Bultrini et al.  
**Published:** 2026-05-21  
**Updated:** 2026-05-21  
**Categories:** quant-ph  

**Abstract:** We introduce a novel software-oriented model of quantum computation motivated by the practical constraints of near-term quantum hardware. In this model, gates are specified by constraints expressed in terms of Pauli observables, with each disjoint layer of gates accompanied by a pairwise or $k$-local quantum state tomography of the device. We prove that the model is equivalent to the coupling-grap...

[View on arXiv](http://arxiv.org/abs/2605.22744v1) | [PDF](https://arxiv.org/pdf/2605.22744v1)

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
