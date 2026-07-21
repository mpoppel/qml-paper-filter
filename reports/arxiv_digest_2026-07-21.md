# arXiv Daily Digest - 2026-07-21

**Search Period:** Last 7 days  
**Papers Found:** 12

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


### [An Agentic Formalization for Certified Quantum Neural Network Design](http://arxiv.org/abs/2607.12981v1)
**Authors:** Mingrui Jing, Lei Zhang, Yusheng Zhao et al.  
**Published:** 2026-07-14  
**Updated:** 2026-07-14  
**Categories:** quant-ph  

**Abstract:** A central model in quantum machine learning is the quantum neural network (QNN), whose design requires balancing expressivity and trainability. Technically, expressivity is studied through circuit-function analysis, such as quantum signal processing, while trainability is analyzed using dynamical-Lie-algebra (DLA) methods. To support certified QNN design, we formalize these major components of QNN...

[View on arXiv](http://arxiv.org/abs/2607.12981v1) | [PDF](https://arxiv.org/pdf/2607.12981v1)

---

### [Lie-Group Mode Connectivity in Quantum Machine Learning from a Dynamical Lie Algebra Perspective](http://arxiv.org/abs/2607.17554v1)
**Authors:** Hiroshi Ohno  
**Published:** 2026-07-20  
**Updated:** 2026-07-20  
**Categories:** quant-ph  

**Abstract:** Mode connectivity has been widely studied in classical machine learning as a geometric property of low-loss regions in parameter space. In quantum machine learning (QML), however, the physically relevant object is not the parameter vector itself but the unitary transformation implemented by a parameterized quantum circuit. In this study, we formulate mode connectivity on the reachable unitary Lie ...

[View on arXiv](http://arxiv.org/abs/2607.17554v1) | [PDF](https://arxiv.org/pdf/2607.17554v1)

---

### [Expressibility and trainability of a two-dimensional pairwise quantum-circuit ansatz](http://arxiv.org/abs/2607.12996v1)
**Authors:** Shuai Zhang, Wei Liu, Ji-Chong Yang  
**Published:** 2026-07-14  
**Updated:** 2026-07-14  
**Categories:** quant-ph  

**Abstract:** Parameterized quantum circuits~(PQCs) constitute a central building block of variational quantum algorithms~(VQAs) and quantum machine learning~(QML) methods. Existing ansatz designs often adopt hardware-agnostic or simplified 1D chain/ring entanglement patterns. However, as quantum hardware continues to develop, native 2D connectivity patterns, such as planar superconducting-qubit architectures, ...

[View on arXiv](http://arxiv.org/abs/2607.12996v1) | [PDF](https://arxiv.org/pdf/2607.12996v1)

---

### [An architectural capacity ceiling, not a barren plateau: why a fixed-encoding variational quantum circuit cannot fit the Lorenz-63 attractor](http://arxiv.org/abs/2604.23743v2)
**Authors:** Tushar Pandey  
**Published:** 2026-04-26  
**Updated:** 2026-07-15  
**Categories:** quant-ph, cs.LG  

**Abstract:** Variational quantum circuits train poorly on chaotic forecasting, usually blamed on barren plateaus (exponentially vanishing gradients). Using an exactly simulable four-qubit variational quantum physics-informed circuit fit to Lorenz-63, we show the barren-plateau explanation fails: the failure is an architectural capacity ceiling fixed by the circuit time-encoding, not its trainable depth. Four m...

[View on arXiv](http://arxiv.org/abs/2604.23743v2) | [PDF](https://arxiv.org/pdf/2604.23743v2)

---

### [A Dynamical Lie-Algebraic Framework for Hamiltonian Engineering and Quantum Control](http://arxiv.org/abs/2603.04916v2)
**Authors:** Yanying Liang, Ruibin Xu, Mao-Sheng Li et al.  
**Published:** 2026-03-05  
**Updated:** 2026-07-19  
**Categories:** quant-ph  

**Abstract:** Determining the unitary dynamics accessible from finite Hamiltonian resources is a central problem in Hamiltonian engineering and quantum control. Dynamical Lie algebras (DLAs) connect available control Hamiltonians with the reachable dynamics, but their use as a design tool for modifying Hamiltonian generator sets remains less developed. In this work, we develop a finite-dimensional DLA framework...

[View on arXiv](http://arxiv.org/abs/2603.04916v2) | [PDF](https://arxiv.org/pdf/2603.04916v2)

---

### [CutBackdoor: A Circuit Cut Triggered Backdoor Attack on Variational Quantum Algorithms](http://arxiv.org/abs/2607.18126v1)
**Authors:** Ahatesham Bhuiyan, Hoang Ngo, Cheng Chu et al.  
**Published:** 2026-07-20  
**Updated:** 2026-07-20  
**Categories:** quant-ph, cs.CR  

**Abstract:** Variational Quantum Algorithms (VQAs) are a leading paradigm for near-term quantum computing, combining parameterized quantum circuits with classical optimization across quantum chemistry, combinatorial optimization, and quantum machine learning. Since real-world VQA deployments routinely require circuits that exceed available hardware capacity, quantum circuit cutting has become an indispensable ...

[View on arXiv](http://arxiv.org/abs/2607.18126v1) | [PDF](https://arxiv.org/pdf/2607.18126v1)

---

### [Quantum Topological Data Encoding](http://arxiv.org/abs/2607.13847v1)
**Authors:** Adam Wesołowski, Dimitrios Thanos, Daniel Leykam et al.  
**Published:** 2026-07-15  
**Updated:** 2026-07-15  
**Categories:** quant-ph, cs.LG  

**Abstract:** Many datasets encountered across a wide range of domains possess rich geometric and topological structure that is difficult to capture using conventional vector-based representations. Quantum machine learning offers the possibility of processing high-dimensional data in Hilbert spaces, but its practical success depends critically on how classical data is encoded into quantum states. We introduce \...

[View on arXiv](http://arxiv.org/abs/2607.13847v1) | [PDF](https://arxiv.org/pdf/2607.13847v1)

---

### [Spin-Adapted Fermionic Unitaries: From Lie Algebras to Compact Quantum Circuits](http://arxiv.org/abs/2511.13485v2)
**Authors:** Ilias Magoulas, Francesco A. Evangelista  
**Published:** 2025-11-17  
**Updated:** 2026-07-20  
**Categories:** quant-ph, physics.chem-ph  

**Abstract:** Conservation of symmetries is crucial for reliable quantum simulations of molecular systems, yet compact circuit implementations of fully symmetry-adapted fermionic unitaries have remained elusive beyond the simplest excitation classes. Here we address this issue for the set of singlet spin-adapted generalized singles and doubles operators (saGSD). Using the Wei--Norman approach, we derive exact p...

[View on arXiv](http://arxiv.org/abs/2511.13485v2) | [PDF](https://arxiv.org/pdf/2511.13485v2)

---

### [Rethinking Quantum Continual Learning with Quantum Fisher Information](http://arxiv.org/abs/2607.16030v1)
**Authors:** Yu-Chao Hsu, Yu-Cheng Lin, Tai-Yue Li et al.  
**Published:** 2026-07-17  
**Updated:** 2026-07-17  
**Categories:** quant-ph, cs.AI, cs.LG  

**Abstract:** Quantum continual learning aims to train quantum models on sequential tasks without losing previously learned knowledge. However, variational quantum classifiers (VQCs) are prone to catastrophic forgetting under nonstationary task distributions. We propose quantum elastic weight consolidation (QEWC), a quantum Fisher information (QFI)-informed regularization method for mitigating forgetting. Unlik...

[View on arXiv](http://arxiv.org/abs/2607.16030v1) | [PDF](https://arxiv.org/pdf/2607.16030v1)

---

### [The Complexity of Dynamical Correlators: Operator Shadows and Exponential Learning Separations](http://arxiv.org/abs/2607.15493v1)
**Authors:** Shao-Hen Chiew, Armando Angrisani, Zoe Holmes  
**Published:** 2026-07-16  
**Updated:** 2026-07-16  
**Categories:** quant-ph  

**Abstract:** Quantum platforms can realize many-body dynamics beyond classical simulation yet complete readout remains intractable: the cost of extracting accessible information scales exponentially with system size. Classical shadows and Bell sampling offer scalable, multi-observable estimation from randomized or entanglement-assisted measurements. Here we aim to push these ideas beyond static snapshots to dy...

[View on arXiv](http://arxiv.org/abs/2607.15493v1) | [PDF](https://arxiv.org/pdf/2607.15493v1)

---

### [A Lie-algebraic approach to non-Markovian quantum dynamics](http://arxiv.org/abs/2607.13865v1)
**Authors:** Haijin Ding, Stephen S. -T. Yau, Zhiwen Zhang  
**Published:** 2026-07-15  
**Updated:** 2026-07-15  
**Categories:** quant-ph  

**Abstract:** In this paper, we study the non-Markovian quantum dynamics in quantum computations from the perspective of a Lie algebraic approach based on numerical analysis. By vectorizing the density matrix of quantum states, the non-Markovian evolutions can be represented with high-dimensional linear time-varying equations, where the time-varying parameters arise from the non-Markovian interactions between t...

[View on arXiv](http://arxiv.org/abs/2607.13865v1) | [PDF](https://arxiv.org/pdf/2607.13865v1)

---

### [Quantum circuit design via dynamic Pauli constraints](http://arxiv.org/abs/2605.22744v3)
**Authors:** James R. Wootton, Merlin Incerti-Medici, Daniel Bultrini et al.  
**Published:** 2026-05-21  
**Updated:** 2026-07-17  
**Categories:** quant-ph  

**Abstract:** We introduce the Motte model, a software-oriented model of quantum computation motivated by the practical constraints of near-term quantum hardware. In this model, gates are specified by constraints expressed in terms of Pauli observables, with each disjoint layer of gates accompanied by a pairwise or k-local quantum state tomography of the device. We prove that the model is equivalent to the coup...

[View on arXiv](http://arxiv.org/abs/2605.22744v3) | [PDF](https://arxiv.org/pdf/2605.22744v3)

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
