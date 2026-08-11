# arXiv Daily Digest - 2026-08-11

**Search Period:** Last 7 days  
**Papers Found:** 6

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


### [Dynamical Lie Algebras Cannot Describe Shallow QAOA: Cragged Terrains, Barren Plateaus, and Empirical Hardness Models](http://arxiv.org/abs/2608.04252v1)
**Authors:** Harrison Copp, Charlton Li, Anžej Margeta-Cacace et al.  
**Published:** 2026-08-04  
**Updated:** 2026-08-04  
**Categories:** quant-ph, cs.ET, cs.LG  

**Abstract:** The dynamical Lie algebraic (DLA) theory of variational quantum algorithms (VQAs) predicts commonplace exponentially vanishing loss and gradient variances for sufficiently deep parametrized circuits. In this work, we show that these predictions fail dramatically in the shallow-circuit (and particularly constant-depth) regime for the Quantum Approximate Optimization Algorithm (QAOA) applied to the ...

[View on arXiv](http://arxiv.org/abs/2608.04252v1) | [PDF](https://arxiv.org/pdf/2608.04252v1)

---

### [Cautious optimism for deep parameterized quantum circuits](http://arxiv.org/abs/2607.21409v2)
**Authors:** Marie Kempkes, Elies Gil-Fuster, Carlos Bravo-Prieto et al.  
**Published:** 2026-07-23  
**Updated:** 2026-08-05  
**Categories:** quant-ph, cs.LG, stat.ML  

**Abstract:** A central challenge in quantum machine learning is understanding the scaling behavior of parameterized quantum circuits (PQCs). In particular, it remains unclear how their performance on unseen data changes as the number of trainable parameters increases. Prior works have derived formal generalization guarantees for quantum models, but it is well-known that many such results do not fully character...

[View on arXiv](http://arxiv.org/abs/2607.21409v2) | [PDF](https://arxiv.org/pdf/2607.21409v2)

---

### [From Barren Plateaus to SPSA Optimization in Variational Quantum Eigensolvers](http://arxiv.org/abs/2608.09810v1)
**Authors:** Zhen Qin  
**Published:** 2026-08-10  
**Updated:** 2026-08-10  
**Categories:** quant-ph, math.OC  

**Abstract:** The barren plateau (BP) phenomenon poses a fundamental challenge to the trainability of variational quantum eigensolvers (VQEs) by causing exponentially vanishing gradients as the system size increases. While extensive studies have investigated the geometric origins of BP, its impact on the optimization dynamics and complexity of practical algorithms under finite-shot measurements remains poorly u...

[View on arXiv](http://arxiv.org/abs/2608.09810v1) | [PDF](https://arxiv.org/pdf/2608.09810v1)

---

### [Mitigating Barren Plateaus in Quantum Denoising Diffusion Probabilistic Model](http://arxiv.org/abs/2512.06695v3)
**Authors:** Haipeng Cao, Kaining Zhang, Dacheng Tao et al.  
**Published:** 2025-12-07  
**Updated:** 2026-08-10  
**Categories:** cs.LG, quant-ph  

**Abstract:** Quantum generative models exploit quantum superposition and entanglement to enhance learning efficiency for both classical and quantum data. Recently, inspired by classical diffusion frameworks, the quantum denoising diffusion probabilistic model has emerged as a powerful tool for learning correlated noise models, many-body phases, and topological data structures. However, we demonstrate that this...

[View on arXiv](http://arxiv.org/abs/2512.06695v3) | [PDF](https://arxiv.org/pdf/2512.06695v3)

---

### [QuScope: An Open-Source Python Framework for Quantum-Circuit Simulation of Transmission Electron Microscopy](http://arxiv.org/abs/2608.02782v2)
**Authors:** Sean D. Lam, Roberto dos Reis  
**Published:** 2026-08-03  
**Updated:** 2026-08-10  
**Categories:** quant-ph, cond-mat.mtrl-sci  

**Abstract:** Image formation in transmission electron microscopy (TEM) is governed by the coherent evolution of the electron wavefunction through the specimen and the objective lens. This physics maps naturally onto the gate model of quantum computation. We present QuScope, an open-source Python framework that expresses the complete TEM image-formation pipeline as quantum circuits. The $N\times N$ electron wav...

[View on arXiv](http://arxiv.org/abs/2608.02782v2) | [PDF](https://arxiv.org/pdf/2608.02782v2)

---

### [Classical $\mathrm{SU}(2)$ Models Match or Exceed Shallow Variational Quantum Circuits on Vision Benchmarks](http://arxiv.org/abs/2608.07822v1)
**Authors:** Christopher Fulton, Irene Tsapara, Lawrence Fulton  
**Published:** 2026-08-07  
**Updated:** 2026-08-07  
**Categories:** cs.PF, cs.LG  

**Abstract:** Quaternion-valued neural networks and variational quantum circuits (VQCs) both derive local transformations from $\mathrm{SU}(2)$ geometry, yet their performance on classical supervised learning remains poorly understood. We compare real-valued, quaternion-valued, and quantum classification heads on identical frozen features across MNIST, FashionMNIST, and CIFAR-10. CIFAR-10 uses a learned 16-dime...

[View on arXiv](http://arxiv.org/abs/2608.07822v1) | [PDF](https://arxiv.org/pdf/2608.07822v1)

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
