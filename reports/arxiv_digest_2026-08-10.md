# arXiv Daily Digest - 2026-08-10

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

### [QuScope: An Open-Source Python Framework for Quantum-Circuit Simulation of Transmission Electron Microscopy](http://arxiv.org/abs/2608.02782v1)
**Authors:** Sean D. Lam, Roberto dos Reis  
**Published:** 2026-08-03  
**Updated:** 2026-08-03  
**Categories:** quant-ph, cond-mat.mtrl-sci  

**Abstract:** Image formation in transmission electron microscopy (TEM) is governed by the coherent evolution of the electron wavefunction through the specimen and the objective lens. This physics maps naturally onto the gate model of quantum computation. We present QuScope, an open-source Python framework that expresses the complete TEM image-formation pipeline as quantum circuits. The $N\times N$ electron wav...

[View on arXiv](http://arxiv.org/abs/2608.02782v1) | [PDF](https://arxiv.org/pdf/2608.02782v1)

---

### [Factorization of Exclusive-Sum-Of-Products Expressions with Rectangle Covering to Reduce Quantum Circuit Cost](http://arxiv.org/abs/2608.03188v1)
**Authors:** Audrey Hou, Lucia Zhang, Ali Al-Bayaty et al.  
**Published:** 2026-08-04  
**Updated:** 2026-08-04  
**Categories:** quant-ph  

**Abstract:** The implementation of quantum circuits is currently very expensive, especially due to the usage of large Toffoli gates. Therefore, it is critical to optimize circuit costs by factoring expressions as they become more complex. In the proposed algorithms to factor ESOP expressions, each product term is converted into a cell in a 2D matrix, and optimal factored AND/EXOR solutions are determined using...

[View on arXiv](http://arxiv.org/abs/2608.03188v1) | [PDF](https://arxiv.org/pdf/2608.03188v1)

---

### [Magneto-oscillations, nonlinearity, and nonreciprocity of Coulomb drag in quantum circuits](http://arxiv.org/abs/2608.02812v1)
**Authors:** Alex Levchenko, Mingyang Zheng, Dominique Laroche  
**Published:** 2026-08-03  
**Updated:** 2026-08-03  
**Categories:** cond-mat.mes-hall, quant-ph  

**Abstract:** We consider the problem of Coulomb drag in interactively coupled quantum circuits built of adiabatic constrictions: quantum point contacts and short quantum-wire channels. The interplay of spatial confinement and magnetic field leads to a rich oscillatory response of the drag current as a function of gate voltage and magnetic field: drag peaks track the depopulation of magnetoelectric subbands, ar...

[View on arXiv](http://arxiv.org/abs/2608.02812v1) | [PDF](https://arxiv.org/pdf/2608.02812v1)

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
