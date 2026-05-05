# arXiv Daily Digest - 2026-05-05

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


### [Beyond Single Trajectories: Optimal Control and Jordan-Lie Algebra in Hybrid Quantum Walks for Combinatorial Optimization](http://arxiv.org/abs/2604.25760v1)
**Authors:** Tianen Chen, Yun Shang  
**Published:** 2026-04-28  
**Updated:** 2026-04-28  
**Categories:** quant-ph  

**Abstract:** The Quantum Approximate Optimization Algorithm (QAOA) follows a single, fixed evolution path, overlooking the potential computational advantage of coherently superposing multiple trajectories. Here we overcome this limitation with a hybrid quantum walk (HQW) ansatz that super poses multiple Hamiltonian-driven paths coherently within each circuit layer via a dynamical coin operator. QAOA emerges as...

[View on arXiv](http://arxiv.org/abs/2604.25760v1) | [PDF](https://arxiv.org/pdf/2604.25760v1)

---

### [Barren Plateaus as Destructive Interference: A Diagnostic Framework and Implications for Structured Ansatzes](http://arxiv.org/abs/2605.01319v1)
**Authors:** Pilsung Kang  
**Published:** 2026-05-02  
**Updated:** 2026-05-02  
**Categories:** quant-ph, cs.LG  

**Abstract:** Barren plateaus (BPs) are usually described by the exponential suppression of gradient variance, but the mechanism by which gradient signal disappears remains unclear. We show that this phenomenon can be understood as destructive interference among termwise gradient contributions. To make this perspective operational, we introduce a diagnostic framework based on the cancellation ratio $R_k$, the e...

[View on arXiv](http://arxiv.org/abs/2605.01319v1) | [PDF](https://arxiv.org/pdf/2605.01319v1)

---

### [Parameterized Quantum Circuits as Feature Maps: Representation Quality and Readout Effects in Multispectral Land-Cover Classification](http://arxiv.org/abs/2604.26675v1)
**Authors:** Ralntion Komini, Aikaterini Mandilara, Georgios Maragkopoulos et al.  
**Published:** 2026-04-29  
**Updated:** 2026-04-29  
**Categories:** quant-ph, cs.LG  

**Abstract:** We investigate variational quantum classifiers (VQCs) for land-cover classification from multispectral satellite imagery, adopting a feature-map perspective in which the quantum circuit defines a nonlinear data embedding while the readout determines how this representation is exploited. Using the EuroSAT-MS dataset, we perform a systematic one-vs-one evaluation across all class pairs under a contr...

[View on arXiv](http://arxiv.org/abs/2604.26675v1) | [PDF](https://arxiv.org/pdf/2604.26675v1)

---

### [Operator spreading and recoverability of local quantum Fisher information in a $U(1)$-broken spin chain](http://arxiv.org/abs/2605.02774v1)
**Authors:** Marcin Płodzień, Jan Chwedeńczuk  
**Published:** 2026-05-04  
**Updated:** 2026-05-04  
**Categories:** quant-ph  

**Abstract:** While out-of-time-order correlators establish a causal light cone for operator spreading, they do not guarantee that the parameter sensitivity carried by the operator remains locally recoverable. We examine the distinction between operator spreading and metrological recoverability for a parameter encoded in a single site of an XX spin chain subjected to a $U(1)$-breaking transverse field. We evalu...

[View on arXiv](http://arxiv.org/abs/2605.02774v1) | [PDF](https://arxiv.org/pdf/2605.02774v1)

---

### [Topological protection of local quantum Fisher information](http://arxiv.org/abs/2605.00770v1)
**Authors:** Marcin Płodzień, Jan Chwedeńczuk  
**Published:** 2026-05-01  
**Updated:** 2026-05-01  
**Categories:** quant-ph, cond-mat.quant-gas  

**Abstract:** In many-body quantum systems, unitary dynamics generically delocalize locally encoded information, causing single-site metrological sensitivity to vanish. We analytically demonstrate that a topological phase can prevent this dispersal. In the open Kitaev chain, a Majorana zero mode fixes the boundary quantum Fisher information (QFI) at a nonzero plateau that persists for times exponentially long i...

[View on arXiv](http://arxiv.org/abs/2605.00770v1) | [PDF](https://arxiv.org/pdf/2605.00770v1)

---

### [Superiority of Krylov shadow tomography in estimating quantum Fisher information: From bounds to exactness](http://arxiv.org/abs/2602.17361v2)
**Authors:** Yuan-Hao Wang, Da-Jian Zhang  
**Published:** 2026-02-19  
**Updated:** 2026-04-29  
**Categories:** quant-ph  

**Abstract:** Estimating the quantum Fisher information (QFI) is a crucial yet challenging task with widespread applications across quantum science and technologies. The recently proposed Krylov shadow tomography (KST) opens a new avenue for this task by introducing a series of Krylov bounds on the QFI. In this work, we address the practical applicability of the KST, unveiling that the Krylov bounds of low orde...

[View on arXiv](http://arxiv.org/abs/2602.17361v2) | [PDF](https://arxiv.org/pdf/2602.17361v2)

---

### [Qvine: Vine Structured Quantum Circuits for Loading High Dimensional Distributions](http://arxiv.org/abs/2604.26213v1)
**Authors:** David Quiroga, Hannes Leipold, Bibhas Adhikari  
**Published:** 2026-04-29  
**Updated:** 2026-04-29  
**Categories:** quant-ph, cs.AI  

**Abstract:** Loading high dimensional distributions is an important task for utilizing quantum computers on applications ranging from machine learning to finance. The high dimensionality leads to a curse of dimensionality, representing a d-dimensional distribution with k resolution requires dk qubits and an unstructured parameterized circuit would express a unitary in an exponential operator space in the numbe...

[View on arXiv](http://arxiv.org/abs/2604.26213v1) | [PDF](https://arxiv.org/pdf/2604.26213v1)

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
