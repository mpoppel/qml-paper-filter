# arXiv Daily Digest - 2026-11-02

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


### [From Barren Plateaus to SPSA Optimization in Variational Quantum Eigensolvers](http://arxiv.org/abs/2608.09810v2)
**Authors:** Zhen Qin  
**Published:** 2026-08-10  
**Updated:** 2026-10-28  
**Categories:** quant-ph, math.OC  

**Abstract:** The barren plateau (BP) phenomenon poses a fundamental challenge to the trainability of variational quantum eigensolvers (VQEs) by causing exponentially vanishing gradients as the system size increases. While extensive studies have investigated the geometric origins of BP, its impact on the optimization dynamics and complexity of practical algorithms under finite-shot measurements has received com...

[View on arXiv](http://arxiv.org/abs/2608.09810v2) | [PDF](https://arxiv.org/pdf/2608.09810v2)

---

### [Barren Plateaus in Quantum Linear Combinations: Interference and Lie-Algebraic Control](http://arxiv.org/abs/2610.38193v1)
**Authors:** Amir Koutahi, Pouya Kananian, Hans-Arno Jacobsen  
**Published:** 2026-10-28  
**Updated:** 2026-10-28  
**Categories:** quant-ph, cs.ET, cs.LG  

**Abstract:** A coherent sum of quantum circuits introduces interference that can alter the training signal. We distinguish a single linear combination of unitaries (LCU) from a circuit that repeatedly alternates branch operations and ancilla mixing. For a single-pass LCU, independent branches with zero matrix means cannot erase the differentiated branch's gradient variance: the total variance is at least the b...

[View on arXiv](http://arxiv.org/abs/2610.38193v1) | [PDF](https://arxiv.org/pdf/2610.38193v1)

---

### [Digital-Analog Quantum Neural Network with Long-Range Interactions](http://arxiv.org/abs/2610.39847v1)
**Authors:** Zongyuan Ge, Daili Li, Zewen Zhang  
**Published:** 2026-10-29  
**Updated:** 2026-10-29  
**Categories:** quant-ph  

**Abstract:** Protocols combining single-qubit digital operations with analog quantum evolutions bypass large numbers of two-qubit entangling gates, offering a promising approach to variational algorithms and hybrid quantum simulation. Focusing on the digital-analog quantum neural network (DA-QNN) as a representative framework, this work demonstrates that incorporating long-range interactions enhances the effic...

[View on arXiv](http://arxiv.org/abs/2610.39847v1) | [PDF](https://arxiv.org/pdf/2610.39847v1)

---

### [Overcoming the Representational Limitations of Amplitude Encoding for Quantum Neural Networks](http://arxiv.org/abs/2610.41824v1)
**Authors:** Vincent Gurgul, David Prinz  
**Published:** 2026-10-30  
**Updated:** 2026-10-30  
**Categories:** quant-ph, cs.ET, cs.LG, math.OC  

**Abstract:** Quantum Neural Networks (QNNs) offer a framework for learning functions from classical data, but their expressive power depends critically on how inputs are encoded into quantum states. Angle encoding can require resources that scale with the number of features, whereas amplitude encoding represents $d$ features using only $\mathcal{O}(\log_2 d)$ qubits. This compactness comes at a representationa...

[View on arXiv](http://arxiv.org/abs/2610.41824v1) | [PDF](https://arxiv.org/pdf/2610.41824v1)

---

### [Universal Bounds for Out-of-Distribution Unitary Learning](http://arxiv.org/abs/2610.03690v2)
**Authors:** Joachim Favre, Armando Angrisani, Zoë Holmes  
**Published:** 2026-10-02  
**Updated:** 2026-10-29  
**Categories:** quant-ph  

**Abstract:** How much can the action of an unknown unitary dynamics on one family of states reveal about its action elsewhere? We develop a general framework for out-of-distribution quantum dynamics learning based on the first and second moments of the input distribution. These define a notion of the bias and expressivity of an ensemble, which together control generalization between arbitrary mixed-state train...

[View on arXiv](http://arxiv.org/abs/2610.03690v2) | [PDF](https://arxiv.org/pdf/2610.03690v2)

---

### [Quantum Fisher Information Deficit as a Geometric Signature of Strong-to-Weak Spontaneous Symmetry Breaking](http://arxiv.org/abs/2610.37092v1)
**Authors:** Zhang-Chi Yin, Wei Zheng  
**Published:** 2026-10-27  
**Updated:** 2026-10-27  
**Categories:** quant-ph, cond-mat.quant-gas, cond-mat.stat-mech, cond-mat.str-el  

**Abstract:** Recently, strong-to-weak spontaneous symmetry breaking (SW-SSB) has been established as a framework for describing quantum phases in open systems. However SW-SSB cannot be characterized by conventional charged two-point correlators. In this work, we show that, for strongly $\mathbb{Z}_2$-symmetric states with local charged involutions, SW-SSB manifests as a persistent deficit from the maximal quan...

[View on arXiv](http://arxiv.org/abs/2610.37092v1) | [PDF](https://arxiv.org/pdf/2610.37092v1)

---

### [Dynamical quantum simulations as laboratories for quantum information](http://arxiv.org/abs/2610.37589v1)
**Authors:** Su Yeon Chang, Hailey S. Murray, Brian Barch et al.  
**Published:** 2026-10-27  
**Updated:** 2026-10-27  
**Categories:** quant-ph, cond-mat.mtrl-sci  

**Abstract:** Dynamical quantum simulations are commonly designed to access a particular response function or correlation function. Yet the underlying protocol prepares a family of many-body states that contains information well beyond the observable for which the simulation was originally designed. In this work, we explore the idea of using these states as laboratories for quantum information. As a case study,...

[View on arXiv](http://arxiv.org/abs/2610.37589v1) | [PDF](https://arxiv.org/pdf/2610.37589v1)

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
