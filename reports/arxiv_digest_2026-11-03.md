# arXiv Daily Digest - 2026-11-03

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

### [HQNN-CL: Regime-Aware Hybrid Quantum Neural Network with Continual Learning for Financial Forecasting](http://arxiv.org/abs/2611.02091v1)
**Authors:** Muhammad Atif Saeed, Akhtar Jamil, Subhan Ullah  
**Published:** 2026-11-01  
**Updated:** 2026-11-01  
**Categories:** cs.LG, cs.AI  

**Abstract:** Financial time-series forecasting is still difficult owing to the non-stationary nature of the market, regime changes, and catastrophic forgetting, which makes conventional deep learning models hard to adapt in the long term. To achieve uncertainty-aware robust financial forecasting in the face of changing market conditions, this study proposes a Hybrid Quantum-Classical Neural Network with Contin...

[View on arXiv](http://arxiv.org/abs/2611.02091v1) | [PDF](https://arxiv.org/pdf/2611.02091v1)

---

### [Universal Bounds for Out-of-Distribution Unitary Learning](http://arxiv.org/abs/2610.03690v2)
**Authors:** Joachim Favre, Armando Angrisani, Zoë Holmes  
**Published:** 2026-10-02  
**Updated:** 2026-10-29  
**Categories:** quant-ph  

**Abstract:** How much can the action of an unknown unitary dynamics on one family of states reveal about its action elsewhere? We develop a general framework for out-of-distribution quantum dynamics learning based on the first and second moments of the input distribution. These define a notion of the bias and expressivity of an ensemble, which together control generalization between arbitrary mixed-state train...

[View on arXiv](http://arxiv.org/abs/2610.03690v2) | [PDF](https://arxiv.org/pdf/2610.03690v2)

---

### [Dynamical quantum simulations as laboratories for quantum information](http://arxiv.org/abs/2610.37589v1)
**Authors:** Su Yeon Chang, Hailey S. Murray, Brian Barch et al.  
**Published:** 2026-10-27  
**Updated:** 2026-10-27  
**Categories:** quant-ph, cond-mat.mtrl-sci  

**Abstract:** Dynamical quantum simulations are commonly designed to access a particular response function or correlation function. Yet the underlying protocol prepares a family of many-body states that contains information well beyond the observable for which the simulation was originally designed. In this work, we explore the idea of using these states as laboratories for quantum information. As a case study,...

[View on arXiv](http://arxiv.org/abs/2610.37589v1) | [PDF](https://arxiv.org/pdf/2610.37589v1)

---

### [Quantum circuit design via dynamic Pauli constraints](http://arxiv.org/abs/2605.22744v4)
**Authors:** James R. Wootton, Merlin Incerti-Medici, Daniel Bultrini et al.  
**Published:** 2026-05-21  
**Updated:** 2026-11-02  
**Categories:** quant-ph  

**Abstract:** We introduce the Motte model, a software-oriented model of quantum computation motivated by the practical constraints of near-term quantum hardware. In this model, gates are specified by constraints expressed in terms of Pauli observables, with each disjoint layer of gates accompanied by a pairwise or k-local quantum state tomography of the device. We prove that the model is equivalent to the coup...

[View on arXiv](http://arxiv.org/abs/2605.22744v4) | [PDF](https://arxiv.org/pdf/2605.22744v4)

---

### [Where Does a Variational Quantum Classifier Lose Accuracy? Angle Encoding, Readout and the Fidelity-Kernel Reference in Cyberattack Detection](http://arxiv.org/abs/2611.00885v1)
**Authors:** Angelos Thomos, Theodore Andronikos  
**Published:** 2026-10-31  
**Updated:** 2026-10-31  
**Categories:** quant-ph  

**Abstract:** When a variational quantum classifier performs poorly at detecting network attacks, it is hard to tell which part is to blame: the encoding, the trained circuit or the readout, which turns measurements into a class. In this work, we build upon a recent amplitude-encoded classifier, and we try to separate their contributions to the NSL-KDD benchmark in noiseless simulation. We compare angle and amp...

[View on arXiv](http://arxiv.org/abs/2611.00885v1) | [PDF](https://arxiv.org/pdf/2611.00885v1)

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
