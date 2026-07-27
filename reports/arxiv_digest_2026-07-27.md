# arXiv Daily Digest - 2026-07-27

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


### [Pulsed learning for quantum data re-uploading models](http://arxiv.org/abs/2512.10670v2)
**Authors:** Ignacio B. Acedo, Pablo Rodriguez-Grasa, Pablo Garcia-Azorin et al.  
**Published:** 2025-12-11  
**Updated:** 2026-07-22  
**Categories:** quant-ph  

**Abstract:** While Quantum Machine Learning (QML) holds great potential, its practical realization on Noisy Intermediate-Scale Quantum (NISQ) hardware has been hindered by the limitations of variational quantum circuits (VQCs). Recent evidence suggests that VQCs suffer from severe trainability and noise-related issues, leading to growing skepticism about their long-term viability. However, the possibility of i...

[View on arXiv](http://arxiv.org/abs/2512.10670v2) | [PDF](https://arxiv.org/pdf/2512.10670v2)

---

### [Cautious optimism for deep parameterized quantum circuits](http://arxiv.org/abs/2607.21409v1)
**Authors:** Marie Kempkes, Elies Gil-Fuster, Carlos Bravo-Prieto et al.  
**Published:** 2026-07-23  
**Updated:** 2026-07-23  
**Categories:** quant-ph, cs.LG, stat.ML  

**Abstract:** A central challenge in quantum machine learning is understanding the scaling behavior of parameterized quantum circuits (PQCs). In particular, it remains unclear how their performance on unseen data changes as the number of trainable parameters increases. Prior works have derived formal generalization guarantees for quantum models, but it is well-known that many such results do not fully character...

[View on arXiv](http://arxiv.org/abs/2607.21409v1) | [PDF](https://arxiv.org/pdf/2607.21409v1)

---

### [Parameterized Quantum Circuits as Feature Maps: Representation Quality and Readout Effects in Multispectral Land-Cover Classification](http://arxiv.org/abs/2604.26675v2)
**Authors:** Ralntion Komini, Aikaterini Mandilara, Georgios Maragkopoulos et al.  
**Published:** 2026-04-29  
**Updated:** 2026-07-24  
**Categories:** quant-ph, cs.LG  

**Abstract:** We investigate variational quantum classifiers (VQCs) for land-cover classification from multispectral satellite imagery, adopting a feature-map perspective in which the quantum circuit defines a nonlinear data embedding while the readout determines how this representation is exploited. Using the EuroSAT-MS dataset, we perform a systematic one-vs-one evaluation across all class pairs under a contr...

[View on arXiv](http://arxiv.org/abs/2604.26675v2) | [PDF](https://arxiv.org/pdf/2604.26675v2)

---

### [CutBackdoor: A Circuit Cut Triggered Backdoor Attack on Variational Quantum Algorithms](http://arxiv.org/abs/2607.18126v1)
**Authors:** Ahatesham Bhuiyan, Hoang Ngo, Cheng Chu et al.  
**Published:** 2026-07-20  
**Updated:** 2026-07-20  
**Categories:** quant-ph, cs.CR  

**Abstract:** Variational Quantum Algorithms (VQAs) are a leading paradigm for near-term quantum computing, combining parameterized quantum circuits with classical optimization across quantum chemistry, combinatorial optimization, and quantum machine learning. Since real-world VQA deployments routinely require circuits that exceed available hardware capacity, quantum circuit cutting has become an indispensable ...

[View on arXiv](http://arxiv.org/abs/2607.18126v1) | [PDF](https://arxiv.org/pdf/2607.18126v1)

---

### [Barren-plateau free variational quantum simulation of Z2 lattice gauge theories](http://arxiv.org/abs/2507.19203v4)
**Authors:** Fariha Azad, Matteo Inajetovic, Stefan Kühn et al.  
**Published:** 2025-07-25  
**Updated:** 2026-07-24  
**Categories:** quant-ph  

**Abstract:** In this work, we design a variational quantum eigensolver (VQE) suitable for investigating ground states and static string breaking in a $\mathbb{Z}_2$ lattice gauge theory (LGT). We consider a two-leg ladder lattice coupled to Kogut-Susskind staggered fermions and verify the results of the VQE simulations using tensor network methods. We find that for varying Hamiltonian parameter regimes and in ...

[View on arXiv](http://arxiv.org/abs/2507.19203v4) | [PDF](https://arxiv.org/pdf/2507.19203v4)

---

### [Enhancing Blood Cells Classification using Hybrid Quantum Neural Networks](http://arxiv.org/abs/2605.23324v3)
**Authors:** Guilherme Cruz, Nouhaila Innan, Alberto Marchisio et al.  
**Published:** 2026-05-22  
**Updated:** 2026-07-24  
**Categories:** cs.CV, quant-ph  

**Abstract:** Accurate classification of microscopic blood cells is still a critical task in medical image analysis, where subtle variations and limited data can challenge conventional deep learning models. As such, we investigate in this work the potential of Hybrid Quantum-Classical Neural Networks (HQNNs) to enhance feature representation and improve classification performance in this domain. We propose a mo...

[View on arXiv](http://arxiv.org/abs/2605.23324v3) | [PDF](https://arxiv.org/pdf/2605.23324v3)

---

### [PN-QNN: Harnessing Physical Noise as a Native Regularizer in Photonic Hybrid Quantum Neural Networks](http://arxiv.org/abs/2607.20045v1)
**Authors:** Farah Elnakhal, Alberto Marchisio, Nouhaila Innan et al.  
**Published:** 2026-07-22  
**Updated:** 2026-07-22  
**Categories:** quant-ph, cs.LG  

**Abstract:** Physical noise in near-term quantum hardware is usually treated as a nuisance to suppress. We ask whether it can instead act as a hardware-native regularizer for photonic hybrid quantum-classical neural networks (PHQCNNs), analogous to noise-injection regularization in classical deep learning. Using Quandela's Perceval simulator and the MerLin framework, we build PHQCNNs for Iris, Digits, and MNIS...

[View on arXiv](http://arxiv.org/abs/2607.20045v1) | [PDF](https://arxiv.org/pdf/2607.20045v1)

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
