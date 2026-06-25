# arXiv Daily Digest - 2026-06-25

**Search Period:** Last 7 days  
**Papers Found:** 11

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


### [Barren Plateaus Beyond Observable Concentration](http://arxiv.org/abs/2603.18479v2)
**Authors:** Zi-Shen Li, Bujiao Wu, Xiao-Wei Li et al.  
**Published:** 2026-03-19  
**Updated:** 2026-06-19  
**Categories:** quant-ph  

**Abstract:** Parameterized quantum circuits (PQCs) are central to quantum machine learning and near-term quantum simulation, but their scalability is often hindered by barren plateaus (BPs), where gradients decay exponentially with system size. Prior explanations, including expressivity, entanglement, locality, and noise, are often presented in ways that conflate two distinct issues: concentration of the measu...

[View on arXiv](http://arxiv.org/abs/2603.18479v2) | [PDF](https://arxiv.org/pdf/2603.18479v2)

---

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

### [Mitigating Measurement-Induced Training Instability in Hybrid Quantum Neural Networks for Protein Classification](http://arxiv.org/abs/2606.22551v1)
**Authors:** Milton Mondal, Sushovan Chanda, Mohamad Mahdi Alawieh et al.  
**Published:** 2026-06-21  
**Updated:** 2026-06-21  
**Categories:** cs.LG, cs.CV  

**Abstract:** Hybrid Quantum Neural Network (QNN) classifiers produce logits as expectation values of quantum measurement operators. For standard Pauli measurements, these outputs are intrinsically bounded to the interval [-1,1]. When such bounded logits are used directly with the cross-entropy loss applied to softmax-normalized logits for multi-class classification, the loss function operates in a regime of we...

[View on arXiv](http://arxiv.org/abs/2606.22551v1) | [PDF](https://arxiv.org/pdf/2606.22551v1)

---

### [A Correlation Aware Quantum Feature Map for Variational Quantum Classification](http://arxiv.org/abs/2606.21570v1)
**Authors:** Murat Kurt  
**Published:** 2026-06-19  
**Updated:** 2026-06-19  
**Categories:** quant-ph  

**Abstract:** Quantum machine learning has emerged as a promising research area for learning complex data patterns. However, most existing quantum feature maps employ fixed encoding strategies that do not explicitly consider the relationships among features within a dataset. In this study, we propose a Correlation Aware Quantum Feature Map (CAQFM) which integrates feature dependencies into the quantum encoding ...

[View on arXiv](http://arxiv.org/abs/2606.21570v1) | [PDF](https://arxiv.org/pdf/2606.21570v1)

---

### [Exploiting More Than Symmetry in Variational Quantum Machine Learning](http://arxiv.org/abs/2606.20316v1)
**Authors:** Markus Baumann, Claudia Linnhoff-Popien  
**Published:** 2026-06-18  
**Updated:** 2026-06-18  
**Categories:** quant-ph  

**Abstract:** The success of variational quantum learning models crucially depends on choosing parametrizations that reflect the structure of the problem at hand. Symmetries provide one of the clearest such structures: whenever transformations of the input leave the desired outcome unchanged, this invariance should be built into the model rather than discovered during training. However, imposing a symmetry does...

[View on arXiv](http://arxiv.org/abs/2606.20316v1) | [PDF](https://arxiv.org/pdf/2606.20316v1)

---

### [On a Central Limit Theorem and Sanov's principle for quantum neural networks](http://arxiv.org/abs/2606.21721v1)
**Authors:** Anderson Melchor Hernandez  
**Published:** 2026-06-19  
**Updated:** 2026-06-19  
**Categories:** quant-ph, math-ph, math.PR  

**Abstract:** In this work, we study the fluctuations of a Mixture of Experts (MoE) generated by a quantum neural network trained via gradient flow on supervised learning problems. Our main results establish the Central Limit Theorem (CLT), and Sanov's principle for an MoE as the number of experts diverges. We demonstrate that the fluctuations of the empirical measure of its parameters close to its correspondin...

[View on arXiv](http://arxiv.org/abs/2606.21721v1) | [PDF](https://arxiv.org/pdf/2606.21721v1)

---

### [Evaluation of Variational Quantum Classifiers (VQC) for Cyberattack Detection in the NISQ Era](http://arxiv.org/abs/2606.21715v1)
**Authors:** Angelos Thomos, Theodore Andronikos  
**Published:** 2026-06-19  
**Updated:** 2026-06-19  
**Categories:** quant-ph  

**Abstract:** This paper investigates the effectiveness and structural limits of Variational Quantum Classifiers (VQC) for detecting network anomalies in the era of Noisy Intermediate-Scale Quantum (NISQ) systems. Using the official 20\% research subset of the NSL-KDD dataset, a 4-qubit classifier featuring a 24-parameter trainable ansatz was developed, utilizing amplitude encoding to embed 16 principal compone...

[View on arXiv](http://arxiv.org/abs/2606.21715v1) | [PDF](https://arxiv.org/pdf/2606.21715v1)

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
