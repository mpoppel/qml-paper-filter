# arXiv Daily Digest - 2026-10-01

**Search Period:** Last 7 days  
**Papers Found:** 19

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


### [Krylov-Lie Algebras for Variational Quantum Algorithms: Geometric, Depth-Aware Insights into Expressivity and Trainability](http://arxiv.org/abs/2607.02626v4)
**Authors:** Anžej Margeta-Cacace  
**Published:** 2026-07-02  
**Updated:** 2026-09-24  
**Categories:** quant-ph, math-ph  

**Abstract:** Variational quantum algorithms (VQAs) are a leading approach to near-term quantum computation, but their utility is limited by barren plateaus and other pathologies in their loss landscapes. Existing landscape theories based on dynamical Lie algebras, Jordan-algebraic Wishart systems, approximate t-designs, and Haar-random circuits are foundational, but they often neglect the finite-depth geometry...

[View on arXiv](http://arxiv.org/abs/2607.02626v4) | [PDF](https://arxiv.org/pdf/2607.02626v4)

---

### [Fourier-Geometric Circuit Design for Gate and Entanglement Placement in Quantum Neural Networks](http://arxiv.org/abs/2609.35489v1)
**Authors:** Seungcheol Oh, Chaemoon Im, Daeyeun Kim et al.  
**Published:** 2026-09-28  
**Updated:** 2026-09-28  
**Categories:** quant-ph  

**Abstract:** The output of a parameterized quantum circuit (PQC) can be expressed as a finite Fourier series whose accessible frequencies are fixed by the data-encoding gates. While the encoder determines which frequencies can appear, the corresponding Fourier coefficients depend on how the trainable and entangling gates are arranged. Although existing studies provide metrics for characterizing how gate struct...

[View on arXiv](http://arxiv.org/abs/2609.35489v1) | [PDF](https://arxiv.org/pdf/2609.35489v1)

---

### [Block-Wise Variational Quantum Algorithms for PDEs with Interface Penalty Constraints](http://arxiv.org/abs/2609.36710v1)
**Authors:** Hangran Jie, Yuntao Cui, Sunho Kim  
**Published:** 2026-09-29  
**Updated:** 2026-09-29  
**Categories:** quant-ph  

**Abstract:** Global variational quantum algorithm (VQA) frameworks for solving partial differential equations (PDEs) often rely on a single expressive ansatz over a uniform grid, which becomes structurally inefficient when solutions exhibit spatially heterogeneous complexity such as localized singularities or thin boundary layers. A localized nonsmooth feature can degrade the convergence of the entire global q...

[View on arXiv](http://arxiv.org/abs/2609.36710v1) | [PDF](https://arxiv.org/pdf/2609.36710v1)

---

### [Modeling quantum neural network gradient with reinforcement learning](http://arxiv.org/abs/2609.31066v1)
**Authors:** Nhan Trong Luu, Duong Trung Luu, Nam Ngoc Pham et al.  
**Published:** 2026-09-25  
**Updated:** 2026-09-25  
**Categories:** quant-ph, cs.ET, cs.NE  

**Abstract:** Training quantum neural networks (QNNs) on near-term hardware remains hampered by two compounding difficulties: the exponential vanishing of gradient variance known as the barren plateau, and the $\mathcal{O}(L \cdot 2^n)$ time and memory cost of differentiating through an $n$-qubit, $L$-layer circuit. We propose RLQ-Grad, a reinforcement-learning-based optimizer in which a classical policy $π_φ$ ...

[View on arXiv](http://arxiv.org/abs/2609.31066v1) | [PDF](https://arxiv.org/pdf/2609.31066v1)

---

### [Scalable Quantum Machine Learning via Multi-layer Fully-Connected Variational Quantum Circuits](http://arxiv.org/abs/2602.16623v3)
**Authors:** Howard Su, Chen-Yu Liu, Samuel Yen-Chi Chen et al.  
**Published:** 2026-02-18  
**Updated:** 2026-09-28  
**Categories:** quant-ph  

**Abstract:** Variational quantum circuits (VQCs) face an expressivity-trainability dilemma and scalability challenges. We propose Multi-Layer Fully-Connected Variational Quantum Circuits (FC-VQC), a general-purpose quantum machine learning framework that connects local VQC blocks through measurement, deterministic parameter-free routing, and re-encoding. All trainable model parameters reside within the quantum...

[View on arXiv](http://arxiv.org/abs/2602.16623v3) | [PDF](https://arxiv.org/pdf/2602.16623v3)

---

### [Disentangling Expressibility, Symmetry Protection, and Hardware Noise in Variational Quantum Simulation of the Two-Flavor Schwinger Model](http://arxiv.org/abs/2609.30496v1)
**Authors:** Karthikeya Machiraju, Krishna Sujith, Kaustav Bhowmick  
**Published:** 2026-09-24  
**Updated:** 2026-09-24  
**Categories:** quant-ph  

**Abstract:** Existing quantum simulations of the two-flavor Schwinger model have run at a single lattice size, and it is not known how far the variational approach can be pushed or which weakness stops it first. Following the model from N = 2 to 6 staggered lattice sites, we find that the binding constraint at reachable sizes is hardware noise rather than circuit expressibility or trainability, and identify N ...

[View on arXiv](http://arxiv.org/abs/2609.30496v1) | [PDF](https://arxiv.org/pdf/2609.30496v1)

---

### [Encryptability As a Coordinate Choice: Depth-One Homomorphic Federated Learning of Quantum Neural Networks](http://arxiv.org/abs/2609.30581v1)
**Authors:** Marcel Mordarski, Nathan Mani, Arshad Patel et al.  
**Published:** 2026-09-24  
**Updated:** 2026-09-24  
**Categories:** quant-ph, cs.CR, cs.DC, cs.LG  

**Abstract:** Encrypted training relies on keeping server-side updates low-degree. This constraint traditionally excludes models whose weights inhabit a compact Lie group (notably variational quantum circuits, where every trainable weight is an $\mathrm{SU(2)}$ rotation). Expressed in Euler angles or discrete alphabets, these updates appear transcendental, historically demanding prohibitive costs: one client--s...

[View on arXiv](http://arxiv.org/abs/2609.30581v1) | [PDF](https://arxiv.org/pdf/2609.30581v1)

---

### [Factorization dynamics between quantum Fisher information and quantum coherence](http://arxiv.org/abs/2609.36571v1)
**Authors:** Xinzhi Zhao, Xinglei Yu, Liangsheng Li et al.  
**Published:** 2026-09-29  
**Updated:** 2026-09-29  
**Categories:** quant-ph  

**Abstract:** Quantum Fisher information (QFI) quantifies the sensitivity of a quantum state to a parameter change and plays a key role in quantum metrology. Meanwhile, quantum coherence is a crucial resource for quantum information processing. However, despite extensive studies on various aspects of quantum metrology, the interplay between the dynamics of the QFI and quantum coherence remains unexplored. Here,...

[View on arXiv](http://arxiv.org/abs/2609.36571v1) | [PDF](https://arxiv.org/pdf/2609.36571v1)

---

### [Practical fermionic shadows enabled by improved sample-complexity bounds](http://arxiv.org/abs/2609.40173v1)
**Authors:** Maxwell West, Su Yeon Chang, Luke Coffman et al.  
**Published:** 2026-09-30  
**Updated:** 2026-09-30  
**Categories:** quant-ph  

**Abstract:** Classical shadow tomography is widely touted as supplying a family of methods for extracting information from quantum systems with polynomially scaling sample-complexities. In the current era of quantum computers possessing on the order of hundreds of qubits, however, polynomial scaling can nonetheless be prohibitive. Thus, there is a strong practical need for obtaining sample-complexity bounds wh...

[View on arXiv](http://arxiv.org/abs/2609.40173v1) | [PDF](https://arxiv.org/pdf/2609.40173v1)

---

### [Deciding the Attainability of the Multiparameter Quantum Fisher Information is NP-Hard](http://arxiv.org/abs/2609.35975v1)
**Authors:** Lorcan O. Conlon, Laura Shou, V Vijendran et al.  
**Published:** 2026-09-28  
**Updated:** 2026-09-28  
**Categories:** quant-ph  

**Abstract:** The quantum Fisher information (QFI) sets a fundamental bound on the attainable precision when estimating multiple parameters simultaneously. Incompatibility among the individually optimal measurements can, in some cases, imply that the precision limit set by the QFI is not attainable. In certain special cases, including pure states and full-rank states, the exact conditions for when the precision...

[View on arXiv](http://arxiv.org/abs/2609.35975v1) | [PDF](https://arxiv.org/pdf/2609.35975v1)

---

### [Quantum Fisher Information as the Speed Limit for Multipartite Entanglement](http://arxiv.org/abs/2609.31853v1)
**Authors:** Zain H. Saleem, Da-Wei Luo, Anjala M. Babu et al.  
**Published:** 2026-09-25  
**Updated:** 2026-09-25  
**Categories:** quant-ph  

**Abstract:** We derive a universal upper bound on the speed of multipartite entanglement generation. For arbitrary differentiable pure multipartite states undergoing parameter-dependent evolution, we prove that the rate of change of generalized concurrence is bounded by the square root of the quantum Fisher information (QFI). The proof follows directly from the symmetric logarithmic derivative formalism, revea...

[View on arXiv](http://arxiv.org/abs/2609.31853v1) | [PDF](https://arxiv.org/pdf/2609.31853v1)

---

### [Exact Characterization of the Holevo Bound by a Quantum Fisher Information Family](http://arxiv.org/abs/2609.31601v1)
**Authors:** Koji Yamaguchi, Hiroyasu Tajima  
**Published:** 2026-09-25  
**Updated:** 2026-09-25  
**Categories:** quant-ph, cond-mat.stat-mech, math-ph  

**Abstract:** The quantum Cramér-Rao bound constrains the precision of parameter estimation through the symmetric logarithmic derivative quantum Fisher information (SLD QFI). It is asymptotically achievable for regular single-parameter estimation models, but generally not in the multiparameter setting, where optimal measurements for different parameters may be incompatible. For multiparameter estimation, allowi...

[View on arXiv](http://arxiv.org/abs/2609.31601v1) | [PDF](https://arxiv.org/pdf/2609.31601v1)

---

### [Disorder-induced quantum Fisher information in topological quantum systems](http://arxiv.org/abs/2609.30142v1)
**Authors:** Advay Burte, Keshav Das Agarwal, Leela Ganesh Chandra Lakkaraju et al.  
**Published:** 2026-09-24  
**Updated:** 2026-09-24  
**Categories:** quant-ph, cond-mat.dis-nn, cond-mat.str-el  

**Abstract:** The topological order of the Kitaev toric code is insensitive to disorder in its couplings since all star and plaquette operators commute; consequently, the stabilizer ground-state manifold is independent of the individual coupling strengths, and the phase remains stable against weak local perturbations. This insensitivity, however, is a property of the eigenstates and not of the dynamics they gen...

[View on arXiv](http://arxiv.org/abs/2609.30142v1) | [PDF](https://arxiv.org/pdf/2609.30142v1)

---

### [SQUARE: Structured Quantum Representation Adapters as Compact Quadratic Feature Maps for Frozen Language Models](http://arxiv.org/abs/2609.37134v1)
**Authors:** Emily Jimin Roh, Hyojun Ahn, Hoyeong Lee et al.  
**Published:** 2026-09-29  
**Updated:** 2026-09-29  
**Categories:** quant-ph, cs.AI  

**Abstract:** Frozen language models (LMs) are increasingly used as fixed feature extractors for downstream reranking, scoring, and preference modeling, raising a practical question: how should a compact module represent interactions among features in a fixed low-dimensional bottleneck? Common linear and low-rank adapters remain linear at the adaptation module itself, whereas explicit second-order alternatives ...

[View on arXiv](http://arxiv.org/abs/2609.37134v1) | [PDF](https://arxiv.org/pdf/2609.37134v1)

---

### [Quantum Entanglement in Variational Quantum Classification for Breast Cancer Diagnosis](http://arxiv.org/abs/2609.34617v1)
**Authors:** Zineb Hazmoun, Zoubida Sakhi, Mohamed Bennai  
**Published:** 2026-09-28  
**Updated:** 2026-09-28  
**Categories:** quant-ph  

**Abstract:** This study looks at how the entangling structure of a variational quantum classifier (VQC) relates to its performance in breast cancer diagnosis, using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. We tested three three-qubit configurations that share the same EfficientSU2 ansatz, COBYLA optimizer, and stratified five-fold cross-validation, but differ in their entangling structure: a sing...

[View on arXiv](http://arxiv.org/abs/2609.34617v1) | [PDF](https://arxiv.org/pdf/2609.34617v1)

---

### [The ZZ feature map induces a signless Laplacian metric: a closed-form classical surrogate for quantum kernel regression](http://arxiv.org/abs/2608.29422v2)
**Authors:** Erkut Tekeli  
**Published:** 2026-08-29  
**Updated:** 2026-09-26  
**Categories:** quant-ph  

**Abstract:** Bandwidth-tuned quantum kernels have been shown to lose their advantage over classical kernels and to resemble radial basis function kernels, but the analytical support for that observation rests on separable encoding circuits and captures entangling circuits only qualitatively. We close this gap for the ZZ feature map. We prove that in the small-bandwidth regime the induced kernel has, to leading...

[View on arXiv](http://arxiv.org/abs/2608.29422v2) | [PDF](https://arxiv.org/pdf/2608.29422v2)

---

### [From Projected Subspaces to Full-Space Implementations: Representation-Equivalence Audits for Open-Shell ADAPT-VQE](http://arxiv.org/abs/2609.36060v1)
**Authors:** Lucia Malíčková, Petr Klenovský  
**Published:** 2026-09-28  
**Updated:** 2026-09-28  
**Categories:** quant-ph  

**Abstract:** A variational state optimized in a symmetry-projected space need not be the state prepared by exponentiating the corresponding unprojected parent generators. For an orthogonal target-space projector $P$, $Q=I-P$, and an anti-Hermitian generator $A$, exact equivalence for all amplitudes requires target-space invariance, $QAP=0$. We formulate this condition as a representation-equivalence audit for ...

[View on arXiv](http://arxiv.org/abs/2609.36060v1) | [PDF](https://arxiv.org/pdf/2609.36060v1)

---

### [Dissipation accelerates quantum and classical simulation of open-system dynamics](http://arxiv.org/abs/2609.40174v1)
**Authors:** Armando Angrisani, Ricard Puig, Yanting Teng et al.  
**Published:** 2026-09-30  
**Updated:** 2026-09-30  
**Categories:** quant-ph, cond-mat.stat-mech  

**Abstract:** Simulating open quantum systems reveals how environmental coupling shapes relaxation, excitation transport, and the dynamics of quantum correlations. On quantum hardware, dissipative channels add operations and might seem to increase cost. However, we show that a broad class of Pauli noise, including depolarization, can ease quantum and classical simulation by exponentially suppressing high-weight...

[View on arXiv](http://arxiv.org/abs/2609.40174v1) | [PDF](https://arxiv.org/pdf/2609.40174v1)

---

### [Adaptive Rotation for iSOMA: Geometry, Benchmarking, and Noise Robustness in Variational Quantum Objectives](http://arxiv.org/abs/2609.37193v1)
**Authors:** Vojtěch Novák, Ivan Zelinka  
**Published:** 2026-09-29  
**Updated:** 2026-09-29  
**Categories:** cs.NE, quant-ph  

**Abstract:** We study whether the coordinate dependence of the improved Self-Organizing Migrating Algorithm (iSOMA) can be reduced while retaining its inexpensive leader-directed migration mechanism. We introduce iSOMA-AR, which learns a basis from successful migration displacements and selectively applies the standard perturbation mask in that basis. On the complete noiseless BBOB suite, iSOMA- AR significant...

[View on arXiv](http://arxiv.org/abs/2609.37193v1) | [PDF](https://arxiv.org/pdf/2609.37193v1)

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
