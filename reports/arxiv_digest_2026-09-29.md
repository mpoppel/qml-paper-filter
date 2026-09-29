# arXiv Daily Digest - 2026-09-29

**Search Period:** Last 7 days  
**Papers Found:** 17

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

### [Equivalence of maximal and generic reachability for non-universal Variational Quantum Circuits](http://arxiv.org/abs/2609.27053v1)
**Authors:** Vishal S. Ngairangbam, Michael Spannowsky  
**Published:** 2026-09-22  
**Updated:** 2026-09-22  
**Categories:** quant-ph  

**Abstract:** Employing problem-specific non-universal Variational Quantum Circuits aligned with a suitable state preparation has become a standard approach to counter the difficulty in training universal ansätze. However, due to their non-universality, diagnosing their reachability and trainability remains reference-state-specific. In this work, we establish the equivalence of maximal reachability and generic ...

[View on arXiv](http://arxiv.org/abs/2609.27053v1) | [PDF](https://arxiv.org/pdf/2609.27053v1)

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

### [Nonorthogonal variational quantum simulation for quantum chemistry](http://arxiv.org/abs/2609.29337v1)
**Authors:** Zongkang Zhang, Jiajun Ren, Xiao Yuan  
**Published:** 2026-09-24  
**Updated:** 2026-09-24  
**Categories:** quant-ph  

**Abstract:** Dynamical simulation of quantum many-body systems is a central task in quantum chemistry and requires efficient wavefunction representations. Although tensor-network and neural-network quantum states have achieved considerable success in ground-state calculations, entanglement growth hinders their application to quantum dynamics. Quantum computing may offer a route to quantum advantage, but algori...

[View on arXiv](http://arxiv.org/abs/2609.29337v1) | [PDF](https://arxiv.org/pdf/2609.29337v1)

---

### [Strong matchgate designs in nearly optimal depth](http://arxiv.org/abs/2609.26677v1)
**Authors:** Maxwell West, M. Cerezo, Martin Larocca  
**Published:** 2026-09-22  
**Updated:** 2026-09-22  
**Categories:** quant-ph  

**Abstract:** Understanding the resources required to generate approximately random unitaries over various groups is a natural goal of quantum information theory. With respect to one notion of approximation, that of a design, it is known that the full unitary group can be approximated in logarithmic depth by one-dimensional circuits of nearest-neighbour 2-local gates. On the other hand, remarkably, circuits wit...

[View on arXiv](http://arxiv.org/abs/2609.26677v1) | [PDF](https://arxiv.org/pdf/2609.26677v1)

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

### [Efficient learning of Clifford disentanglers and typical $t$-doped unitaries with exponentially more $T$ gates](http://arxiv.org/abs/2609.27565v1)
**Authors:** Gerard Aguilar, Sofiene Jerbi, Jens Eisert et al.  
**Published:** 2026-09-23  
**Updated:** 2026-09-23  
**Categories:** quant-ph  

**Abstract:** Highly entangled and highly non-stabilizer quantum states need not be hard to learn. We give efficient algorithms for testing and recovering hidden tensor-product structure in unknown pure state vectors of the form $\lvertψ\rangle = U_C \bigotimes_i \lvertψ_i\rangle$, where $U_C$ is an arbitrary unknown Clifford unitary. Although the Clifford can thoroughly scramble the visible product structure, ...

[View on arXiv](http://arxiv.org/abs/2609.27565v1) | [PDF](https://arxiv.org/pdf/2609.27565v1)

---

### [The LHC is not enough: the LZ High-Recoil Event at FCC-hh and a Muon Collider](http://arxiv.org/abs/2609.26870v1)
**Authors:** Benedikt Maier, Michael Spannowsky  
**Published:** 2026-09-22  
**Updated:** 2026-09-22  
**Categories:** hep-ph, hep-ex, hep-th  

**Abstract:** The high-energy nuclear-recoil candidate reported by LUX-ZEPLIN could point to dark matter beyond the reach of the LHC. We examine how future colliders could test this interpretation, starting with the thermal Higgsino and extending to a broad class of electroweak multiplets containing neutral and charged particles. We assume an approximately pure multiplet protected by an exact stabilizing symmet...

[View on arXiv](http://arxiv.org/abs/2609.26870v1) | [PDF](https://arxiv.org/pdf/2609.26870v1)

---

### [Exact Quantum Circuit Optimization is co-NQP-hard](http://arxiv.org/abs/2510.16420v3)
**Authors:** Adam Husted Kjelstrøm, Andreas Pavlogiannis, Jaco van de Pol  
**Published:** 2025-10-18  
**Updated:** 2026-09-23  
**Categories:** quant-ph, cs.CC  

**Abstract:** As quantum computing resources remain scarce and error rates high, minimizing the resource consumption of quantum circuits is essential for achieving practical quantum advantage. Here we consider the natural problem of, given a circuit $C$, computing a circuit $C'$ that behaves equivalently on a desired subspace, and that minimizes a quantum resource type, expressed as the count or depth of (i) ar...

[View on arXiv](http://arxiv.org/abs/2510.16420v3) | [PDF](https://arxiv.org/pdf/2510.16420v3)

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
