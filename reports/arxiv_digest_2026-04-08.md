# arXiv Daily Digest - 2026-04-08

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


### [LieTrunc-QNN: Lie Algebra Truncation and Quantum Expressivity Phase Transition from LiePrune to Provably Stable Quantum Neural Networks](http://arxiv.org/abs/2604.02697v1)
**Authors:** Haijian Shao, Dalong Zhao, Xing Deng et al.  
**Published:** 2026-04-03  
**Updated:** 2026-04-03  
**Categories:** cs.LG  

**Abstract:** Quantum Machine Learning (QML) is fundamentally limited by two challenges: barren plateaus (exponentially vanishing gradients) and the fragility of parameterized quantum circuits under noise. Despite extensive empirical studies, a unified theoretical framework remains lacking.   We introduce LieTrunc-QNN, an algebraic-geometric framework that characterizes trainability via Lie-generated dynamics. ...

[View on arXiv](http://arxiv.org/abs/2604.02697v1) | [PDF](https://arxiv.org/pdf/2604.02697v1)

---

### [Classical shadows with arbitrary group representations](http://arxiv.org/abs/2604.01429v1)
**Authors:** Maxwell West, Frederic Sauvage, Aniruddha Sen et al.  
**Published:** 2026-04-01  
**Updated:** 2026-04-01  
**Categories:** quant-ph  

**Abstract:** Classical shadows (CS) has recently emerged as an important framework to efficiently predict properties of an unknown quantum state. A common strategy in CS protocols is to parametrize the basis in which one measures the state by a random group action; many examples of this have been proposed and studied on a case-by-case basis. In this work, we present a unified theory that allows us to simultane...

[View on arXiv](http://arxiv.org/abs/2604.01429v1) | [PDF](https://arxiv.org/pdf/2604.01429v1)

---

### [Quantum Fisher Information for Entropy of Gibbs States](http://arxiv.org/abs/2603.16456v3)
**Authors:** Francis J. Headley  
**Published:** 2026-03-17  
**Updated:** 2026-04-07  
**Categories:** quant-ph, cond-mat.stat-mech  

**Abstract:** We derive the quantum Fisher information for entropy estimation in a Gibbs state and show that it equals the inverse of the heat capacity, which is dual to the temperature Fisher information given by the heat capacity divided by the square of the temperature. Their product is independent of the Hamiltonian and depends only on the temperature, leading to a metrological uncertainty relation between ...

[View on arXiv](http://arxiv.org/abs/2603.16456v3) | [PDF](https://arxiv.org/pdf/2603.16456v3)

---

### [Codimension-controlled universality of quantum Fisher information singularities at topological band-touching defects](http://arxiv.org/abs/2604.01515v1)
**Authors:** C. A. S. Almeida  
**Published:** 2026-04-02  
**Updated:** 2026-04-02  
**Categories:** quant-ph, cond-mat.mes-hall  

**Abstract:** Topological phase transitions in generic multiband systems are mediated by band-touching defects whose codimension -- the number of momentum directions along which the gap closes linearly -- varies across universality classes. Although singular behavior of fidelity susceptibilities and quantum Fisher information (QFI) has been computed for specific models, no unifying principle connecting these re...

[View on arXiv](http://arxiv.org/abs/2604.01515v1) | [PDF](https://arxiv.org/pdf/2604.01515v1)

---

### [Recurrent Quantum Feature Maps for Reservoir Computing](http://arxiv.org/abs/2604.03469v1)
**Authors:** Utkarsh Singh, Aaron Z. Goldberg, Christoph Simon et al.  
**Published:** 2026-04-03  
**Updated:** 2026-04-03  
**Categories:** quant-ph, cs.LG  

**Abstract:** Reservoir computing promises a fast method for handling large amounts of temporal data. This hinges on constructing a good reservoir--a dynamical system capable of transforming inputs into a high-dimensional representation while remembering properties of earlier data. In this work, we introduce a reservoir based on recurrent quantum feature maps where a fixed quantum circuit is reused to encode bo...

[View on arXiv](http://arxiv.org/abs/2604.03469v1) | [PDF](https://arxiv.org/pdf/2604.03469v1)

---

### [Shot-Based Quantum Encoding: A Data-Loading Paradigm for Quantum Neural Networks](http://arxiv.org/abs/2604.06135v1)
**Authors:** Basil Kyriacou, Viktoria Patapovich, Maniraman Periyasamy et al.  
**Published:** 2026-04-07  
**Updated:** 2026-04-07  
**Categories:** quant-ph, cs.AI, cs.LG  

**Abstract:** Efficient data loading remains a bottleneck for near-term quantum machine-learning. Existing schemes (angle, amplitude, and basis encoding) either underuse the exponential Hilbert-space capacity or require circuit depths that exceed the coherence budgets of noisy intermediate-scale quantum hardware. We introduce Shot-Based Quantum Encoding (SBQE), a data embedding strategy that distributes the har...

[View on arXiv](http://arxiv.org/abs/2604.06135v1) | [PDF](https://arxiv.org/pdf/2604.06135v1)

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
