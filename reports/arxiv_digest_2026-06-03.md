# arXiv Daily Digest - 2026-06-03

**Search Period:** Last 7 days  
**Papers Found:** 3

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


### [Equivalence between exponential concentration in quantum machine learning kernels and barren plateaus in variational algorithms](http://arxiv.org/abs/2501.07433v4)
**Authors:** Pranav Kairon, Jonas Jäger, Roman V. Krems  
**Published:** 2025-01-13  
**Updated:** 2026-06-01  
**Categories:** quant-ph  

**Abstract:** We formalize a rigorous connection between barren plateaus (BP) in variational quantum algorithms and exponential concentration of quantum kernels for machine learning. Our results imply that recently proposed strategies to build BP-free quantum circuits can be utilized to construct useful quantum kernels for machine learning. This is illustrated by a numerical example employing a provably BP-free...

[View on arXiv](http://arxiv.org/abs/2501.07433v4) | [PDF](https://arxiv.org/pdf/2501.07433v4)

---

### [Mitigating Noise-Induced Barren Plateaus Using a Non-Unitary Ansatz: Application to Molecular Electronic Transport](http://arxiv.org/abs/2605.30572v1)
**Authors:** Sasanka Dowarah, Abeda Sultana Shamma, Yazdan Maghsoud et al.  
**Published:** 2026-05-28  
**Updated:** 2026-05-28  
**Categories:** quant-ph  

**Abstract:** Variational quantum algorithms (VQAs) offer a promising route toward simulating many-body quantum systems on noisy intermediate-scale quantum (NISQ) hardware. However, their scalability is severely limited by noise-induced barren plateaus (NIBPs), where hardware noise causes the gradients of the cost function to vanish exponentially with circuit depth, rendering optimization impossible. In this wo...

[View on arXiv](http://arxiv.org/abs/2605.30572v1) | [PDF](https://arxiv.org/pdf/2605.30572v1)

---

### [How hard is it to verify a classical shadow?](http://arxiv.org/abs/2510.08515v3)
**Authors:** Georgios Karaiskos, Dorian Rudolph, Johannes Jakob Meyer et al.  
**Published:** 2025-10-09  
**Updated:** 2026-05-27  
**Categories:** quant-ph, cs.CC  

**Abstract:** Classical shadows are succinct classical representations of quantum states which allow one to encode a set of properties P of a quantum state rho, while only requiring measurements on logarithmically many copies of rho in the size of P. In this work, we initiate the study of verification of classical shadows, denoted classical shadow validity (CSV), from the perspective of computational complexity...

[View on arXiv](http://arxiv.org/abs/2510.08515v3) | [PDF](https://arxiv.org/pdf/2510.08515v3)

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
