# FKC Derivation and BlendSplice Steering Adaptation

## 1. Feynman–Kac Correctors (FKC) – Step-by-Step Derivation

### Step 1: Base reverse diffusion
We start from a pretrained reverse diffusion process:
p_t(x)

### Step 2: Introduce reward
Define a reward function:
h_t(x)

### Step 3: Tilted path distribution
We reweight trajectories:
p*(path) ∝ exp( ∫ h_s(x_s) ds )

### Step 4: Marginal evolution (continuous case)
∂p_t^{FK}(x)/∂t =
- ∇·(p_t^{FK}(x) v_t(x))
+ (g_t^2 / 2) Δ p_t^{FK}(x)
+ p_t^{FK}(x)(h_t(x) - E[h_t(x)])

### Step 5: Discrete case (relevant for DNA)
∂p_t^{FK}(i)/∂t =
Σ_j [A_t(j,i)p_t(j) - A_t(i,j)p_t(i)]
+ p_t(i)(h_t(i) - E[h_t(i)])

### Step 6: Particle approximation
For each timestep:
1. Propagate particles
2. Update weights: log w += h_t(x)
3. Normalize + resample

---

## 2. Adaptation to BlendSplice

### Goal
Instead of post-hoc blending:
p_final = (1-λ)p_model + λp_freq

We use trajectory steering:
p_t^{BS}(x) ∝ p_t(x) exp(λ H_t(x))

---

## 3. Reward Design

### Frequency reward
H_freq(x) = Σ_i log p̂_i(x_i | context)

### Motif reward
Encodes donor/acceptor structure

### Proxy reward
H_proxy(x) = log f(x)

### Combined
H_t(x) = w1 H_freq + w2 H_motif + w3 H_proxy

---

## 4. Algorithm (FKC-BlendSplice)

Initialize K particles

For t = T → 1:
- Sample x_{t-1} ~ pθ
- Compute H(x)
- Update weights
- Normalize
- Resample if needed

---

## 5. Simpler Approximation (Logit Steering)

Modify logits:
l'(b) = l(b) + τ * reward(b)

---

## 6. Key Insight

BlendSplice becomes:
- From post-hoc correction
→ to trajectory-level control

---

## 7. Summary

FKC transforms diffusion into a guided search process.
BlendSplice can adopt this by introducing splice-aware rewards during sampling.
