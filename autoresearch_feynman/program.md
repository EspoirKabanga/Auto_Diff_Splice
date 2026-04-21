# AutoResearch: Feynman-Kac Correctors for Diffusion

This is an experiment loop to use an LLM/agent to implement and optimize Feynman-Kac Correctors from the DISCO paper into this Splice Site Diffusion codebase to increase biological generation fidelity.

## Mission
1. Modify `diffusion_feynman.py` to implement the Feynman-Kac Corrector algorithm (e.g. SDE weighting and tilting marginals based on the prior likelihoods). You can refer to DISCO paper's continuous or discrete FKC derivations.
2. The model uses `cuda:0`. **Do not change the architecture** of the U-Net neural network or dataset processing functions. Only modify `generate_full_sequence_enhanced` or closely associated generation utilities in the reverse sampling step.
3. Your goal is to **maximize the `quality_score`** output by executing `python evaluate_sequences.py > run.log 2>&1`.

## Experimentation Setup
1. Read `diffusion_feynman.py`. Understand how it currently loops in reverse diffusion (reverse timestep loop `for i in reversed(range(timesteps)):`).
2. Read `evaluate_sequences.py` to understand evaluating logic. **Do not modify this file.**
3. Create experiments by implementing logic directly into `diffusion_feynman.py`. Use the dictionaries `full_prev_dist` and `full_next_dist` provided as args for estimating your positional reward $h_t(x)$ in the FKC math during sampling.

## Execution and Logging
1. Modify `diffusion_feynman.py` locally.
2. Run `python evaluate_sequences.py > run.log 2>&1`
3. Identify metric: `grep "^quality_score:" run.log`
4. Log to `results.tsv` using format: `commit\tquality_score\tmemory_gb\tstatus\tdescription`
5. If the score improves, keep the local code changes. If it crashes or the score decreases, revert the change. Keep iterating infinitely.
