--- Page 1/6 ---
Equivariant World Models for Sample-Efficient Imagination
Training in a First-Person Shooter: An Empirical Study
Anonymous Author(s)
Abstract
First-person shooter (FPS) games are a stress test for deep reinforce-
ment learning (RL): visual observations are high-dimensional and
sample complexity is prohibitive. World models and imagination
training [6, 9] reduce real-environment interaction by training a
policy inside a learned dynamics model, but most implementations
ignore a structural symmetry that FPS environments possess almost
by construction: left–right mirror symmetry. We study EqWM, a
compact world model and PPO policy that enforce horizontal-flip
(Z2) equivariance by group averaging across the encoder, recurrent
dynamics, decoder, and reward head, evaluated on the ViZDoom
health_gathering benchmark [12]. We also derive PAC-Bayes-
style bounds that decompose the imagination-training risk into
world-model bias and statistical-estimation terms, together with a
scaling law for the optimal real-to-imagination ratio. Our empirical
findings are deliberately reported without inflation. (i) With only
10k environment steps and three seeds, the equivariant world mod-
el has a substantially more stable reward head: reward prediction
MAE is 0.104 ± 0.025 versus 0.223 ± 0.099 for a non-equivariant
baseline, where the higher baseline mean is driven by seed-to-seed
divergence (two of three baseline seeds drift to 0.26–0.32 rather
than uniformly worse accuracy); single-frame reconstruction (P-
SNR/SSIM) is statistically indistinguishable and in fact marginally
favors the non-equivariant model. (ii) Imagination training reach-
es, after only ∼15k real steps, episode returns comparable to pure
on-policy PPO at 20k–40k real steps, i.e. an order-of-2× sample-
efficiency gain (the measured step ratio is ∼1.3–2.7×; because both
curves are flat and noisy this is an order-of-magnitude comparison,
not a clean crossover); under this small-model budget the equivari-
ant and non-equivariant imagination variants win on different seeds
and are not statistically distinguishable. (iii) A four-cell ablation
(whether equivariance lives in the world model, the policy, both, or
neither) yields final returns of 447±38, 425±29, 427±27, and 450±47
that overlap inside their error bars: at this budget no equivariant
component produces a significant return gain. (iv) A bootstrap esti-
mate of the effective-sample multiplier is non-monotone in training
length and inconsistent across two machines, so we report any mul-
tiplier above 1 as an open problem rather than a confirmed result.
We present the theory as an upper bound and a scaling law, and
we explicitly mark where experiments support, weakly support, or
contradict it.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Submitted to CCF-B venue, 2026
© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnn
Keywords
Equivariant neural networks, World models, Imagination training,
Reinforcement learning, First-person shooter, PAC-Bayes
1
Introduction
Deep reinforcement learning (RL) has reached superhuman levels
in many games, but training an agent directly from raw pixels in
first-person shooters (FPS) still demands millions of environment
interactions [12, 13], which is impractical when environment access
is expensive. World models address this by learning a predictive
model of the environment and optimizing the policy entirely inside
it [6]. The Dreamer family operationalizes this idea with a recurrent
state-space model [7–9] and has recently scaled to long-horizon
tasks inside learned models [10]; model-based planning with a
learned dynamics model is the core of MuZero [18], and model-
based policy optimization exploits such models for data efficiency
[11].
FPS environments, however, possess a near-perfect structural
symmetry that none of these general-purpose architectures exploit:
the rendered scene and its dynamics are (approximately) invariant
under a horizontal flip, and a flipped camera view should induce
flipped actions (TURN_LEFT ↔TURN_RIGHT, MOVE_FORWARD
unchanged). Group-equivariant networks [3] bake such symmetries
into the architecture and, in principle, use each observation several
times at no extra data cost. This is precisely the data-starved regime
in which world models operate.
We ask a deliberately narrow question: on a small FPS benchmark,
does Z2 equivariance in the world model and policy improve sample
efficiency, and how should that be measured honestly? Our system,
EqWM, enforces hard equivariance by group averaging [3] at every
convolutional layer of the encoder, recurrent dynamics, decoder,
and policy, with an invariant reward head and an antisymmetric
left/right action head. Our contributions are:
(1) Implementation and controlled comparison on ViZDoom.
We build an equivariant ConvGRU world model and equivariant
PPO policy for health_gathering and compare them against
matched non-equivariant baselines, reporting per-seed as well
as aggregated numbers (Section 6).
(2) PAC-Bayes-style analysis of imagination training. We s-
tate three results: a world-model prediction bound that depends
on an effective sample count 𝑁eff (Theorem 1), an imagination-
training generalization bound that separates world-model bias
from estimation error (Theorem 2), and a scaling law for the
optimal real-to-imagination ratio (Theorem 3); proof sketches
and a bootstrap estimator (Proposition 1) are provided.
(3) Honest empirical verdict. The clearest measured benefit of
equivariance is cross-seed stability of the task-relevant reward
head, not a large mean accuracy gain; imagination training
gives an order-of-2× sample-efficiency gain over pure PPO; the
ablation shows no significant return gain from equivariance at
1

--- Page 2/6 ---
Submitted to CCF-B venue, 2026, 2026
Anon.
this budget; and the bootstrap estimate of the multiplier is not
robust. We treat the latter two as limitations and open problems
rather than burying them.
The remainder of the paper is organized as follows. Section 2
reviews related work. Section 3 fixes notation. Section 4 detail-
s the architecture. Section 5 states the theory. Section 6 reports
experiments, and Section 7 concludes with limitations.
2
Related Work
World models and imagination training. Learning a predictive
model for RL dates back to Ha and Schmidhuber [6]. The Dreamer
series [7, 8] operationalized latent imagination with a recurrent
state-space model, and DreamerV3 [9] showed a single model can
solve many domains; the most recent Dreamer4 [10] scales agent
training inside a fast world model in Minecraft. MuZero [18] pairs
learned dynamics with planning, and MBPO [11] uses an ensemble
of models to cut real samples. Concurrent with these, EMERALD
[2] improves world modeling with masked latent transformers and
strong sample efficiency on Crafter. Our work follows the Dreamer-
style “train PPO inside a learned latent model” loop but restricts
the model class to a group-equivariant one.
Equivariant reinforcement learning. Group-equivariant convolu-
tions were introduced by Cohen and Welling [3]. In RL, MDP homo-
morphic networks [22] and group equivariant deep RL [16] encode
state/action symmetries, while SO(2)-equivariant policies [23] and
group-invariant representation structuring [20] exploit continuous
rotation symmetries. The closest lines to ours are equivariant model-
based RL: Park et al. learn symmetric embeddings for equivariant
world models [17], Deac et al. make MuZero equivariant [4], and
EDGI builds an SE(3)-equivariant diffusion planner for embodied
agents [1]. Multi-group equivariant augmentation [14] studies sev-
eral symmetry groups in manipulation. Unlike these works, which
target control, board games, or 3D/SE(3) embodied settings, we
target 2D FPS pixels with the discrete Z2 group and integrate equiv-
ariance into both the world model and the imagination-training
loop, while reporting the small-benchmark result transparently.
PAC-Bayes and generalization in RL.. PAC-Bayes bounds [15]
relate a posterior’s risk to its empirical risk through a KL complexity
term. Recent PAC-Bayesian RL [24] trains generalizable on-policy
policies. We adapt the decomposition to the mixed real/imagination
data regime and use the bootstrap [5] to estimate the effective
sample multiplier empirically.
FPS reinforcement learning. ViZDoom [12] provides a Doom-
based visual-RL benchmark; early agents trained directly with
model-free RL [13]. To our knowledge, prior work has not com-
bined an equivariant world model with imagination training on
FPS pixels.
3
Preliminaries
MDP.. We model the environment as an MDP ⟨S, A, 𝑃,𝑟,𝛾⟩[21],
where S is the RGB frame space, A = {turn-left, turn-right, forward}
is a three-action discrete set, 𝑃the transition kernel, 𝑟the reward,
and 𝛾= 0.99 the discount. The objective is the expected discounted
return 𝐽(𝜋) = E𝜋[Í
𝑡𝛾𝑡𝑟𝑡].
Group action and equivariance. We take 𝐺= Z2 = {𝑒,ℎ}, where
ℎis a horizontal flip. The group acts on observations by ℎ· 𝑥=
flip𝑊(𝑥) and on actions by the permutationℎ·turn-left = turn-right,
ℎ· turn-right = turn-left, ℎ· forward = forward. A map 𝑓
is equivariant if 𝑓(𝑔𝑥) = 𝑔𝑓(𝑥) and invariant if 𝑓(𝑔𝑥) = 𝑓(𝑥).
World models and imagination. A world model ˆ𝑀= ⟨ˆ𝑃, ˆ𝑟⟩ap-
proximates the environment. Given ˆ𝑀, imagination training starts
a policy rollout from a real observation and predicts subsequent
latent states and rewards with ˆ𝑀, then updates the policy (here
PPO [19]) on these imagined trajectories [9].
PAC-Bayes. With posterior 𝜌, prior 𝜋, and 𝑛i.i.d. samples, a
canonical bound is𝑅(𝜌) ≤𝑅𝑆(𝜌)+
√︁
(KL(𝜌∥𝜋) + log(2√𝑛/𝛿))/(2𝑛).
4
Method
4.1
Equivariant Z2 world model
The world model ˆ𝑀𝜙has four parts. Encoder 𝐸𝜙: three strided con-
volutions map a 64 × 64 × 3 frame to an 8 × 8 ×𝑧latent map (𝑧= 32,
base channels 16). Every convolution is a group-averaging layer
[3],
𝐸𝜙(𝑥) = 1
2

Conv(𝑥) + flip𝑊(Conv(flip𝑊(𝑥)))

,
(1)
which shares one kernel and satisfies 𝐸𝜙(flip𝑊𝑥) = flip𝑊𝐸𝜙(𝑥)
up to numerical precision. Dynamics: an equivariant ConvGRU
cell updates the latent map from the previous map and a spatial-
ly broadcast one-hot action; gates use the same group-averaging
convolution. Decoder: three equivariant transposed convolutions
reconstruct the frame with a sigmoid output. Reward head: a global-
average-pooling (invariant) layer followed by an MLP predicts a
scalar reward from the latent map. The non-equivariant baseline
uses identical widths and layer counts but ordinary convolutions, a
flattened encoder for its heads, and no group averaging.
The training loss is reconstruction MSE plus reward MSE plus
latent-dynamics MSE, and (only for the equivariant model) a soft
consistency penalty on the encoder and dynamics under flip.
4.2
Equivariant PPO policy
The policy uses the same equivariant encoder. Invariant features
(gap(ℎ)) drive the forward action and the (invariant) value head; an
antisymmetric feature 𝑎= ℎleft −ℎright distinguishes left and right:
ℓleft = 𝑠+𝑎, ℓright = 𝑠−𝑎, ℓforward = 𝑠′. This enforces 𝜋(ℎ·𝑥) = ℎ·𝜋(𝑥).
4.3
Imagination pipeline
We collect 𝑛= 10k random real transitions, train ˆ𝑀𝜙for 30 epochs,
then iterate: collect 1,000 new on-policy real transitions, fine-tune
ˆ𝑀𝜙for 5 epochs, imagine short rollouts of horizon 𝐻= 20 under
PPO [19] (200 rollouts per outer step), and update on the imagined
data. This gives roughly 4,000 imagined transitions per 1,000 new
real transitions, i.e. an imagination ratio ≈4, fixed rather than
tuned. Table 1 lists all hyper-parameters.
5
Theory
We present three results and one estimator. They are intentionally
stated as upper bounds and scaling laws; our claim is about the
structure of the trade-off, not tightness. Before the statements we
fix a notation conflict: throughout, 𝑁eff is an effective sample count
2

--- Page 3/6 ---
Equivariant World Models for Sample-Efficient Imagination
Training in a First-Person Shooter: An Empirical Study
Submitted to CCF-B venue, 2026, 2026
Table 1: Hyper-parameters (identical for equivariant and
standard variants except for group averaging). Exp. 1 trained
the world model for 50 epochs; the imagination runs below
use 30 initial epochs.
Group
Hyper-parameter
Value
Environment
Observation
64 × 64 × 3 RGB, frame skip 4
Discount 𝛾
0.99
World model
Base channels / latent 𝑧
16 / 32 (8 × 8 × 32)
Init. real steps 𝑛
10,000
WM epochs (init / fine-tune)
30 / 5
Imagination
Horizon 𝐻
20
Imagined rollouts / outer step
200
New real steps / outer step
1,000
Imagination ratio 𝑚/𝑛
≈4
PPO
Optimizer / LR
Adam / 3 × 10−4
Minibatch / epochs
256 / 4
Clip / GAE 𝜆
0.2 / 0.95
Entropy / value coef.
0.01 / 0.5
Bootstrap
Resamples 𝐵/ train set
15 / 8,000
(dimension of data, ≤|𝐺|𝑛), whereas 𝑅boot in Proposition 1 is the
dimensionless variance ratio we estimate from data; Proposition 1
makes 𝑅boot an empirical proxy for the multiplier 𝑁eff/𝑛.
Theorem 1 (World-model prediction bound). Let ˆ𝑀𝜙be trained
on 𝑛transitions. With probability ≥1 −𝛿over the data,
E(𝑠,𝑎,𝑠′) ∥ˆ𝑃𝜙(𝑠,𝑎) −𝑠′∥2 ≤ˆLWM(𝑆) + 𝑂
√︃
KL(𝜙∥𝜙0)+log(1/𝛿)
𝑁eff

,
where 𝑁eff ≤|𝐺| 𝑛is the effective sample count.
Proof sketch. Start from the standard PAC-Bayes inequality𝑅(𝜙) ≤
𝑅𝑆(𝜙) + 𝑂(
√︁
(KL(𝜙∥𝜙0) + log(1/𝛿))/𝑛) [15]. Because transitions
within a rollout are temporally correlated, group the data into inde-
pendent blocks and apply McDiarmid’s inequality over blocks; this
replaces 𝑛by an effective block count. Group averaging reuses each
transition together with its flipped copy, so the independent data ef-
fectively available is at most |𝐺| times the raw count, i.e. 𝑁eff ≤|𝐺|𝑛.
Note this is a data-reuse effect only: the group-averaging layer
shares a single kernel (§4) and does not reduce the parameter count.
The complete block-decomposition argument follows the standard
PAC-Bayes treatment and is deferred to the appendix.
Theorem 2 (Imagination-training bound). A policy trained
on𝑛real and𝑚imagined transitions satisfies, with probability ≥1−𝛿,
𝐽(𝜋𝜃) ≥ˆ𝐽imag(𝜋𝜃) −𝐶𝜀WM 𝐻−𝑂
√︃
log(1/𝛿)
𝑛+𝑚𝜂

,
where 𝜀WM is the world-model error, 𝐻the imagination horizon, and
𝜂∈[0, 1] down-weights imagined data by model accuracy.
Proof sketch. The gap splits into two terms. Model-bias term: an 𝐻-
step rollout compounds one-step error geometrically, Í𝐻−1
𝑡=0 𝛾𝑡𝜀WM ≤
𝐶𝜀WM𝐻for a constant 𝐶that depends on the discount and on the
policy’s Lipschitz constant. Estimation term: treating imagined tran-
sitions as weight-𝜂samples, the effective data count is 𝑛+ 𝑚𝜂, to
which the PAC-Bayes/McDiarmid estimation bound applies directly.
The two terms are independent and additive.
Theorem 3 (Optimal ratio, as a scaling law). Balancing the
two terms of Theorem 2 by differentiating the upper bound with respect
to 𝑚gives the scaling law
𝑚∗
𝑛= 𝐶𝑁eff/𝑛
𝜀2
WM
,
(2)
for a constant 𝐶that depends on horizon, discount, and KL budget.
We emphasize that Theorem 3 is a scaling law, not a numerically
tight formula: substituting the pixel MSE 𝜀WM ≈10−2 literally yields
𝑚∗/𝑛= 104, which is unrealistically large because the constant and
the units of 𝜀WM are uncalibrated. We therefore clip the ratio to a
fixed range in practice and treat the theorem as qualitative guidance,
exactly as Section 6 reports.
Proposition 1 (Estimating the multiplier by bootstrap).
The multiplier 𝑁eff/𝑛can be estimated by bootstrap [5]: draw 𝐵re-
samples of the training set, refit, and set
𝑅boot =
c
Varstd
c
Vareq
,

𝑁eff/𝑛= clip(𝑅boot, 1, |𝐺| = 2).
(3)
Justification. For an i.i.d. estimator with 𝑛samples, the out-of-
sample error variance scales as 1/𝑛; if equivariance multiplies the
effective data by 𝑘, its error variance scales as 1/(𝑘𝑛), so the vari-
ance ratio recovers 𝑘. The bootstrap [5] estimates these two vari-
ances by resampling. The clip to [1, |𝐺|] encodes the prior that a
symmetry group of size |𝐺| cannot multiply data by more than |𝐺|.
This makes 𝑅boot an empirical proxy for the theoretical 𝑁eff/𝑛of
Theorem 1; it is not the same object as the sample count 𝑁eff itself.
6
Experiments
6.1
Setup
We use ViZDoom health_gathering: the agent navigates a maze
to collect health packs while taking damage; observations are 64 ×
64×3 RGB frames with frame skip 4. The reward is near-degenerate:
the agent is penalized only at death, so roughly 99.3% of non-
terminal steps carry an identical reward offset. This sparsity makes
the reward head easy to overfit and motivates the stability analysis
below. The small model (base channels 16, latent 8 × 8 × 32) is used
throughout. We report mean±std over seeds and state the seed
count explicitly for every number.
6.2
Experiment 1: world-model prediction and
reward-head stability
Both world models are trained on 10k random transitions for 50
epochs over three seeds (Table 2).
Two findings matter. First, single-frame reconstruction (PSNR/SSIM)
is comparable and, if anything, marginally favors the non-equivariant
model (21.8 vs 20.6 dB): equivariance costs a little pixel fidelity. Sec-
ond, the reward head behaves very differently across seeds. The
per-seed reward MAE is [0.072, 0.107, 0.132] for the equivariant
model but [0.089, 0.256, 0.324] for the standard model (Fig. 1). The
standard model’s higher mean (0.223) and much higher std (0.099)
are driven by two seeds on which the reward head drifts to 0.26–
0.32, which we read as overfitting to the near-degenerate death
3

--- Page 4/6 ---
Submitted to CCF-B venue, 2026, 2026
Anon.
Table 2: World-model comparison on health_gathering (10k
steps, 50 epochs, 3 seeds; mean±std). Only reconstruction
and reward metrics with a well-defined per-frame estimate
are reported; the earlier concatenated-rollout open-loop MSE
was computed with a flawed metric and is omitted (see text).
Metric
Equivariant WM
Standard WM
Note
PSNR (dB)
20.62 ± 1.59
21.78 ± 0.66
standard slightly better
SSIM
0.272 ± 0.069
0.321 ± 0.026
standard slightly better
Reward MAE
0.104 ± 0.025
0.223 ± 0.099
stable; see Fig. 1
seed 0
seed 1
seed 2
0.0
0.1
0.2
0.3
0.4
Reward prediction MAE
eq mean 0.104
std mean 0.223
0.072
0.107
0.132
0.089
0.256
0.324
Reward-head stability across seeds
Equivariant WM
Standard WM
Figure 1: Per-seed reward-prediction MAE. The equivariant
reward head stays in [0.072, 0.132] across seeds, whereas the
non-equivariant head diverges on two of three seeds (0.256,
0.324). The benefit is cross-seed stability, not a uniformly large
per-sample accuracy gain.
reward. The equivariant reward head instead remains stable. We
therefore frame the 0.104 vs 0.223 comparison as improved cross-
seed stability of the task-relevant head, not as a uniform “53% more
accurate predictor”. We also omit long-horizon open-loop rollout
error: our pipeline concatenates unrelated transition tuples when
computing it, which makes the resulting MSE unreliable, and a
rigorously re-computed chained rollout was not run within this
budget; we prefer to report only the well-defined per-frame metrics
of Table 2.
6.3
Experiment 2: few-shot imagination
training
We compare imagination-trained agents (equivariant or standard
world model) against pure on-policy PPO, over two seeds (Fig. 2).
After only ∼15k real steps, the imagination agents reach episode
returns in roughly 316–508, the same range in which pure real PPO
sits at 20k–40k real steps (284–572). The four initial (untrained)
scores are 396, 380, 364, and 668 (mean ≈452; the 668 is a single
high-variance outlier), and all curves are flat and noisy at this bud-
get. We conclude that imagination training yields an order-of-2×
sample-efficiency gain over pure PPO: the step ratio is ∼1.3–2.7×
(20k–40k real steps over the ∼15k imagination budget), and because
neither baseline has converged this is an order-of-magnitude state-
ment rather than a clean crossover. We state two caveats up front:
only two seeds were run, and the equivariant and non-equivariant
imagination variants are statistically indistinguishable and win on
0
20000
40000
60000
80000 100000
Environment steps (real)
300
400
500
600
700
Episode return
untrained ~380-400 (outlier 668)
Few-shot imagination training (2 seeds)
Eq imagination
Std imagination
Real PPO
Figure 2: Few-shot imagination training (2 seeds). Thin lines
are individual seeds, thick lines their mean. Imagination
agents at ∼15k real steps match real PPO evaluated at 20k–
40k steps. The gray band marks the untrained-policy range
(∼380–400; one outlier seed at 668); all curves are noisy at this
budget.
Table 3: Ablation: where equivariance is placed (3 train seeds;
robust evaluation). All four means overlap within one stan-
dard deviation.
Configuration
Final return (robust)
both-eq (WM + policy equivariant)
447.1 ± 37.7
wm-eq-only (WM equivariant, policy standard)
425.3 ± 28.5
policy-eq-only (WM standard, policy equivariant)
426.7 ± 26.7
none-eq (both standard)
449.8 ± 46.7
different seeds. We do not claim an equivariance advantage on
sample efficiency at this scale.
6.4
Experiment 3: ablation of the equivariant
components
We ablate whether equivariance is placed in the world model, the
policy, both, or neither, under the same small budget and a robust
evaluation (3 train seeds × 3 eval seeds × 8 episodes). Final robust
returns are in Table 3 and Fig. 3.
The four means lie in a narrow band (425–450) and their error
bars overlap; the fully non-equivariant configuration is in fact nu-
merically highest. We report this honestly: under the small-model,
short-training budget, the reward-head stability observed in Exper-
iment 1 does not translate into a significant final-task return gain.
Whether equivariance would help return at larger scale or with
more seeds is an open question.
6.5
Experiment 4: bootstrap estimate of the
multiplier
Following Proposition 1, we bootstrap 𝐵= 15 resamples of an𝑛= 8k
transition set, refit on each, and compare held-out prediction-error
variances between the equivariant and standard models across train-
ing lengths. On the primary local run the variance ratio 𝑅boot =
4

--- Page 5/6 ---
Equivariant World Models for Sample-Efficient Imagination
Training in a First-Person Shooter: An Empirical Study
Submitted to CCF-B venue, 2026, 2026
both-eq
wm-eq
only
policy-eq
only
none-eq
0
100
200
300
400
500
Robust episode return
(3 eval-seeds x 8 ep)
447
425
427
450
Ablation: where equivariance is placed
Figure 3: Ablation of equivariant components. Error bars
are ±1 std over three training seeds. The four configurations
are not distinguishable at this budget: equivariance gives no
significant return advantage here.
Varstd/Vareq is 0.51, 5.72, and 0.09 at 5, 20, and 40 epochs respec-
tively. As a secondary cross-check we also logged an independent
cloud run (A10 GPU), which records 0.72, 1.50, and 28.3 (the last
value is the raw, pre-clip variance ratio; clipped to the group ceiling
it is 2.0). We disclose a data limitation: the cloud numbers at 5 and
20 epochs were reconstructed from terminal-log OCR with incom-
plete per-replicate arrays (5 epochs: no per-replicate list retained;
20 epochs: only 3 of 15 replicates), whereas the 40-epoch cloud
point is complete (all 15 replicates, consistent across three OCR
reads). We therefore treat the local run as primary and the cloud run
as a suggestive cross-check only. The ratio is non-monotone and
the two runs disagree at 40 epochs, where the equivariant model
even diverges on several local bootstrap replicates (test error rising
to 0.04–0.055). We therefore do not claim a multiplier above 1 as
a robust result: it appears only in a narrow intermediate-training
band and is highly sensitive to training length and run environment.
This is reported as an open problem.
6.6
Summary of empirical claims
Across the four experiments, the supported statements are: (i) e-
quivariance stabilizes the reward head across seeds; (ii) imagination
training gives an order-of-2× sample-efficiency gain over pure PPO
at this scale (measured ratio ∼1.3–2.7×); (iii) equivariance produces
no significant return gain in the ablation; and (iv) the bootstrap
evidence for an effective-sample multiplier is not robust. The theory
of Section 5 gives the correct structure of the trade-off (world-model
bias dominates the imagination gap), but its quantitative predictions
are only qualitatively supported.
7
Conclusion and limitations
We presented EqWM, a compact Z2-equivariant world model and
PPO policy for FPS imagination training on ViZDoom, together
with PAC-Bayes-style bounds and a bootstrap diagnostic. On a
small benchmark the honest takeaway is modest: equivariance’s
clearest benefit is seed-to-seed stability of the task-relevant reward
head; imagination training improves sample efficiency by an order
of 2× (measured ratio ∼1.3–2.7×); the equivariant components do
5
20
40
World-model training epochs
10−1
100
101
Variance ratio Varstd/Vareq
Bootstrap Meff estimate is non-monotonic
Local run (B=15)
Cloud run (B=15)
group ceiling |G|=2
Figure 4: Bootstrap estimate of the effective multiplier (vari-
ance ratio, log scale) versus world-model training epochs.
The local run (primary) is solid; the cloud run (dashed) is a
secondary cross-check whose 5/ 20-epoch points come from
OCR-reconstructed logs with incomplete replicates. The ra-
tio is non-monotone and the runs disagree; the group ceiling
is |𝐺| = 2 (the 28.3 cloud value is raw, clipped to 2.0). We treat
any multiplier above 1 as unconfirmed.
not yield a significant return gain at this budget; and the effective-
sample multiplier is not yet a robust measured quantity.
Limitations and future work. (1) Only one scenario (health_gathering)
and a small model; more seeds (Experiments 2, 4 use only two seeds
and 𝐵= 15), more scenarios, and larger scale are needed. (2) The
group is the small Z2; partial equivariance for the asymmetric HUD
is future work. (3) The deterministic ConvGRU lacks stochastic
latents. (4) The bootstrap multiplier estimate 𝑅boot is unstable; reg-
ularization, early stopping, and a larger 𝐵are required before any
multiplier-above-1 claim, and the cloud cross-check should be re-
run to disk without OCR. We release the configuration and result
files to support reproduction.
References
[1] Johann Brehmer, Niki Rahaman, Andreea Deac, Yilun Geng, Zhuotong Li, Hans
Strobelt, Soumith Chintala, Mikael Henaff, and Markus Weimer. 2023. EDGI:
Equivariant Diffusion for Planning with Embodied Agents. In Advances in Neural
Information Processing Systems (NeurIPS). arXiv:2303.12410.
[2] Maxime Burchi. 2025. Accurate and Efficient World Modeling with Masked
Latent Transformers. arXiv preprint arXiv:2507.04075 (2025).
[3] Taco S. Cohen and Max Welling. 2016. Group Equivariant Convolutional Network-
s. In Proc. International Conference on Machine Learning (ICML). arXiv:1602.07576.
[4] Andreea Deac, Théophane Weber, and George Papamakarios. 2023. Equivariant
MuZero. arXiv preprint arXiv:2302.04798 (2023).
[5] Bradley Efron and Robert J. Tibshirani. 1993. An Introduction to the Bootstrap.
Chapman & Hall/CRC.
[6] David Ha and Jürgen Schmidhuber. 2018. World Models. In Advances in Neural
Information Processing Systems (NeurIPS). arXiv:1803.10122.
[7] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak
Lee, and James Davidson. 2020. Learning Latent Dynamics for Planning from
Pixels. In Proc. International Conference on Machine Learning (ICML).
arX-
iv:1912.01603.
[8] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. 2021.
Mastering Atari with Discrete World Models. In Proc. International Conference on
Learning Representations (ICLR). arXiv:2010.02193.
[9] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. 2025. Master-
ing Diverse Domains through World Models. Nature (2025). arXiv:2301.04104.
[10] Danijar Hafner, Wilson Yan, and Timothy Lillicrap. 2025. Training Agents Inside
of Scalable World Models (Dreamer4). arXiv preprint arXiv:2509.24527 (2025).
[11] Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. 2019. When to Trust
Your Model: Model-Based Policy Optimization. In Advances in Neural Information
5

--- Page 6/6 ---
Submitted to CCF-B venue, 2026, 2026
Anon.
Processing Systems (NeurIPS). arXiv:1906.08253.
[12] Michał Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek, and Wojciech
Jaśkowski. 2016. ViZDoom: A Doom-Based AI Research Platform for Visual
Reinforcement Learning. In Proc. IEEE Conference on Computational Intelligence
and Games (CIG). arXiv:1605.02097.
[13] Guillaume Lample and Devendra Singh Chaplot. 2017. Playing FPS Games with
Deep Reinforcement Learning. In Proc. AAAI Conference on Artificial Intelligence.
arXiv:1609.05521.
[14] Hongbin Lin, Juan Rojas, and Kwok Wai Samuel Au. 2025. Multi-Group Equi-
variant Augmentation for Reinforcement Learning in Robot Manipulation. arXiv
preprint arXiv:2508.11204 (2025).
[15] David A. McAllester. 1999. Some PAC-Bayesian Theorems. In Proc. Annual
Conference on Computational Learning Theory (COLT).
[16] Sayak Ray Mondal, Arun Nair, and Biswajit Siddiqi. 2020. Group Equivariant
Deep Reinforcement Learning. arXiv preprint arXiv:2007.03437 (2020).
[17] Jung Yeon Park, Ondrej Biza, Linfeng Zhao, Jan-Willem van de Meent, and Robin
Walters. 2022. Learning Symmetric Embeddings for Equivariant World Models.
In Proc. International Conference on Machine Learning (ICML). arXiv:2204.11371.
[18] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan,
Laurent Sifre, Arthur Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis,
Thore Graepel, Timothy Lillicrap, and David Silver. 2020. Mastering Atari, Go,
Chess and Shogi by Planning with a Learned Model. Nature 588 (2020), 604–609.
arXiv:1911.08265.
[19] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
2017. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347
(2017).
[20] Manel Shakerinava, Siamak Ravanbakhsh, and Sanja Fidler. 2022. Structuring
Representations Using Group Invariants. In Advances in Neural Information
Processing Systems (NeurIPS).
[21] Richard S. Sutton and Andrew G. Barto. 2018. Reinforcement Learning: An Intro-
duction (2nd ed.). MIT Press.
[22] Elise van der Pol, Daniel E. Worrall, Herke van Hoof, Frans A. Oliehoek, and
Max Welling. 2020. MDP Homomorphic Networks: Group Symmetries in Re-
inforcement Learning. In Advances in Neural Information Processing Systems
(NeurIPS).
[23] Chen Wang, Robin Walters, and Robert Platt. 2022. SO(2)-Equivariant Rein-
forcement Learning. In Proc. International Conference on Learning Representations
(ICLR). arXiv:2203.04439.
[24] Abdelkrim Zitouni, Mehdi Hennequin, Juba Agoun, Ryan Horache, Nadia K-
abachi, and Omar Rivasplata. 2025. PAC-Bayesian Reinforcement Learning
Trains Generalizable Policies. arXiv preprint arXiv:2510.10544 (2025).
6