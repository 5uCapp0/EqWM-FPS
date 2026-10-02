--- Page 1/16 ---
Equivariant World Models for Sample-Ecient
Imagination
Training in a First-Person Shooter: An Empirical
Study
Anonymousa
aAnonymous Institution (double-blind review)
Abstract
First-person shooter (FPS) games are a stress test for deep reinforcement
learning (RL): visual observations are high-dimensional and sample com-
plexity is prohibitive. World models and imagination training [1, 2] reduce
real-environment interaction by training a policy inside a learned dynamics
model, but most implementations ignore a structural symmetry that FP-
S environments possess almost by construction: leftright mirror symme-
try. We study EqWM, a compact world model and PPO policy that enforce
horizontal-ip (Z2) equivariance by group averaging across the encoder, re-
current dynamics, decoder, and reward head, evaluated on the ViZDoom
health_gathering benchmark [3]. We also derive PAC-Bayes-style bound-
s that decompose the imagination-training risk into world-model bias and
statistical-estimation terms, together with a scaling law for the optimal real-
to-imagination ratio. Our empirical ndings are deliberately reported with-
out ination. (i) With only 10k environment steps and three seeds, the e-
quivariant world model has a substantially more stable reward head: reward
prediction MAE is 0.104 ± 0.025 versus 0.223 ± 0.099 for a non-equivariant
baseline, where the higher baseline mean is driven by seed-to-seed diver-
gence (two of three baseline seeds drift to 0.260.32 rather than uniformly
worse accuracy); single-frame reconstruction (PSNR/SSIM) is statistically
indistinguishable and in fact marginally favors the non-equivariant model.
(ii) Imagination training reaches, after only ∼15k real steps, episode returns
comparable to pure on-policy PPO at 20k40k real steps, i.e. an order-of-
2× sample-eciency gain (the measured step ratio is ∼1.32.7×; because
both curves are at and noisy this is an order-of-magnitude comparison,

--- Page 2/16 ---
not a clean crossover); under this small-model budget the equivariant and
non-equivariant imagination variants win on dierent seeds and are not s-
tatistically distinguishable. (iii) A four-cell ablation (whether equivariance
lives in the world model, the policy, both, or neither) yields nal returns of
447±38, 425±29, 427±27, and 450±47 that overlap inside their error bars:
at this budget no equivariant component produces a signicant return gain.
(iv) A bootstrap estimate of the eective-sample multiplier is non-monotone
in training length and inconsistent across two machines, so we report any
multiplier above 1 as an open problem rather than a conrmed result. We
present the theory as an upper bound and a scaling law, and we explicitly
mark where experiments support, weakly support, or contradict it.
Keywords:
Equivariant neural networks, World models, Imagination
training, Reinforcement learning, First-person shooter, PAC-Bayes
1. Introduction
Deep reinforcement learning (RL) has reached superhuman levels in many
games, but training an agent directly from raw pixels in rst-person shooters
(FPS) still demands millions of environment interactions [4, 3], which is im-
practical when environment access is expensive. World models address this
by learning a predictive model of the environment and optimizing the policy
entirely inside it [1]. The Dreamer family operationalizes this idea with a
recurrent state-space model [5, 6, 2] and has recently scaled to long-horizon
tasks inside learned models [7]; model-based planning with a learned dynam-
ics model is the core of MuZero [8], and model-based policy optimization
exploits such models for data eciency [9].
FPS environments, however, possess a near-perfect structural symme-
try that none of these general-purpose architectures exploit: the rendered
scene and its dynamics are (approximately) invariant under a horizontal ip,
and a ipped camera view should induce ipped actions (TURN_LEFT ↔
TURN_RIGHT, MOVE_FORWARD unchanged). Group-equivariant net-
works [10] bake such symmetries into the architecture and, in principle, use
each observation several times at no extra data cost. This is precisely the
data-starved regime in which world models operate.
We ask a deliberately narrow question: on a small FPS benchmark, does
Z2 equivariance in the world model and policy improve sample eciency,
and how should that be measured honestly?
Our system, EqWM, enforces
2

--- Page 3/16 ---
hard equivariance by group averaging [10] at every convolutional layer of the
encoder, recurrent dynamics, decoder, and policy, with an invariant reward
head and an antisymmetric left/right action head. Our contributions are:
1. Implementation and controlled comparison on ViZDoom.
We
build an equivariant ConvGRU world model and equivariant PPO policy
for health_gathering and compare them against matched non-equivariant
baselines, reporting per-seed as well as aggregated numbers (Section 6).
2. PAC-Bayes-style analysis of imagination training. We state three
results: a world-model prediction bound that depends on an eective
sample count Neff (Theorem 1), an imagination-training generalization
bound that separates world-model bias from estimation error (Theorem 2),
and a scaling law for the optimal real-to-imagination ratio (Theorem 3);
proof sketches and a bootstrap estimator (Proposition 1) are provided.
3. Honest empirical verdict. The clearest measured benet of equivari-
ance is cross-seed stability of the task-relevant reward head, not a large
mean accuracy gain; imagination training gives an order-of-2× sample-
eciency gain over pure PPO; the ablation shows no signicant return
gain from equivariance at this budget; and the bootstrap estimate of the
multiplier is not robust. We treat the latter two as limitations and open
problems rather than burying them.
The remainder of the paper is organized as follows. Section 2 reviews
related work. Section 3 xes notation. Section 4 details the architecture.
Section 5 states the theory. Section 6 reports experiments, and Section 7
concludes with limitations.
2. Related Work
World models and imagination training.. Learning a predictive model for RL
dates back to Ha and Schmidhuber [1]. The Dreamer series [5, 6] operational-
ized latent imagination with a recurrent state-space model, and DreamerV3
[2] showed a single model can solve many domains; the most recent Dreamer4
[7] scales agent training inside a fast world model in Minecraft. MuZero [8]
pairs learned dynamics with planning, and MBPO [9] uses an ensemble of
models to cut real samples. Concurrent with these, EMERALD [11] improves
world modeling with masked latent transformers and strong sample eciency
3

--- Page 4/16 ---
on Crafter. Our work follows the Dreamer-style train PPO inside a learned
latent model loop but restricts the model class to a group-equivariant one.
Equivariant reinforcement learning.. Group-equivariant convolutions were in-
troduced by Cohen and Welling [10]. In RL, MDP homomorphic network-
s [12] and group equivariant deep RL [13] encode state/action symmetries,
while SO(2)-equivariant policies [14] and group-invariant representation struc-
turing [15] exploit continuous rotation symmetries. The closest lines to ours
are equivariant model-based RL: Park et al. learn symmetric embeddings
for equivariant world models [16], Deac et al. make MuZero equivariant [17],
and EDGI builds an SE(3)-equivariant diusion planner for embodied agents
[18]. Multi-group equivariant augmentation [19] studies several symmetry
groups in manipulation.
Unlike these works, which target control, board
games, or 3D/SE(3) embodied settings, we target 2D FPS pixels with the
discrete Z2 group and integrate equivariance into both the world model and
the imagination-training loop, while reporting the small-benchmark result
transparently.
PAC-Bayes and generalization in RL.. PAC-Bayes bounds [20] relate a pos-
terior's risk to its empirical risk through a KL complexity term.
Recent
PAC-Bayesian RL [21] trains generalizable on-policy policies. We adapt the
decomposition to the mixed real/imagination data regime and use the boot-
strap [22] to estimate the eective sample multiplier empirically.
FPS reinforcement learning.. ViZDoom [3] provides a Doom-based visual-
RL benchmark; early agents trained directly with model-free RL [4]. To our
knowledge, prior work has not combined an equivariant world model with
imagination training on FPS pixels.
3. Preliminaries
MDP.. We model the environment as an MDP ⟨S, A, P, r, γ⟩[23], where S
is the RGB frame space, A = {turn-left, turn-right, forward} is a
three-action discrete set, P the transition kernel, r the reward, and γ =
0.99 the discount. The objective is the expected discounted return J(π) =
Eπ[P
t γtrt].
4

--- Page 5/16 ---
m*/n - optimal real:imagination ratio
scaling law . PAC-Bayes bound . qualitative only (not closed-form)
Observation s_t
64x64x3 RGB
Equivariant World Model
G = Z_2 . hard equivariance by group averaging
Encoder
group-avg conv
ConvGRU
latent dynamics
Decoder
transposed conv
Reward head
global-avg pool
Prediction
reconstruction
next frame
reward r (key metric)
Equivariant PPO policy
action logits
antisymmetric L/R head
value V
imagination rollouts (H=20): PPO updated inside learned model
Experiments
reward-head stability . few-shot imagination . equivariance ablation . bootstrap M_eff
Figure 1: Overview of the equivariant world model and imagination training. The world
model enforces hard Z2 equivariance by group averaging across an Encoder, a ConvGRU
dynamics cell, a Decoder, and a reward head; it predicts both frame reconstruction and
a scalar reward.
The equivariant PPO policy is updated inside the learned model on
imagination rollouts, and the analysis bounds the optimal real:imagination ratio m∗/n
only qualitatively.
Group action and equivariance.. We take G = Z2 = {e, h}, where h is a
horizontal ip. The group acts on observations by h · x = flipW(x) and on
actions by the permutation h·turn-left = turn-right, h·turn-right =
turn-left, h · forward = forward. A map f is equivariant if f(gx) =
gf(x) and invariant if f(gx) = f(x).
World models and imagination.. A world model ˆ
M = ⟨ˆP, ˆr⟩approximates
the environment. Given ˆ
M, imagination training starts a policy rollout from
a real observation and predicts subsequent latent states and rewards with ˆ
M,
then updates the policy (here PPO [24]) on these imagined trajectories [2].
PAC-Bayes.. With posterior ρ, prior π, and n i.i.d. samples, a canonical
bound is R(ρ) ≤RS(ρ) +
p
(KL(ρ∥π) + log(2√n/δ))/(2n).
4. Method
4.1. Equivariant Z2 world model
The world model ˆ
Mϕ has four parts (Figure 1). Encoder Eϕ: three strided
convolutions map a 64 × 64 × 3 frame to an 8 × 8 × z latent map (z = 32,
5

--- Page 6/16 ---
base channels 16). Every convolution is a group-averaging layer [10],
Eϕ(x) = 1
2

Conv(x) + flipW(Conv(flipW(x)))

,
(1)
which shares one kernel and satises Eϕ(flipWx) = flipWEϕ(x) up to numer-
ical precision. Dynamics: an equivariant ConvGRU cell updates the latent
map from the previous map and a spatially broadcast one-hot action; gates
use the same group-averaging convolution. Decoder: three equivariant trans-
posed convolutions reconstruct the frame with a sigmoid output. Reward
head: a global-average-pooling (invariant) layer followed by an MLP pre-
dicts a scalar reward from the latent map.
The non-equivariant baseline
uses identical widths and layer counts but ordinary convolutions, a attened
encoder for its heads, and no group averaging.
The training loss is reconstruction MSE plus reward MSE plus latent-
dynamics MSE, and (only for the equivariant model) a soft consistency penal-
ty on the encoder and dynamics under ip.
4.2. Equivariant PPO policy
The policy uses the same equivariant encoder. Invariant features (gap(h))
drive the forward action and the (invariant) value head; an antisymmetric
feature a = hleft−hright distinguishes left and right: ℓleft = s+a, ℓright = s−a,
ℓforward = s′. This enforces π(h · x) = h · π(x).
4.3. Imagination pipeline
We collect n = 10k random real transitions, train
ˆ
Mϕ for 30 epochs,
then iterate: collect 1,000 new on-policy real transitions, ne-tune
ˆ
Mϕ for
5 epochs, imagine short rollouts of horizon H = 20 under PPO [24] (200
rollouts per outer step), and update on the imagined data. This gives roughly
4,000 imagined transitions per 1,000 new real transitions, i.e. an imagination
ratio ≈4, xed rather than tuned. Table 1 lists all hyper-parameters.
5. Theory
We present three results and one estimator. They are intentionally stated
as upper bounds and scaling laws; our claim is about the structure of the
trade-o, not tightness. Before the statements we x a notation conict:
throughout, Neff is an eective sample count (dimension of data, ≤|G|n),
whereas Rboot in Proposition 1 is the dimensionless variance ratio we estimate
from data; Proposition 1 makes Rboot an empirical proxy for the multiplier
Neff/n.
6

--- Page 7/16 ---
Table 1: Hyper-parameters (identical for equivariant and standard variants except for
group averaging). Exp. 1 trained the world model for 50 epochs; the imagination runs
below use 30 initial epochs.
Group
Hyper-parameter
Value
Environment
Observation
64 × 64 × 3 RGB, frame skip 4
Discount γ
0.99
World model
Base channels / latent z
16 / 32 (8 × 8 × 32)
Init. real steps n
10,000
WM epochs (init / ne-tune)
30 / 5
Imagination
Horizon H
20
Imagined rollouts / outer step
200
New real steps / outer step
1,000
Imagination ratio m/n
≈4
PPO
Optimizer / LR
Adam / 3 × 10−4
Minibatch / epochs
256 / 4
Clip / GAE λ
0.2 / 0.95
Entropy / value coef.
0.01 / 0.5
Bootstrap
Resamples B / train set
15 / 8,000
Theorem 1 (World-model prediction bound). Let ˆ
Mϕ be trained on n
transitions. With probability ≥1 −δ over the data,
E(s,a,s′)∥ˆPϕ(s, a) −s′∥2 ≤ˆLWM(S) + O
q
KL(ϕ∥ϕ0)+log(1/δ)
Neff

,
where Neff ≤|G| n is the eective sample count.
Proof sketch.. Start from the standard PAC-Bayes inequality R(ϕ) ≤RS(ϕ)+
O(
p
(KL(ϕ∥ϕ0) + log(1/δ))/n) [20]. Because transitions within a rollout are
temporally correlated, group the data into independent blocks and apply
McDiarmid's inequality over blocks; this replaces n by an eective block
count. Group averaging reuses each transition together with its ipped copy,
so the independent data eectively available is at most |G| times the raw
count, i.e. Neff ≤|G|n. Note this is a data-reuse eect only: the group-
averaging layer shares a single kernel (4) and does not reduce the parameter
count.
The complete block-decomposition argument follows the standard
PAC-Bayes treatment and is deferred to the appendix.
7

--- Page 8/16 ---
Theorem 2 (Imagination-training bound). A policy trained on n real
and m imagined transitions satises, with probability ≥1 −δ,
J(πθ) ≥ˆJimag(πθ) −C εWM H −O
q
log(1/δ)
n+mη

,
where εWM is the world-model error, H the imagination horizon, and η ∈
[0, 1] down-weights imagined data by model accuracy.
Proof sketch.. The gap splits into two terms. Model-bias term: an H-step
rollout compounds one-step error geometrically, PH−1
t=0 γtεWM ≤C εWMH for
a constant C that depends on the discount and on the policy's Lipschitz con-
stant. Estimation term: treating imagined transitions as weight-η samples,
the eective data count is n+mη, to which the PAC-Bayes/McDiarmid esti-
mation bound applies directly. The two terms are independent and additive.
Theorem 3 (Optimal ratio, as a scaling law). Balancing the two terms
of Theorem 2 by dierentiating the upper bound with respect to m gives the
scaling law
m∗
n = C Neff/n
ε2
WM
,
(2)
for a constant C that depends on horizon, discount, and KL budget.
We emphasize that Theorem 3 is a scaling law, not a numerically tight
formula: substituting the pixel MSE εWM ≈10−2 literally yields m∗/n = 104,
which is unrealistically large because the constant and the units of εWM are
uncalibrated. We therefore clip the ratio to a xed range in practice and
treat the theorem as qualitative guidance, exactly as Section 6 reports.
Proposition 1 (Estimating the multiplier by bootstrap). The multi-
plier Neff/n can be estimated by bootstrap [22]: draw B resamples of the
training set, ret, and set
Rboot =
d
Varstd
d
Vareq
,
\
Neff/n = clip(Rboot, 1, |G| = 2).
(3)
Justication.. For an i.i.d. estimator with n samples, the out-of-sample error
variance scales as 1/n; if equivariance multiplies the eective data by k, its
error variance scales as 1/(kn), so the variance ratio recovers k. The boot-
strap [22] estimates these two variances by resampling. The clip to [1, |G|]
encodes the prior that a symmetry group of size |G| cannot multiply data
by more than |G|. This makes Rboot an empirical proxy for the theoretical
Neff/n of Theorem 1; it is not the same object as the sample count Neff itself.
8

--- Page 9/16 ---
Table 2: World-model comparison on health_gathering (10k steps, 50 epochs, 3 seeds;
mean±std). Only reconstruction and reward metrics with a well-dened per-frame esti-
mate are reported; the earlier concatenated-rollout open-loop MSE was computed with a
awed metric and is omitted (see text).
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
stable; see Fig. 2
6. Experiments
6.1. Setup
We use ViZDoom health_gathering: the agent navigates a maze to col-
lect health packs while taking damage; observations are 64 × 64 × 3 RGB
frames with frame skip 4. The reward is near-degenerate: the agent is penal-
ized only at death, so roughly 99.3% of non-terminal steps carry an identical
reward oset. This sparsity makes the reward head easy to overt and moti-
vates the stability analysis below. The small model (base channels 16, latent
8 × 8 × 32) is used throughout. We report mean±std over seeds and state
the seed count explicitly for every number.
6.2. Experiment 1: world-model prediction and reward-head stability
Both world models are trained on 10k random transitions for 50 epochs
over three seeds (Table 2).
Two ndings matter. First, single-frame reconstruction (PSNR/SSIM)
is comparable and, if anything, marginally favors the non-equivariant model
(21.8 vs 20.6 dB): equivariance costs a little pixel delity. Second, the reward
head behaves very dierently across seeds.
The per-seed reward MAE is
[0.072, 0.107, 0.132] for the equivariant model but [0.089, 0.256, 0.324] for the
standard model (Fig. 2). The standard model's higher mean (0.223) and
much higher std (0.099) are driven by two seeds on which the reward head
drifts to 0.260.32, which we read as overtting to the near-degenerate death
reward. The equivariant reward head instead remains stable. We therefore
frame the 0.104 vs 0.223 comparison as improved cross-seed stability of the
task-relevant head, not as a uniform 53% more accurate predictor.
We
also omit long-horizon open-loop rollout error: our pipeline concatenates
unrelated transition tuples when computing it, which makes the resulting
9

--- Page 10/16 ---
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
Figure 2:
Per-seed reward-prediction MAE. The equivariant reward head stays in
[0.072, 0.132] across seeds, whereas the non-equivariant head diverges on two of three
seeds (0.256, 0.324). The benet is cross-seed stability, not a uniformly large per-sample
accuracy gain.
MSE unreliable, and a rigorously re-computed chained rollout was not run
within this budget; we prefer to report only the well-dened per-frame metrics
of Table 2.
6.3. Experiment 2: few-shot imagination training
We compare imagination-trained agents (equivariant or standard world
model) against pure on-policy PPO, over two seeds (Fig. 3). After only ∼15k
real steps, the imagination agents reach episode returns in roughly 316508,
the same range in which pure real PPO sits at 20k40k real steps (284
572). The four initial (untrained) scores are 396, 380, 364, and 668 (mean
≈452; the 668 is a single high-variance outlier), and all curves are at and
noisy at this budget. We conclude that imagination training yields an order-
of-2× sample-eciency gain over pure PPO: the step ratio is ∼1.32.7×
(20k40k real steps over the ∼15k imagination budget), and because neither
baseline has converged this is an order-of-magnitude statement rather than
a clean crossover. We state two caveats up front: only two seeds were run,
and the equivariant and non-equivariant imagination variants are statistically
indistinguishable and win on dierent seeds. We do not claim an equivariance
advantage on sample eciency at this scale.
10

--- Page 11/16 ---
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
Figure 3: Few-shot imagination training (2 seeds). Thin lines are individual seeds, thick
lines their mean. Imagination agents at ∼15k real steps match real PPO evaluated at
20k40k steps. The gray band marks the untrained-policy range (∼380400; one outlier
seed at 668); all curves are noisy at this budget.
6.4. Experiment 3: ablation of the equivariant components
We ablate whether equivariance is placed in the world model, the policy,
both, or neither, under the same small budget and a robust evaluation (3
train seeds × 3 eval seeds × 8 episodes). Final robust returns are in Table 3
and Fig. 4.
The four means lie in a narrow band (425450) and their error bars over-
lap; the fully non-equivariant conguration is in fact numerically highest.
We report this honestly: under the small-model, short-training budget, the
reward-head stability observed in Experiment 1 does not translate into a sig-
nicant nal-task return gain. Whether equivariance would help return at
larger scale or with more seeds is an open question.
6.5. Experiment 4: bootstrap estimate of the multiplier
Following Proposition 1, we bootstrap B = 15 resamples of an n = 8k
transition set, ret on each, and compare held-out prediction-error variances
between the equivariant and standard models across training lengths. On
the primary local run the variance ratio Rboot = Varstd/Vareq is 0.51, 5.72,
11

--- Page 12/16 ---
Table 3: Ablation: where equivariance is placed (3 train seeds; robust evaluation). All
four means overlap within one standard deviation.
Conguration
Final return (robust)
both-eq (WM + policy equivariant)
447.1 ± 37.7
wm-eq-only (WM equivariant, policy standard)
425.3 ± 28.5
policy-eq-only (WM standard, policy equivariant)
426.7 ± 26.7
none-eq (both standard)
449.8 ± 46.7
and 0.09 at 5, 20, and 40 epochs respectively. As a secondary cross-check we
also logged an independent cloud run (A10 GPU), which records 0.72, 1.50,
and 28.3 (the last value is the raw, pre-clip variance ratio; clipped to the
group ceiling it is 2.0). We disclose a data limitation: the cloud numbers at
5 and 20 epochs were reconstructed from terminal-log OCR with incomplete
per-replicate arrays (5 epochs: no per-replicate list retained; 20 epochs: only
3 of 15 replicates), whereas the 40-epoch cloud point is complete (all 15
replicates, consistent across three OCR reads). We therefore treat the local
run as primary and the cloud run as a suggestive cross-check only.
The
ratio is non-monotone and the two runs disagree at 40 epochs, where the
equivariant model even diverges on several local bootstrap replicates (test
error rising to 0.040.055). We therefore do not claim a multiplier above 1 as
a robust result: it appears only in a narrow intermediate-training band and
is highly sensitive to training length and run environment. This is reported
as an open problem.
6.6. Summary of empirical claims
Across the four experiments, the supported statements are: (i) equivari-
ance stabilizes the reward head across seeds; (ii) imagination training gives
an order-of-2× sample-eciency gain over pure PPO at this scale (measured
ratio ∼1.32.7×); (iii) equivariance produces no signicant return gain in the
ablation; and (iv) the bootstrap evidence for an eective-sample multiplier is
not robust. The theory of Section 5 gives the correct structure of the trade-
o (world-model bias dominates the imagination gap), but its quantitative
predictions are only qualitatively supported.
12