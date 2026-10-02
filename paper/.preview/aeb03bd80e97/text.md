--- Page 1/1 ---
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