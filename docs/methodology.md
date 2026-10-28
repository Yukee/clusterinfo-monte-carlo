# Model specification and interpretation

This implements the proposal as a two-level, paired-potential-outcome model.
It provides a way to explore assumptions; the numerical defaults do not settle
the proposal's empirical questions. See [provenance](provenance.md).

## 1. Outer draws: population uncertainty

For each outer run:

1. Jointly bootstrap the observed help-seeking and actionable-learning
   quantities. Fix `a2` to the selected sensitivity value.
2. Draw the action probability conditional on intent and the correction from
   monthly browser/person IDs to visitors. These two triangular priors are
   assumed independent of each other and the clinical parameters.
3. Draw the joint diagnosis/access population distribution and the uncertain
   clinical transition, efficacy-gain and headstart parameters once. Reuse
   these population values for all patients in this run.

The clinical parameter model uses a shared latent normal factor so that
access/improvement probabilities and headstart/effect magnitudes need not
vary independently. This is an explicit dependence assumption, not a fitted
correlation matrix. A separate patient-level shared factor induces dependence
among individual opportunities, barriers, gains and headstarts. Higher
barriers lower access/optimization probabilities and gains, while increasing
delays and headstarts. `population_dependence` is the correlation between
the latent normal draws, not necessarily between the transformed outcomes.
The initial-state Dirichlet is independent of this factor.

`clinical_uncertainty_logit_sd` defaults to zero, preserving the reference
episodic proportion, acute-success center and preventive-response probability.
If increased, it applies independent logit-normal uncertainty to these three
biological parameters; they do not share the access-barrier factor.

Selected illustrative assumptions (full settings in `examples/default.json`):

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `visit_during_bout_probability` | .8 | Episodic visitor is in an active bout |
| `natural_diagnosis_probability` | .6 | Later diagnosis opportunity |
| `natural_acute_probability` / `natural_preventive_probability` | .65 / .6 | Ordinary access opportunity once diagnosed |
| `natural_diagnosis_delay_days` / `natural_access_delay_days` | 365 / 90 | Median baseline diagnosis delay / access lag |
| `ci_diagnosis_probability` | .75 | Candidate diagnosis opportunity |
| `ci_acute_access_probability` / `ci_preventive_access_probability` | .55 / .5 | Candidate direct acquisition opportunity |
| `ci_optimization_probability` | .65 | Each treatment class has a nonzero-gain opportunity |
| `uptake_delay_days` / `acquisition_delay_days` | 14 / 30 | Median uptake delay / acquisition lag |
| `headstart_days_by_stratum` | [180, 90, 45] | Median headstart for diagnosis / access / optimization pathways |

The probability settings are centers on the logit scale before shared outer
and individual barrier shifts, rather than final marginal rates. Timing
settings are lognormal medians before those shifts. Boundary probabilities
0/1 remain fixed, which also enables deterministic controls.

### Joint visitor bootstrap

The target cohort contains 1,591 IDs. Categories within each engagement score
are ordered Yes / No / unanswered:

| E | Yes | No | Unanswered |
| --- | ---: | ---: | ---: |
| 0 | 29 | 47 | 607 |
| 1 | 59 | 96 | 256 |
| 2 | 71 | 81 | 151 |
| 3 | 58 | 58 | 78 |

Resample the target cohort by a multinomial draw using sequential conditional
binomials. If `K` is its resampled E=3/Yes count, draw the number of overlapping
actionable-learning respondents as `Binomial(K, 8/58)`. Separately resample
the 12 outside-cohort respondents: `outside_yes ~ Binomial(12, 11/12)`.
Then:

```text
a3 = (overlap + outside_yes) / (overlap + 12)
s_e = y_e / (y_e + z_e)
h_e = n_e * s_e
w_e = h_e / sum(h_e)
P(help-seeking) = sum(h_e) / sum(n_e)
P(action intent | help-seeking) = a2*w_2 + a3*w_3
```

The eight shared respondents have the same sampled multiplicities in both
estimators. Help-seeking and intent are not independent marginal draws. The
standalone bootstrap uses seeds 20261020 / 20261021 and 30,000 resamples;
percentiles interpolate at `(B-1)*p`. Undefined denominators raise an error
rather than silently rejecting or repairing a sample.

Identification still assumes response independence conditional on engagement,
truthful responses, and transfer of the broader-window `a3` estimate. `a0=a1=0`
is held fixed. Actionable learning is treated as a proxy for action intent;
the subsequent actual-action gate remains an assumption. `a2` endpoints form
a sensitivity range, not one identified probability's confidence interval.

## 2. Visitors, traits and initial state

Each inner visitor passes the help-seeking, conditional intent, and conditional
action gates in sequence. A help-seeker is modeled as one CH patient (a
caregiver acts as a proxy for that patient). Others contribute zero to modeled
CH burden and benefit. Funnel status is currently independent of clinical
stratum, subtype and severity. This selection assumption needs calibration;
ID correction alone cannot resolve several people helping the same patient.

Initial diagnosis and treatment access are drawn from eight **joint** cells:

| Diagnosis | Neither | Acute only | Preventive only | Both |
| --- | ---: | ---: | ---: | ---: |
| Undiagnosed | .43 | .03 | .02 | .02 |
| Diagnosed | .22 | .09 | .15 | .04 |

The means imply 50% diagnosed, 18% acute access, 23% preventive access.
Only the last two marginals come from the reference model; the joint cells
are illustrative. An outer Dirichlet draw varies the joint distribution;
zero concentration disables this uncertainty. Access without a recorded
diagnosis is allowed. The 18%/23% values are never used as probabilities
conditional on diagnosis.

Fixed patient traits follow the pinned model: episodic probability .85;
Gamma-distributed bouts/year (mean 1.2, SD 1.1), weeks/bout (8.5, 5.7),
and attacks/active day (episodic 2, 1.5; chronic 2.5, 1.8), with attack rates
clipped to [.5, 8]. Patient severity is Normal(7, 1.2), clipped to [1,10].
Attack severity adds Normal(0,.9), rounds ties to even and clips to [1,10].
Intrinsic duration is LogNormal(log(45), .7) times the severity factor
`clip(1 + .08*(peak-severity_mean), .3, 2.5)`, then clipped to [5,360]
minutes. With default `severity_mean=7`, this matches the reference formula.

## 3. Time and paired treatment histories

The time origin is the website visit. Both arms use the same horizon and
underlying patient, attack times, severity, duration, response status and
treatment random variates. CI changes events in the treatment history; it
does not independently redraw the patient or their attacks.

### Bouts and attacks

For episodic patients, a continuous bouts/year draw defines a recurring
period; active length is the drawn weeks/bout, capped to 273/365 of the period
(the reference model's 39 weeks per 365-day year). Visit phase is
drawn within the active interval or remission according to a configurable
probability. Poisson attack opportunities occur only in active intervals.
This accounts for delayed uptake during or after a bout directly.

Chronic patients have Poisson attacks at their rate times the reference .9
active-time fraction. No separate chronic bout schedule is assumed.
The event-based construction replaces the reference model's rounded/capped
annual attack counts. Prevention uses shared-uniform thinning rather than
removing an exactly rounded fixed number of attacks. These changes enable
time-varying care and are not bit-for-bit reproduction of the annual model.
As in the reference model, attack burdens are summed. Poisson attack onsets
can overlap; this approximation does not resolve overlapping pain or impose
a calendar-time upper bound on summed DLES. Events are evaluated using care
available at attack onset, without a within-attack treatment-transition model.
For an attack crossing the horizon, the duration is clipped first and then
the compressed reference profile is scored. This boundary approximation
preserves monotone paired effects; it does not integrate the exact prefix
of the uncompressed original profile.

### Diagnosis, access, optimization and catch-up

Natural diagnosis and acute/preventive access may occur later without CI,
or not occur during the horizon. Natural treatment uses the baseline
efficacy distributions. For actors, candidate CI diagnosis, access and
efficacy-improvement events occur after uptake/acquisition delays, subject
to opportunity/success draws. Existing access can be optimized without
having to reacquire it.

A diagnosis gained earlier also advances ordinary access using the same
access chance and lag in each arm, even if direct CI acquisition fails.
A successful candidate intervention also has a no-CI catch-up event at
its candidate time plus a shared headstart. Earlier natural diagnosis or
access takes precedence. Thus treatment access already acquired elsewhere
does not count again as new CI access. Improvement of existing treatment
can still create a benefit until the no-CI efficacy catch-up event.

**Catch-up is a structural assumption:** conditional on the latent action and
opportunity draws, no-CI eventually gets the same successful intervention.
The model does not currently assign a separate probability of permanent
CI-only access. Potential catch-up events are defined before applying the
actual-action gate or comparing the arms; the no-CI control preserves them.
This keeps the patient's no-CI history fixed when uptake assumptions change.
This construction makes the
assumed headstart explicit but does not empirically estimate it.

Headstart distributions overlap and depend on the candidate pathway in the
patient's history: earlier diagnosis first, otherwise earlier access,
otherwise optimization. Candidate events are compared with ordinary natural
care before headstart is sampled, avoiding circular dependence. Default
medians are 180 / 90 / 45 days respectively. An undiagnosed patient with
existing treatment whose diagnosis is not advanced therefore uses the
optimization prior. The shared barrier factor links these times with access
and efficacy gains. Alternative orderings are explicit sensitivity scenarios.
`initial_stratum` records starting status; `headstart_stratum` separately
records the chosen pathway, including latent opportunities for nonactors.

Uptake time, sampled headstart and **realized** acceleration differ: natural
care can precede a candidate CI event, no treatment benefit may occur, and
no active bout may overlap the time advantage. A long sampled headstart
alone does not guarantee benefit.

### Baseline treatment and improvements

Acute access gives an 85% chance of treating an attack. Patient abortion
success is Normal(.64,.22), clipped to [0,.95]. On success, duration becomes
the minimum of the original duration and Normal(15,6) bounded below at 3
minutes; peak falls 10%, rounded to an integer.

Preventive users respond with probability .42. Responders' frequency
reduction is Normal(.55,.20), clipped to [.50,.95]; the distribution's
post-clipping mean differs from .55. Remaining attacks get a 10% rounded
peak reduction. Nonusers and nonresponders get no preventive effect.

CI may change treated-attack coverage, success probability, aborted duration,
preventive response/frequency reduction and peak reductions. Gains include
zero-improvement draws and nonresponders. Shared uniforms couple baseline
and improved response/attack outcomes. Preventive peak reduction occurs
before acute reduction, rounding each step, and intrinsic duration is drawn
from untreated severity. Both arms obtain identical incremental effects
once their respective improvement event has occurred.

Gain parameters have distinct meanings. Treated-attack coverage and preventive
response gains are fractions of remaining headroom to 1; acute success gains
use headroom to .95; preventive frequency gains use headroom to the configured
reduction cap. Duration gain is a proportional shortening, with the duration
floor retained. Peak gain is an additive reduction fraction, capped at 1.
For example, acute success .64 with gain .15 becomes `.64+.15*(.95-.64)`,
not .79. Population gain centers are logit-normal; individual gain fractions
are that draw times Uniform(0,2), divided by the patient's barrier factor,
then clipped to [0,1]. Optimization failure supplies a point mass at zero.

The model permits only nonnegative treatment improvements. It excludes
adverse effects, substitution away from effective care, treatment burden,
mortality and spontaneous long-term disease evolution. Consequently its
benefit distribution cannot quantify net harm from those mechanisms.

## 4. Extreme pain and outputs

For integer peak `p >= 9` and duration `d`, the reference extreme-pain proxy is:

```text
r = min(d, 29 minutes)
extreme_minutes = (d - r) + r * (p - 8) / p
DLES = sum(extreme_minutes) / 1440
```

It is zero for `p < 9`. The 29 minutes combine the reference attack's ramp
and decay time. This is the reference model's **integer NRS extreme-pain
bucket proxy**: its ramps use cutoff 8. It is not literally time above a
continuous NRS threshold of 9, nor all minutes of an attack peaking at 9.
Other pain buckets and their welfare weights are intentionally omitted.

For each visitor `i`, `delta_i = no_CI_DLES_i - CI_DLES_i`. For each outer draw:

```text
AP = mean(delta_i)                         # DLES per corrected visitor
AM = monthly_unique_IDs * ID_correction * AP
hours = DLES * 24
```

Nonresponse, no action, no improvement and natural catch-up are already in
`delta_i`. Do not multiply by another “helped” fraction, counterfactual share,
headstart length, or episodic correction. Headstart and active-bout overlap
are already integrated in the trajectories.

AM is the cumulative benefit of a **monthly arrival cohort** over the chosen
horizon, not benefit realized during a calendar month. The model counts a
cohort once; it does not identify repeat visitors across different months.
Finite horizons censor benefits after their end. Inspect longer horizons
as a separate sensitivity analysis rather than silently extrapolating.

Within-run standard errors measure variation of the paired `delta_i` mean.
Outer percentiles include assumed parameter uncertainty, bootstrap sampling
uncertainty, and residual inner Monte Carlo error. Small runs can be dominated
by rare actors/high-burden patients. Increase inner and outer sizes and compare
independent seeds before treating the empirical percentiles as stable.
These are neither identified confidence intervals nor calibrated posteriors.

## 5. Randomness and sensitivity checks

Stable SHA-256-derived substreams isolate the cohort bootstrap, overlap,
outer priors, clinical population, visitor gates and each patient's clinical
history. Scenario comparisons use common random numbers. Exact reproduction
requires the same code, configuration and Python random implementation;
the outputs record those inputs and a source fingerprint.

Required sensitivity scenarios are no **additional CI** peak reduction and
zero acute/preventive peak reductions in **both** arms. Reversed and equal
headstart orderings test the proposed ordering. `no_ci` and `zero_headstart`
are exact-zero controls. The `a2=0/1` comparisons retain the same bootstrap
resamples; neither is claimed to be a guaranteed lower/upper bound on true
net impact under unmodeled bias or harms.
