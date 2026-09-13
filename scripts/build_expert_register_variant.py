"""Build T5_expert_register: a tutor whose delivery is mis-pitched and overloaded. NO MODEL CALLS.

Spec and pre-registered predictions:
``docs/evaluation_completeness_v2/28_EXPERT_REGISTER_PREREGISTRATION.md`` -- frozen before this
text was authored.

WHY THIS CONDITION EXISTS
-------------------------
``clarity_cognitive_load`` and ``student_level_calibration`` have never been tested. The
pedagogical-degradation condition strips diagnosis and scaffolding; it does not introduce
unexplained terminology or mis-pitch the student's level, so those two dimensions have nothing to
detect in it. Reporting them as failed would repeat the SP-2 error this project already
documented.

THE TRANSFORMATION RULE, APPLIED TO ALL 63 TURNS
------------------------------------------------
1. formal register: technical vocabulary the dialogue never introduces and the exercise sheets do
   not use;
2. gratuitous forward reference: each turn cites material the student has not met and does not
   need for the question asked;
3. no signposting: enumerations and staged build-ups collapse into one dense block.

WHAT IS PRESERVED, AND WHY EACH MATTERS
----------------------------------------
Diagnosis and scaffolding stay (else T5 is a second T2_answer_dumping). Every factual claim stays
correct (else it is T3_subtly_wrong). Self-consistency, dependency order and resolution stay (else
it is T4_macro_degraded). What changes is only how the content is delivered.

The student asks for an analogy at idx 8, 15, 20, 41 and 52. Each of those turns still SUPPLIES an
analogy -- drawn from a domain a beginner will not follow. Removing it would be a scaffolding
failure; supplying an inaccessible one is a calibration failure, which is the distinction this
whole condition rests on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data/input/dialogues/inbox/dm1_human_student_llm_tutor_teaching_rich_dialogue.json"
OUT_DIR = REPO_ROOT / "data/input/dialogues/inbox"
GROUND_TRUTH = REPO_ROOT / "data/gold/expert_register_variant_ground_truth.json"

# Turns where the student explicitly requests an analogy. Each must still contain one.
ANALOGY_TURNS = {8, 15, 20, 41, 52}

T5_TEXT: dict[int, str] = {
    0: "Understood. I will operate as a corrective-feedback channel over your hypothesis space, "
       "issuing diagnostic interventions where your stated posterior diverges from the "
       "curricular prior, interleaving worked derivations with elicitation prompts, and "
       "maintaining anaphoric coreference across the discourse. Content remains grounded in the "
       "uploaded exercise corpus, though you should be aware that most of what follows "
       "generalises to structured-prediction settings and to the PAC-learnability results we "
       "will not cover.",
    1: "Your characterisation is under-constrained. Classification is the induction of a "
       "hypothesis h: X to Y over a discrete codomain under empirical risk minimisation, where "
       "supervision denotes access to a labelled sample drawn i.i.d. from the joint distribution. "
       "Partition induction absent label supervision is a distinct objective, addressed under "
       "Exercise 6, and relates to the density-based and spectral formulations you will meet "
       "elsewhere.",
    2: "Affirmative. Defaulted Borrower constitutes the response variable with codomain {Y, N}; "
       "Home Owner, Marital Status, Level of Education and Annual Income constitute the "
       "covariate vector. This is the standard design-matrix formulation, and the same "
       "decomposition underlies generalised linear models and the feature-map view used in "
       "kernel methods.",
    3: "Your terminology conflates the operational split with the conceptual one. The canonical "
       "decomposition is induction and deduction: the learning phase performs inductive "
       "hypothesis construction over the labelled sample; the classification phase performs "
       "deductive application of the induced hypothesis to an unseen instance. Testing is an "
       "instantiation of the latter used for generalisation estimation, which is itself a "
       "question of bias-variance decomposition.",
    4: "Not equivalent. The learning set is the full labelled sample available for both induction "
       "and generalisation estimation; it admits a partition into training and test subsets so "
       "that the estimate of generalisation error is computed on a sample disjoint from the one "
       "used for hypothesis selection. Failure to enforce disjointness yields an optimistically "
       "biased estimator, which is the same pathology that motivates nested cross-validation "
       "under hyperparameter search.",
    5: "Correct. Home Owner is a binary nominal covariate; Marital Status is nominal with "
       "cardinality three and no induced total order; Level of Education is ordinal, admitting a "
       "total order without a well-defined metric; Annual Income is ratio-scaled; Defaulted "
       "Borrower is the nominal response. The measurement-scale taxonomy here is Stevens's, and "
       "it constrains which similarity functions are admissible downstream.",
    6: "The orthography is not the discriminating property; the induced order relation is. Low, "
       "Moderate and High admit a total order, hence ordinal. Single, Married and Divorced admit "
       "no such relation, hence nominal. Note that ordinality without an interval structure "
       "means differences are not defined, which is why monotone transformations are permissible "
       "but arithmetic means are not.",
    7: "Negative. You have committed a base-rate fallacy by conflating the likelihood P(+ | S) "
       "with the posterior P(S | +). Under a prevalence of 0.01, the healthy stratum dominates "
       "the sample space, so even a false-positive rate of 0.01 generates a substantial absolute "
       "count of positives from that stratum. This is the same phenomenon that governs "
       "precision degradation under class imbalance.",
    8: "Consider a matched-filter detector operating at a signal-to-noise ratio where the "
       "prior probability of signal presence is 0.01. The detector's true-positive rate is 0.99, "
       "but the noise-only ensemble is ninety-nine times larger, so the expected count of "
       "noise-triggered detections approximately equals the expected count of genuine ones. The "
       "receiver-operating characteristic tells you nothing about which hypothesis to favour "
       "post-detection without the prior, which is why the posterior evaluates to 0.5.",
    9: "Let S denote the disease state and + the observed test outcome. By Bayes' theorem with "
       "marginalisation over the partition {S, not-S}: P(S | +) = P(+ | S)P(S) / [P(+ | S)P(S) + "
       "P(+ | not-S)P(not-S)] = (0.99)(0.01) / [(0.99)(0.01) + (0.01)(0.99)] = 0.0099/0.0198 = "
       "0.5. The denominator is the evidence term, and in the general case it is the "
       "normalising constant that makes variational approximation necessary.",
    10: "Naive Bayes performs maximum a posteriori classification under a factorised likelihood: "
        "the discriminant for class c is P(c) times the product over i of P(x_i | c), and the "
        "decision rule is the argmax over c of that quantity. In log-space this is an additive "
        "discriminant, which is what makes it a linear classifier in the space of log-"
        "probabilities, a fact you will need if you meet logistic regression later.",
    11: "The nomenclature refers to the conditional-independence assumption, not to performance. "
        "The model posits that the covariates are mutually independent conditional on the class "
        "variable, which factorises a joint distribution of exponential size into a product of "
        "marginals. The assumption is generally violated, yet the induced decision boundary is "
        "frequently robust to that violation, a result attributable to the argmax being "
        "insensitive to calibration error in the estimated posterior.",
    12: "Estimate the class marginals by maximum likelihood over the empirical distribution. The "
        "sample contains three realisations of DB=Y and seven of DB=N over ten observations, "
        "hence P(DB=Y)=3/10 and P(DB=N)=7/10. These are the prior marginals in the sense that "
        "they are estimable without conditioning on the covariate vector of the query instance; "
        "under a Dirichlet prior they would instead be posterior-mean estimates.",
    13: "Your numerator is correct; your denominator is not. Conditioning induces a restriction of "
        "the sample space to the conditioning event before normalisation. The event DB=Y has "
        "cardinality three, and all three realisations satisfy Home Owner=N, so "
        "P(Home Owner=N | DB=Y)=3/3, not 3/10. Normalising over the unrestricted space would "
        "yield the joint rather than the conditional.",
    14: "Correct. The event (Married, DB=Y) has empty support, so its maximum-likelihood estimate "
        "is identically zero; under a discrete treatment of income, (Income=90, DB=N) likewise "
        "has empty support. Because the discriminant is a product over covariates, a single "
        "zero-valued factor annihilates the entire class discriminant. This is the sparsity "
        "pathology that motivates additive smoothing generally.",
    15: "Consider a product-form utility over a compiler's optimisation passes, where the overall "
        "objective is the product of per-pass gain factors. A pass with a measured gain factor of "
        "exactly zero drives the product to zero regardless of the multiplicative contribution of "
        "every other pass, because the product operator has zero as an absorbing element. The "
        "maximum-likelihood estimate of an unobserved attribute-value pair has the same absorbing "
        "behaviour in the Naive Bayes discriminant.",
    16: "Only under simultaneous denominator correction. The Laplace estimator for a categorical "
        "covariate is (count + 1)/(class_count + |V|), where |V| denotes the cardinality of the "
        "value domain, so that the smoothed estimates remain a normalised probability mass "
        "function. Incrementing the numerator alone violates normalisation and is the common "
        "error; the general form is the symmetric Dirichlet prior with concentration parameter "
        "one.",
    17: "Affirmative, under a discrete treatment of the income covariate for this instantiation. "
        "The DB=N discriminant evaluates to 7/10 * 5/9 * 5/10 * 1/17, approximately 0.0114; the "
        "DB=Y discriminant evaluates to 3/10 * 4/5 * 1/6 * 2/13, approximately 0.00615. The "
        "argmax selects DB=N. Note these are unnormalised posteriors, so the ratio rather than "
        "the magnitude is what carries decision-theoretic content.",
    18: "There is no inconsistency once the parametric assumption is made explicit. A continuous "
        "covariate admits either a discretisation into ordered bins or a class-conditional "
        "parametric density, canonically Gaussian, whose class-specific mean and variance are "
        "estimated by maximum likelihood and whose density value substitutes for the conditional "
        "mass. This instantiation adopts discretisation because the exercise specifies "
        "attribute-value conditionals under additive smoothing.",
    19: "That is precisely the pathology. Information Gain is biased toward high-cardinality "
        "partitions: the ID covariate induces a partition into singleton cells, each of which is "
        "class-pure, so the conditional entropy of the response given the partition vanishes and "
        "the mutual information attains its maximum. The induced hypothesis has no generalisation "
        "capacity, which is a variance-dominated regime in bias-variance terms.",
    20: "Consider a hash function evaluated as a compression scheme. A function that maps every "
        "input to a distinct bucket achieves perfect discrimination on the observed key set and "
        "zero residual collision entropy, yet it encodes only the identity of the observed keys "
        "and transfers no structure to unseen keys. A partition induced by a primary key has "
        "exactly this property, which is why an Information Gain of one from ID is diagnostic of "
        "memorisation rather than of structure.",
    21: "Gain Ratio applies a normalisation by the split information, which is the entropy of the "
        "partition-induced distribution over child cells. A partition into many low-cardinality "
        "cells has high split information, so the normalised statistic is attenuated. "
        "Consequently the ID covariate can attain a mutual information of one while its Gain "
        "Ratio is approximately 0.231. The construction is a crude penalisation term, related in "
        "spirit to the description-length criteria used elsewhere.",
    22: "Entropy is the expected self-information of the class variable under the empirical "
        "distribution, measured in bits. A class-pure node has entropy zero; a binary node with "
        "equiprobable classes, for instance ten positives and ten negatives, has entropy one. "
        "Information Gain is the mutual information between the covariate and the response, "
        "computed as parent entropy minus the size-weighted expectation of child entropies.",
    23: "Because the child entropies must be aggregated under the induced measure rather than "
        "under the counting measure. A cell containing ten observations carries ten times the "
        "probability mass of a cell containing one, so each child entropy enters weighted by the "
        "ratio of child cardinality to parent cardinality. Unweighted averaging would compute an "
        "expectation under the wrong measure and would misrank partitions.",
    24: "Affirmative. The Graduation covariate induces cells with non-degenerate class structure: "
        "Middle School is class-pure, High School is class-dominant, and College is mixed. Its "
        "Gain Ratio is approximately 0.436 against approximately 0.231 for ID, so the normalised "
        "criterion reverses the ranking that unnormalised mutual information would produce. This "
        "is regularisation by criterion design rather than by explicit penalty.",
    25: "The three estimators differ in the evidence set over which the risk functional is "
        "computed. The optimistic estimator evaluates resubstitution error on the induction "
        "sample; the pessimistic estimator augments resubstitution error with a complexity term "
        "linear in leaf cardinality; the reduced-error criterion evaluates on a held-out "
        "validation partition. The first is a biased estimator of generalisation risk, the second "
        "a penalised one, the third an unbiased one under the usual assumptions.",
    26: "k denotes leaf cardinality. The exercise specifies the complexity functional over "
        "terminal nodes, so the penalised risk is resubstitution error plus Omega * k / Ntrain. "
        "Internal-node or total-node cardinality would define a different functional and is not "
        "the specified one; the choice of complexity measure is itself a modelling decision, as "
        "in the structural-risk-minimisation framework.",
    27: "Correct, and the trade-off is the point. A higher-capacity hypothesis attains lower "
        "resubstitution error while incurring a larger complexity penalty, so the penalised "
        "objective can be minimised by the lower-capacity model. This is the discrete analogue of "
        "the regularisation path in penalised estimation, and the Omega parameter plays the role "
        "of the regularisation coefficient.",
    28: "Reduced-error pruning evaluates, for a candidate internal node, the substitution of its "
        "induced subtree by a terminal node labelled with the majority class under the local "
        "empirical distribution. The substitution is accepted when validation risk does not "
        "increase, with ties resolved toward the lower-capacity hypothesis. The procedure is "
        "greedy and bottom-up, hence not guaranteed to reach the global optimum of the penalised "
        "objective.",
    29: "Then the local empirical distribution is equiprobable and the majority-class label is "
        "not identified, so the procedure requires an explicit tie-breaking rule. The rule must "
        "be declared rather than left implicit, since an undeclared convention makes the induced "
        "hypothesis non-reproducible; a consistent bias toward one class is the usual choice, and "
        "the asymmetric-cost formulation would resolve it by expected loss instead.",
    30: "Because the confusion-matrix cells are defined relative to a designated positive class. "
        "Under Y positive, TP counts (actual Y, predicted Y), FN counts (actual Y, predicted N), "
        "FP counts (actual N, predicted Y), and TN counts (actual N, predicted N). Designating N "
        "positive transposes the matrix and consequently transforms every derived functional. "
        "Only the accuracy functional is invariant under that transposition.",
    31: "From the contingency table: TP=2, since two actual-Y instances receive predicted label Y; "
        "FN=1, since one actual-Y instance receives predicted label N; FP=2, since two actual-N "
        "instances receive predicted label Y; TN=5, since five actual-N instances receive "
        "predicted label N. The margins of this table are what the chi-squared independence test "
        "would condition on.",
    32: "The two functionals differ in their normalising margin. Precision conditions on the "
        "predicted-positive margin: TP/(TP+FP) = 2/(2+2) = 0.5. Recall conditions on the "
        "actual-positive margin: TP/(TP+FN) = 2/(2+1) = 0.667. The numerator is common; the "
        "conditioning event differs. In the decision-theoretic view they correspond to different "
        "asymmetric loss functions.",
    33: "Sensitivity is definitionally identical to recall, TP/(TP+FN). Specificity is the "
        "corresponding functional on the actual-negative margin, TN/(TN+FP). Numerically "
        "sensitivity is 2/3, approximately 0.667, and specificity is 5/7, approximately 0.714. "
        "The pair (1 - specificity, sensitivity) is the coordinate that traces the receiver-"
        "operating characteristic as the decision threshold varies.",
    34: "F1 is the harmonic mean of precision and recall, 2PR/(P+R), evaluating to approximately "
        "0.571 at P=0.5 and R=0.667. It is not derivable from accuracy, which is the trace-"
        "normalised functional (TP+TN)/N = 7/10 = 0.7 and conditions on no margin at all. The "
        "harmonic mean is chosen over the arithmetic mean because it is dominated by the smaller "
        "argument, which is the F-beta family at beta equal to one.",
    35: "The margin of error is Z * sqrt(p(1-p)/N), so N enters through the standard error of the "
        "binomial proportion estimator, which decays as N to the power minus one half. "
        "Consequently the interval contracts at the root-N rate. This is the normal approximation "
        "to the binomial and is unreliable when Np or N(1-p) is small, where the Wilson or "
        "Clopper-Pearson construction is preferred.",
    36: "The construction is two-sided, so the 90 percent nominal coverage allocates 5 percent to "
        "each tail and the required quantile is the 0.95 point of the standard normal, giving "
        "Z=1.64. The 1.28 value is the 0.90 quantile and corresponds to a one-sided construction; "
        "substituting it conflates one-sided and two-sided coverage. The distinction is the same "
        "one that governs one- versus two-tailed hypothesis tests.",
    37: "Margin = 1.64 * sqrt((0.85)(0.15)/200), approximately 0.041, hence the interval is 0.85 "
        "plus or minus 0.041, approximately [0.809, 0.891]. Under N=50 the interval dilates by a "
        "factor of two, and under N=500 it contracts by a factor of approximately 1.58, both by "
        "the root-N scaling. The coverage statement is frequentist and does not license a "
        "probability statement about the parameter.",
    38: "Negative. Ranking by point estimate ignores the sampling variability of the difference, "
        "so the exercise specifies a Z-statistic for the difference of two proportions. The "
        "decision rule accepts classifier A as superior only when Z exceeds 1.64, which is the "
        "one-sided 0.95 critical value. A difference failing that threshold is not evidence of "
        "equality, merely absence of evidence of difference.",
    39: "Under the specified statistic with A=NB and B=DT: Congressional Voting Records yields Z "
        "approximately -2.69, so DT dominates; Dermatology yields Z approximately 1.67, so NB "
        "dominates; Employee Attrition yields Z approximately -0.27, which fails the critical "
        "value in both directions and is therefore indeterminate. The resulting tallies are one "
        "win, one indeterminate and one loss for each, so the comparison does not identify a "
        "dominant learner. No multiple-comparison correction is applied here.",
    40: "Affirmative, with the stochastic components specified. A random forest is a bagged "
        "ensemble of decision-tree base learners aggregated by plurality vote, with two "
        "independent randomisation mechanisms: bootstrap resampling of the induction sample per "
        "base learner, and random subspace selection of the covariate set at each split. Both "
        "mechanisms act to decorrelate the base learners, which is what reduces the variance "
        "component of the ensemble risk.",
    41: "Consider an ensemble of independent noisy estimators of a common parameter, each with "
        "identical bias and finite variance. Averaging M such estimators leaves the bias "
        "unchanged and reduces the variance by a factor approaching M under independence, and by "
        "a factor governed by the mean pairwise correlation otherwise. Bootstrap resampling and "
        "random subspace selection are the mechanisms that drive that correlation down; the "
        "ensemble is still a collection of decision trees, so the estimator analogy should not be "
        "pushed past the variance argument.",
    42: "The family is ordered by how the resampling scheme partitions the sample. Hold-out is a "
        "single partition; random subsampling is a Monte Carlo replication of hold-out; k-fold "
        "cross-validation induces a partition into k blocks with each serving once as the "
        "evaluation block; stratified cross-validation constrains the partition to preserve the "
        "class marginal; leave-one-out is the limiting case at k equal to sample size; the "
        "bootstrap resamples with replacement and evaluates on the out-of-bag complement. They "
        "trade bias against variance of the risk estimate differently.",
    43: "Negative. Cluster membership is a latent assignment induced from the covariate geometry, "
        "not a prediction of an observed response variable estimated from labelled supervision. "
        "Classification operates under supervision; partition induction does not, and its "
        "objective is defined over a similarity or dissimilarity functional rather than over an "
        "empirical risk with respect to labels. The latent-variable formulation makes this "
        "explicit under a mixture model.",
    44: "Supervised instance: estimate the response Defaulted Borrower over {Y, N} from a labelled "
        "sample of borrower records. Unsupervised instance: induce a partition of a customer "
        "population over purchase-behaviour covariates with no observed group membership. The "
        "first estimates a known response; the second induces latent structure whose validity "
        "must be assessed by an internal index such as silhouette rather than by held-out risk.",
    45: "Initialise K centroids; assign each observation to the centroid minimising Euclidean "
        "distance, which induces a Voronoi partition of the covariate space; recompute each "
        "centroid as the arithmetic mean of its assigned observations; iterate until the "
        "assignment map is stationary. The procedure is coordinate descent on the within-cluster "
        "sum of squared deviations, SSE, and therefore converges monotonically to a local optimum "
        "with no global guarantee.",
    46: "Negative. The centroid is the arithmetic mean of the assigned observations and therefore "
        "need not lie in the support of the empirical distribution. Following one assignment step "
        "in the Exercise 6 instance a centroid attains (3.5, 6.2), which coincides with no "
        "observation. Requiring the exemplar to be an observed point defines a different "
        "objective, namely the K-medoids formulation.",
    47: "Initialising C1=p10=(7,3) and C2=p11=(6.5,1) and applying the assignment step under "
        "Euclidean distance: C1 receives {p5, p6, p7, p8, p10} and C2 receives "
        "{p1, p2, p3, p4, p9, p11}. The objective evaluated at the initial centroids is "
        "SSE=224.750. That this initialisation is arbitrary is the reason k-means++ seeding "
        "exists.",
    48: "By coordinate-wise arithmetic mean over each cluster's assigned observations. Averaging "
        "{p5, p6, p7, p8, p10} yields C1=(3.5, 6.2); averaging {p1, p2, p3, p4, p9, p11} yields "
        "C2 approximately (3.583, 1.583). The mean is the minimiser of the sum of squared "
        "deviations, which is precisely why the update step takes this form and not, say, the "
        "componentwise median.",
    49: "Affirmative; the referent is the assignment step of the coordinate-descent iteration. "
        "Under the updated centroids, C1 receives {p5, p6, p7, p8} and C2 receives "
        "{p1, p2, p3, p4, p9, p10, p11}, giving C1=(2.625, 7.000) and C2 approximately "
        "(4.071, 1.786), with the objective decreasing to SSE approximately 50.407. Monotone "
        "decrease is guaranteed because each step is an exact minimisation in one block of "
        "variables.",
    50: "Because the initialisation confers no persistent status on the observation. Following the "
        "first update, p10 is an ordinary element of the sample and is reassigned by the same "
        "nearest-centroid rule as any other, so it migrates whenever the updated centroid "
        "geometry favours the alternative cell. Initialisation affects only which local optimum "
        "the descent reaches.",
    51: "At C1=(2.625,7.000) and C2=(4.071,1.786) the assignment map is stationary, so the "
        "coordinate-descent iteration has reached a fixed point and terminates with SSE "
        "approximately 42.830. Stationarity of the assignment map is the standard termination "
        "predicate; it certifies local optimality only, and a different initialisation may "
        "terminate at a different fixed point with a larger objective.",
    52: "Consider the two functionals as instances of the Minkowski p-norm on the difference "
        "vector. Manhattan distance is the p=1 case, sum over i of |x_i - y_i|; Euclidean "
        "distance is the p=2 case, the square root of the sum over i of (x_i - y_i) squared. "
        "Both are metrics in the formal sense, satisfying identity, symmetry and the triangle "
        "inequality, so both are dissimilarity functionals for which smaller values denote "
        "greater proximity; the p=1 case is the one that stays well-behaved in high dimension.",
    53: "Cosine similarity is the inner product normalised by the product of Euclidean norms, "
        "x dot y divided by ||x|| times ||y||, hence the cosine of the angle subtended in the "
        "inner-product space. It is a similarity rather than a dissimilarity functional, so "
        "larger values denote greater proximity. Its range is [-1, 1] in general and [0, 1] on "
        "the nonnegative orthant; it is scale-invariant, which is why it is standard for "
        "term-frequency representations.",
    54: "The favourable properties are low computational complexity, linear in sample size per "
        "iteration, and interpretability of the centroid as a cluster representative. The "
        "unfavourable ones are the requirement to specify K exogenously, sensitivity of the "
        "attained local optimum to initialisation, non-robustness of the mean to outlying "
        "observations, and an implicit isotropic-variance assumption that restricts it to "
        "compact, approximately spherical cells of comparable cardinality and density.",
    55: "Not equivalent. Each bisection applies the K=2 instantiation, but the enclosing procedure "
        "is a divisive hierarchical scheme: it initialises with the trivial partition, selects a "
        "cell for bisection under a specified criterion, evaluates candidate bisections and "
        "retains the one minimising SSE, and recurses until the target cardinality is attained. "
        "The output is therefore a dendrogram truncated at K, not a flat partition.",
    56: "Standard K-Means performs a single simultaneous optimisation over all K cells; bisecting "
        "K-Means performs a sequence of nested binary optimisations. The distinguishing property "
        "is the divisive hierarchical control structure, not the cardinality of each local "
        "instantiation. The former is sensitive to global initialisation; the latter is sensitive "
        "instead to the cell-selection criterion at each level.",
    57: "Affirmative. Exercise 1 establishes the supervised induction framework and the design-"
        "matrix formulation; Exercise 2 develops probabilistic classification under a factorised "
        "likelihood with additive smoothing; Exercise 3 treats recursive partitioning, "
        "impurity-based split criteria and penalised complexity; Exercise 4 addresses risk "
        "estimation, interval construction and paired comparison under sampling variability; "
        "Exercise 6 transitions to unsupervised partition induction, where the objective is "
        "defined over a dissimilarity functional rather than over an empirical risk with respect "
        "to observed labels.",
    58: "Evaluate the following proposition: 'K-Means estimates the response variable of each "
        "observation.' Identify the category error, using the terms supervised, unsupervised and "
        "response variable in your answer.",
    59: "Correct. You have distinguished the two objectives without conflation: supervised "
        "induction estimates an observed response from a labelled sample, whereas K-Means induces "
        "a latent partition under a dissimilarity functional with no observed response. The "
        "distinction is one of objective, not of algorithm, which is why the same distance "
        "computations appear in both settings.",
    60: "Suppose the maximum-likelihood estimate of P(Married | DB=Y) is zero while all remaining "
        "DB=Y conditionals are bounded away from zero. State the resulting value of the DB=Y "
        "discriminant under unsmoothed estimation, and name the estimator specified in the "
        "exercise that resolves it.",
    61: "Correct. Stated precisely, the Laplace estimator is (count + 1)/(class_count + |V|), "
        "where |V| is the cardinality of the value domain; the denominator correction is what "
        "preserves normalisation of the smoothed conditional mass function. It is the posterior "
        "mean under a symmetric Dirichlet prior with concentration parameter one, which is the "
        "general form the exercise instantiates.",
    62: "Adopt a four-stage protocol. Identify the inferential task: supervised induction, "
        "probabilistic classification, recursive partitioning, risk estimation, or partition "
        "induction. State the estimator or functional symbolically before numerical substitution. "
        "Identify the conditioning event or normalising margin explicitly. Interpret each "
        "numerical result decision-theoretically in one statement, for instance that the argmax "
        "selects DB=N, or that the interval contracts at the root-N rate.",
}


def sabotage_checks(source: list[dict], t5: list[dict]) -> list[str]:
    """Pre-registration section 'Sabotage checks before submission', run before anything is
    written so a failure blocks the build rather than surfacing after GPU time is spent."""
    problems = []
    n = len(source)

    if sorted(T5_TEXT) != list(range(n)):
        missing = sorted(set(range(n)) - set(T5_TEXT))
        problems.append(f"T5 must rewrite every turn; missing indices: {missing}")

    unchanged = [i for i in range(min(n, len(t5)))
                 if source[i]["tutor"]["text"] == t5[i]["tutor"]["text"]]
    if unchanged:
        problems.append(f"turns identical to T1 (this is a full rewrite): {unchanged}")

    for i in range(min(n, len(t5))):
        if source[i]["student"]["text"] != t5[i]["student"]["text"]:
            problems.append(f"student turn {i} is not byte-identical to dm1")

    for i, item in enumerate(t5):
        if not item["tutor"]["text"].strip():
            problems.append(f"tutor turn {i} is empty")

    # An analogy turn must still carry an analogy, else the condition has degraded scaffolding
    # rather than register -- which would make it a second answer-dumping arm.
    for i in sorted(ANALOGY_TURNS):
        text = t5[i]["tutor"]["text"].lower() if i < len(t5) else ""
        if "consider" not in text and "analog" not in text:
            problems.append(f"analogy turn {i} carries no analogy")

    # Length is a crude but real proxy for "did the rewrite drop content rather than reword it".
    for i in range(min(n, len(t5))):
        orig, new = len(source[i]["tutor"]["text"]), len(t5[i]["tutor"]["text"])
        if new < 0.6 * orig:
            problems.append(f"turn {i} lost {100 * (1 - new / orig):.0f}% of its length "
                            f"({orig} -> {new}); content may have been dropped, not reworded")

    return problems


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    n = len(source)

    t5, records = [], []
    for i, item in enumerate(source):
        text = T5_TEXT.get(i, item["tutor"]["text"])
        t5.append({"student": {"text": item["student"]["text"]}, "tutor": {"text": text}})
        records.append({
            "exchange_index": i,
            "exchange_id": f"ex_{i:04d}",
            "analogy_requested": i in ANALOGY_TURNS,
            "original_tutor_text": item["tutor"]["text"],
            "variant_tutor_text": text,
        })

    problems = sabotage_checks(source, t5)
    if problems:
        print("SABOTAGE CHECKS FAILED - nothing written:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    out = OUT_DIR / "tv_e_source.json"
    out.write_text(json.dumps(t5, indent=2, ensure_ascii=False), encoding="utf-8")

    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(json.dumps({
        "schema": "expert_register_variant_ground_truth.v1",
        "preregistration": "docs/evaluation_completeness_v2/28_EXPERT_REGISTER_PREREGISTRATION.md",
        "source_dialogue": str(SOURCE.relative_to(REPO_ROOT)),
        "variant": {
            "id": "T5_expert_register",
            "opaque_dialogue_id": "tv_e_20260909",
            "source": "tv_e_source.json",
            "design": "full rewrite; degradation is delivery register only",
        },
        "transformation_rule": [
            "formal register: technical vocabulary the dialogue never introduces",
            "gratuitous forward reference to material the student has not met",
            "no signposting: enumerations collapsed into dense blocks",
        ],
        "preserved": {
            "diagnosis": "the tutor still notices and names the student's error",
            "scaffolding": "hints, elicitation prompts and worked steps still present",
            "factual_correctness": "every claim preserved from T1",
            "dialogue_structure": "self-consistency, dependency order and resolution untouched",
        },
        "analogy_turns": sorted(ANALOGY_TURNS),
        "analogy_note": (
            "The student requests an analogy at these indices. T5 supplies one at each, drawn "
            "from a domain a beginner will not follow. Removing it would be a scaffolding "
            "failure; supplying an inaccessible one is a calibration failure."
        ),
        "blinding_note": (
            "The opaque id tv_e carries no description of the condition. dialogue_id is embedded "
            "in every segment and exchange id and therefore appears inside each judge prompt."
        ),
        "exchanges_total": n,
        "exchanges_modified": len(records),
        "modified_exchanges": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    orig_chars = sum(len(x["tutor"]["text"]) for x in source)
    new_chars = sum(len(x["tutor"]["text"]) for x in t5)
    print("  sabotage checks passed")
    print(f"  {n} exchanges, all {len(records)} rewritten")
    print(f"  tutor text: {orig_chars} -> {new_chars} chars "
          f"({100 * new_chars / orig_chars:.0f}% of T1)")
    print(f"  analogy turns preserved: {sorted(ANALOGY_TURNS)}")
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    print(f"  wrote {GROUND_TRUTH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
