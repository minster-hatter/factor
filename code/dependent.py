"""Often, evidence is causally dependent on over evidence. This situation
can be handled more easily using Bayesian networks. For example, with
H -> E1 -> E2, H -> E2 as a causal structure.
"""
import math

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# Define the model.
model = DiscreteBayesianNetwork([('H', 'E1'), ('E1', 'E2'), ('H', 'E2')])
# H: hypothesis (0 = false, 1 = true).
cpd_H = TabularCPD(
    variable='H',
    variable_card=2,
    values=[[0.7], [0.3]]
)
# E1 which is also a binary variable (H -> E1).
cpd_E1 = TabularCPD(
    variable='E1',
    variable_card=2,
    evidence=['H'],
    evidence_card=[2],
    values=[
        [0.8, 0.3],  # P(E1=0 | H=0), P(E1=0|H=1)
        [0.2, 0.7],  # P(E1=1 | H=0), P(E1=1|H=1)
    ]
)
# E2, a final binary variable (H -> E2, E1-> E2).
cpd_E2 = TabularCPD(
    variable='E2',
    variable_card=2,
    evidence=['H', 'E1'],
    evidence_card=[2, 2],
    values=[
        # E2=0 row.
        [
            0.9,  # P(E2=0 | H=0, E1=0)
            0.6,  # P(E2=0 | H=0, E1=1)
            0.7,  # P(E2=0 | H=1, E1=0)
            0.2,  # P(E2=0 | H=1, E1=1)
        ],
        # E2=1 row.
        [
            0.1,  # P(E2=1 | H=0, E1=0)
            0.4,  # P(E2=1 | H=0, E1=1)
            0.3,  # P(E2=1 | H=1, E1=0)
            0.8,  # P(E2=1 | H=1, E1=1)
        ]
    ]
)
model.add_cpds(cpd_H, cpd_E1, cpd_E2)
print(f'The model is specified well: {model.check_model()}.')
# Inference.
infer = VariableElimination(model)
# Prior odds.
P_H = infer.query(variables=['H'])
K_prior = P_H.values[1] / P_H.values[0]
W_prior = 10 * math.log10(K_prior)
# Marginal likelihoods.
P_E1_E2_given_H1 = infer.query(  # Joint probability under H=1.
    variables=['E1', 'E2'],
    evidence={'H': 1},
).values[1, 1]  # This gives P(E1=1, E2=1 | H=1).
P_E1_E2_given_H0 = infer.query(  # Joint probability under H=0.
    variables=['E1', 'E2'],
    evidence={'H': 0},
).values[1, 1]  # This gives P(E1=1, E2=1 | H=0).
# Evidence Bayes factor.
K_E1E2 = P_E1_E2_given_H1 / P_E1_E2_given_H0
W_E1E2 = 10 * math.log10(K_E1E2)
#Sequential (Turing-style) factors ---
P_E1_H1 = infer.query(
    variables=['E1'],
    evidence={'H': 1}
).values[1]
P_E1_H0 = infer.query(
    variables=['E1'],
    evidence={'H': 0}
).values[1]
K1 = P_E1_H1 / P_E1_H0  # Factor for E1 alone.
W1 = 10 * math.log10(K1)
P_E2_E1_H1 = infer.query(
    variables=['E2'],
    evidence={'H': 1, 'E1': 1}
).values[1]
P_E2_E1_H0 = infer.query(
    variables=['E2'],
    evidence={'H': 0, 'E1': 1}
).values[1]
# Prime notation used to denote dependent evidence.
K2_prime = P_E2_E1_H1 / P_E2_E1_H0  # Factor for E2 given E1=1.
W2_prime = 10 * math.log10(K2_prime)
# Combined factor (which should match joint calculation...).
K_total_seq = K1 * K2_prime
W_total_seq = W1 + W2_prime
# Report findings.
print(f'The prior odds for H=1 were {round(K_prior, 2)}.')
print(f'The weight of evidence in favour of H1 was {round(W_prior, 2)}.')
print(f'P(E1=1, E2=1 | H=1) = {round(P_E1_E2_given_H1, 2)}.')
print(f'P(E1=1, E2=1 | H=0) = {round(P_E1_E2_given_H0, 2)}.')
print(f'The Bayes factor was: {round(K_E1E2, 2)}.')
print(f'The weight of evidence was {round(W_E1E2, 2)} dHart.')
print('\nSequential (Turing-style) factors:')
print(f'K1 (from E1) = {round(K1, 3)} → W1 = {round(W1, 2)} dHart.')
print(
    f"K2' (from E2 | E1) = {round(K2_prime, 2)} → W2' = "
    f"{round(W2_prime, 2)} dHart."
)
print(f'K_total (Bayes factor) = {round(K_total_seq, 3)} → '
      f'W_total = {round(W_total_seq, 2)} dHart'
)

