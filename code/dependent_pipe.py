"""Often, evidence is causally dependent on over evidence. This situation
can be handled more easily using Bayesian networks. For example, with
H -> E1 -> E2 as a causal structure.

Here,
    P(H=1) = P(H=0) = 0.5,
    P(E1=1 | H=1) = 0.3, P(E1=1 | H=0) = 0.1,
    P(E2=1 | E1=1) = 0.6, P(E2 = 1 | E1=0).
    O(E1, E2 | H) = 3 (approx. 4.8 dHart).
"""
import math

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# Define the model.
model = DiscreteBayesianNetwork([('H', 'E1'), ('E1', 'E2')])
# H: hypothesis (0 = false, 1 = true).
cpd_H = TabularCPD(
    variable='H',
    variable_card=2,
    values=[[0.5], [0.5]]
)
# E1 which is also a biary variable (H -> E1).
cpd_E1 = TabularCPD(
    variable='E1',
    variable_card=2,
    evidence=['H'],
    evidence_card=[2],
    values=[
        [0.9, 0.7],  # P(E1=0 | H=0), P(E1=0|H=1)
        [0.1, 0.3],  # P(E1=1 | H=0), P(E1=1|H=1)
    ]
)
# E2, a final binary variable (H -> E2, E1-> E2).
cpd_E2 = TabularCPD(
    variable='E2',
    variable_card=2,
    evidence=['E1'],
    evidence_card=[2],
    values=[
        [0.8, 0.4],  # P(E2=1 | E1=0), P(E2=1 | E1=1).
        [0.2, 0.6],  # P(E2=1 | E1=0), P(E2=1 | E1=1).
    ]
)
model.add_cpds(cpd_H, cpd_E1, cpd_E2)
print(f'The model is specified well: {model.check_model()}.')
# Inference.
infer = VariableElimination(model)
# Marginal likelihoods for E1.
P_E1_given_H1 = infer.query(
    variables=['E1'],
    evidence={'H': 1},
).values[1]  # This gives P(E1=1 | H=1).
P_E1_given_H0 = infer.query(
    variables=['E1'],
    evidence={'H': 0},
).values[1]  # This gives P(E1=1 | H=0).
# Evidence and likelihoods for E1.
K1 = P_E1_given_H1 / P_E1_given_H0
W1 = 10 * math.log10(K1)
# Marginal likelihoods for E2.
P_E2_given_H1_E1 = infer.query(
    variables=['E2'],
    # Future note: is H needed when E1 is known?
    evidence={'H': 1, 'E1': 1}
).values[1]
P_E2_given_H0_E1 = infer.query(
    variables=['E2'],
    evidence={'H': 0, 'E1': 1}
).values[1]
# Evidence and likelihoods for E2. Use "prime" to denote dependent.
K2_prime = P_E2_given_H1_E1 / P_E2_given_H0_E1
W2_prime = 10 * math.log10(K2)
# Report findings.
print(f'P(E1=1 | H=1) = {round(P_E1_given_H1, 2)}.')
print(f'P(E1=1 | H=0) = {round(P_E1_given_H0, 2)}.')
print(f'The Bayes factor for E1 was {round(K1, 2)}.')
print(f'The weight of evidence for E1 was {round(W1, 2)} dHart.\n')
# Note that H does not have an impact as it only goes through E1.
print(f'P(E2=1 | H=1, E1=1) = {round(P_E2_given_H1_E1, 2)}.')
print(f'P(E2=1 | H=0, E1=1) = {round(P_E2_given_H0_E1, 2)}.')
print(f'The Bayes factor for E1 was {round(K2, 2)}.')
print(f'The weight of evidence for E1 was {round(W2, 2)} dHart.\n')
# The Bayes factor can be calculated directly using:
P_E1E2_given_H1 = infer.query(
    variables=['E1', 'E2'],
    evidence={'H': 1}
).values[1, 1]  # E1=1, E2=1
P_E1E2_given_H0 = infer.query(
    variables=['E1', 'E2'],
    evidence={'H': 0}
).values[1, 1]
K_total_direct = P_E1E2_given_H1 / P_E1E2_given_H0
W_total_direct = 10 * math.log10(K_total_direct)
print(f'P(E1=1, E2=1 | H=1) = {round(P_E1E2_given_H1, 2)}.')
print(f'P(E1=1, E2=1 | H=0) = {round(P_E1E2_given_H0, 2)}.')
print(f'The Bayes factor for E1 and E2 was {round(K_total_direct, 2)}.')
print(
    'The weight of evidence for E1 and E2 was '
    f'{round(W_total_direct, 2)} dHart.\n'
)
