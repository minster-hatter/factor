"""A worked example of The Factor Principle by Alan Turing.

Prior Information
-----------------
"Suppose that one man in five dies of heart failure..."
"... of the men who die of heart failure two in three die in their beds..."
"... of the men who die from other causes only one in four dies in their
beds..."
"""
import math

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# Define the model with H as heart failure hypothesis and B as bed.
model = DiscreteBayesianNetwork([('H', 'B')])
cpd_H = TabularCPD(
    variable='H',
    variable_card=2,
    values = [
        [4/5],  # P(H=0) = 1 - P(H=1).
        [1/5],  # P(H=1), which is given above.
    ]
)
cpd_B = TabularCPD(
    variable='B',
    variable_card=2,
    evidence=['H'],
    evidence_card=[2],
    # The below is calculated from the pre-amble.
    values=[
        [3/4, 1/3],  # P(B=0 | H=0), P(B=0 | H=1).
        [1/4, 2/3],  # P(B=1 | H=1), P(B=1 | H=1), both given above.
    ]
)
model.add_cpds(cpd_H, cpd_B)
print(f'The model is specified well: {model.check_model()}.')
# Calculating P(H | B).
infer = VariableElimination(model)
P_H1_given_B1 = infer.query(variables=['H'], evidence={'B': 1}).values[1]
print(
    'The posterior probability for heart failure, given that the person '
    f'was found in bed, was {round(P_H1_given_B1, 2)}.'
)
# Bayes factors.
P_B1_given_H1 = infer.query(variables=['B'], evidence={'H': 1}).values[1]
P_B1_given_H0 = infer.query(variables=['B'], evidence={'H': 0}).values[1]
K = P_B1_given_H1 / P_B1_given_H0
W = 10 * math.log10(K)
print(f'The Bayes factor, K, for this was {round(K, 2)}.')
print(f'The weight of evidence for this was {round(W, 2)} dHarts.')
# Prior odds and weight.
P_H1 = infer.query(variables=['H']).values[1]
P_H0 = infer.query(variables=['H']).values[0]
prior_odds = P_H1 / P_H0
W_prior = 10 * math.log10(prior_odds)
print(f'The prior odds for heart failure were {round(prior_odds, 2)}.')
print(
    f'The prior weight of evidence for heart failure was {round(W_prior, 2)} '
    'dHarts.'
)
print(f'The posterior odds for heart failure were {round(prior_odds * K, 2)}.')

