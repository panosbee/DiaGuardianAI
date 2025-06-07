"""Module for intelligent agents in the DiaGuardianAI library.

This package includes various types of agents responsible for making decisions
related to diabetes management, such as insulin dosing. It also includes
specialized sub-systems like meal detection. The primary agents are often
based on Reinforcement Learning (RL).

Key Components:
    - `RLAgent` (in `decision_agent.py`): The main RL-based agent for
      making therapy decisions (e.g., basal/bolus adjustments).
    - `PatternAdvisorAgent`: An agent designed to learn from past successful
      patterns and advise the `RLAgent`.
    - `MealDetector`: A component (which can be rule-based or ML-based) to
      detect meal events from CGM data and other inputs, providing crucial
      information to the decision-making agents.
"""

# Optionally, make key agent classes available at this level.
# from .decision_agent import RLAgent
# from .pattern_advisor_agent import PatternAdvisorAgent
# from .meal_detector import MealDetector