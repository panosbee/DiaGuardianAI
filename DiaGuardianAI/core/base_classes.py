# DiaGuardianAI Base Classes
# Defines abstract base classes (ABCs) for core components
# to ensure a common interface and facilitate extensibility.

from abc import ABC, abstractmethod
from typing import Optional, Any, Union, List, Dict, Tuple


class BaseSyntheticPatient(ABC):
    """Abstract base class for synthetic patient models.

    Defines the interface for patient models used in the simulation,
    including initialization, stepping through time, and retrieving
    glucose readings.
    """
    @abstractmethod
    def __init__(self, params: Dict[str, Any]):
        """Initializes the synthetic patient.

        Args:
            params (Dict[str, Any]): A dictionary of patient-specific
                parameters, such as Insulin Sensitivity Factor (ISF),
                Carb Ratio (CR), weight, initial glucose levels, etc.
                The exact parameters will depend on the concrete
                implementation of the patient model.
        """
        pass

    @abstractmethod
    def step(self,
             basal_insulin: float,
             bolus_insulin: float,
             carbs_details: Optional[Dict[str, Any]] = None,
             protein_ingested: float = 0.0,
             fat_ingested: float = 0.0,
             exercise_event: Optional[Dict[str, Any]] = None):
        """Advances the patient simulation by one time step.

        This method should update the patient's internal physiological
        state, including glucose levels, based on the inputs.

        Args:
            basal_insulin (float): The current basal insulin rate (e.g., U/hr).
            bolus_insulin (float): The amount of bolus insulin administered
                at this time step (e.g., U).
            carbs_details (Optional[Dict[str, Any]]): Details of carbohydrates ingested.
                Expected keys: "grams" (float), "gi_factor" (float, optional, default 1.0).
                Example: `{"grams": 50, "gi_factor": 1.2}`. Defaults to None.
            protein_ingested (float): The amount of protein ingested at
                this time step (e.g., grams). Defaults to 0.0.
            fat_ingested (float): The amount of fat ingested at
                this time step (e.g., grams). Defaults to 0.0.
            exercise_event (Optional[Dict[str, Any]]): A dictionary
                describing an exercise event occurring during this time step.
                Example: `{"duration_minutes": 30, "intensity": "moderate"}`.
                Defaults to None if no exercise.
        """
        pass

    @abstractmethod
    def get_cgm_reading(self) -> float:
        """Returns the current Continuous Glucose Monitoring (CGM) reading.

        This value represents the glucose level as measured by a CGM
        sensor, which might include noise or delay in a realistic model.

        Returns:
            float: The current CGM glucose reading (e.g., in mg/dL).
        """
        pass

    def get_internal_states(self) -> Dict[str, Any]:
        """(Optional) Returns internal physiological states of the model.

        This method can provide access to other variables of the patient
        model beyond the CGM reading, which might be useful for
        debugging, advanced analysis, or more complex agent states.

        Returns:
            Dict[str, Any]: A dictionary of internal states. Default
                implementation returns an empty dictionary.
        """
        return {}

    @abstractmethod
    def reset(self, initial_state_override: Optional[Dict[str, Any]] = None) -> None:
        """Resets the patient to a defined initial state.

        This method should re-initialize the patient's physiological
        variables to their starting conditions, allowing for new simulation
        episodes to begin from a consistent or specified state.

        Args:
            initial_state_override (Optional[Dict[str, Any]]): A dictionary
                that can be used to override default initial state parameters
                (e.g., starting glucose, IOB, COB). If None, the patient
                should reset to its standard default initial state.
                Defaults to None.
        """
        pass


class BasePredictiveModel(ABC):
    """Abstract base class for glucose predictive models.

    Defines the interface for models that forecast future glucose values
    based on current and historical data.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        """Initializes the predictive model.

        Concrete implementations will define their specific parameters
        through `**kwargs` or explicit arguments.
        """
        pass

    @abstractmethod
    def train(self, data: Any, targets: Optional[Any] = None, **kwargs):
        """Trains the predictive model.

        Args:
            data (Any): Training data. This could be features, a list of
                DataFrames, or any format suitable for the concrete model.
            targets (Optional[Any]): Training targets (e.g., future glucose values).
                May be optional if targets are part of `data` or handled internally.
            **kwargs: Additional training parameters (e.g., epochs, batch_size).
        """
        pass

    @abstractmethod
    def predict(self, current_input: Any, **kwargs) -> Dict[str, List[float]]:
        """Makes glucose predictions based on the current input.

        Args:
            current_input (Any): Input data representing the current
                state or history, formatted as expected by the specific model
                implementation (e.g., a DataFrame of recent history).
            **kwargs: Additional prediction parameters.

        Returns:
            Dict[str, List[float]]: A dictionary containing predictions.
                Expected keys might include "mean" for point predictions,
                and "lower_bound"/"upper_bound" or "std_dev" for uncertainty.
                Each value is a list of floats for defined prediction horizons.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Saves the trained model to the specified path.

        Args:
            path (str): The file path where the model should be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Loads a pre-trained model from the specified path.

        Args:
            path (str): The file path from which to load the model.
        """
        pass


class BaseAgent(ABC):
    """Abstract base class for decision-making agents.

    Defines the interface for agents that interact with the simulation
    environment, making decisions (e.g., insulin dosing) based on
    the observed state and predictions.
    """
    @abstractmethod
    def __init__(self, state_dim: int, action_space: Any, predictor: Optional[BasePredictiveModel] = None, **kwargs):
        """Initializes the agent.

        Args:
            state_dim (int): The dimensionality of the state space that
                the agent observes.
            action_space (Any): The definition of the agent's action
                space. This could be an object from a library like
                Gymnasium (e.g., `gym.spaces.Discrete`,
                `gym.spaces.Box`).
            predictor (Optional[BasePredictiveModel]): An optional
                predictive model instance that the agent can use to get
                glucose forecasts. Defaults to None.
            **kwargs: Additional keyword arguments for specific agent
                implementations.
        """
        self.state_dim = state_dim
        self.action_space = action_space
        self.predictor = predictor
        pass

    @abstractmethod
    def decide_action(self, current_state: Any, **kwargs) -> Any:
        """Decides an action based on the current state and other information.

        Args:
            current_state (Any): The current state representation observed
                by the agent. The structure depends on the specific agent
                implementation.
            **kwargs: Additional keyword arguments that might be passed to
                the agent, such as glucose predictions
                (`predictions=List[float]`) or suggestions from other
                components.

        Returns:
            Any: The action to be taken by the agent. The structure of
                the action (e.g., a dictionary, a numerical value)
                depends on the agent's action space and how it's
                interpreted by the simulation environment.
        """
        pass

    def learn(self, experience: Any):
        """(Optional) Allows the agent to learn from an experience.

        This method is particularly relevant for reinforcement learning
        agents. The structure of the `experience` argument will depend
        on the learning algorithm (e.g., a tuple like
        `(state, action, reward, next_state, done)`).

        Args:
            experience (Any): The experience data from which the agent can learn.
        """
        pass

    def save(self, path: str):
        """(Optional) Saves the agent's state or learned model.

        Args:
            path (str): The file path where the agent's data should be saved.
        """
        pass

    def load(self, path: str):
        """(Optional) Loads the agent's state or learned model.

        Args:
            path (str): The file path from which to load the agent's data.
        """
        pass


class BasePatternRepository(ABC):
    """Abstract base class for a repository of treatment patterns.

    Defines the interface for storing, retrieving, and managing
    patterns that have been identified as effective or interesting
    during simulations or learning processes.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        """Initializes the pattern repository.

        Concrete implementations will define their specific parameters.
        """
        pass

    @abstractmethod
    def add_pattern(self, pattern_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Adds a new pattern to the repository.

        Args:
            pattern_data (Dict[str, Any]): A dictionary containing the
                core data of the pattern (e.g., state features, action
                taken, outcome).
            metadata (Optional[Dict[str, Any]]): Additional metadata
                associated with the pattern (e.g., source, confidence,
                timestamp).
                Defaults to None.
        """
        pass

    @abstractmethod
    def retrieve_relevant_patterns(self, current_state_features: Any, n_top_patterns: int = 1) -> List[Dict[str, Any]]:
        """Retrieves patterns from the repository relevant to the current state.

        Args:
            current_state_features (Any): Features describing the current
                state, used to find similar or matching patterns in the
                repository.
            n_top_patterns (int): The maximum number of top relevant
                patterns to return. Defaults to 1.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each
                dictionary represents a retrieved pattern.
        """
        pass

    @abstractmethod
    def update_pattern_effectiveness(self, pattern_id: Any, new_outcome_data: Dict[str, Any]):
        """Updates the stored effectiveness or outcome information for a pattern.

        This can be used to refine the value or relevance of stored
        patterns based on new observations.

        Args:
            pattern_id (Any): The unique identifier of the pattern to update.
            new_outcome_data (Dict[str, Any]): A dictionary containing
                data about the new outcome observed when this pattern (or a
                similar one) was applied (e.g., reward achieved,
                resulting TIR).
        """
        pass

    @abstractmethod
    def get_pattern_by_id(self, pattern_id: Any) -> Optional[Dict[str, Any]]:
        """Retrieves a specific pattern by its unique ID.

        Args:
            pattern_id (Any): The unique identifier of the pattern to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The pattern dictionary if found, else None.
        """
        pass