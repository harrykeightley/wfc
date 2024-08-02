from typing import Callable, Iterable, Literal, Optional, TypeVar
from abc import ABC, abstractmethod

import random


T = TypeVar("T")
Update = Callable[[T], T]
Weighted = tuple[T, int]
StepResult = Literal["Pending", "Result", "Contradiction"]

INVALID_ENTROPY = -1
STABLE_ENTROPY = 0


def wavefunction_collapse[
    State, Element
](
    state: State,
    get_elements: Callable[[State], Iterable[Element]],
    entropy: Callable[[State, Element], int],
    actions: Callable[[State, Element], list[Weighted[Update[State]]]],
    propagate: Callable[[State, Element], State],
    display: Callable[[State], None] = lambda _: None,
):
    while True:
        res, state = step(state, get_elements, entropy, actions, propagate)
        display(state)

        if res != "Pending":
            break

    return state


def step[
    State, Element
](
    state: State,
    get_elements: Callable[[State], Iterable[Element]],
    entropy: Callable[[State, Element], int],
    actions: Callable[[State, Element], list[Weighted[Update[State]]]],
    propagate: Callable[[State, Element], State],
) -> tuple[StepResult, State]:

    unstable_elements = list(
        filter(lambda element: entropy(state, element) != 0, get_elements(state))
    )
    next_element = min(unstable_elements, key=lambda element: entropy(state, element))
    if (entropy(state, next_element)) == INVALID_ENTROPY:
        return ("Contradiction", state)

    if (len(unstable_elements)) == 0:
        return ("Result", state)

    weighted_actions = actions(state, next_element)
    just_actions, weights = zip(*weighted_actions)
    action = random.choices(just_actions, weights=weights)[0]

    return ("Pending", propagate(action(state), next_element))


class WFC[State, Element](ABC):
    def __init__(self, initial_state: State) -> None:
        self.state = initial_state
        self._status: StepResult = "Pending"

    @property
    def status(self):
        return self._status

    @abstractmethod
    def get_elements(self) -> Iterable[Element]:
        pass

    @abstractmethod
    def entropy(self, element: Element) -> int:
        pass

    @abstractmethod
    def actions(self, element: Element) -> list[Weighted[Update[State]]]:
        pass

    @abstractmethod
    def propagate(self, last_collapsed_element: Element) -> None:
        pass

    def _is_unstable(self, element: Element) -> bool:
        return self.entropy(element) != STABLE_ENTROPY

    def step(self):
        unstable_elements = list(filter(self._is_unstable, self.get_elements()))

        if (len(unstable_elements)) == 0:
            self._status = "Result"
            return

        next_element = min(unstable_elements, key=lambda element: self.entropy(element))

        if (self.entropy(next_element)) == INVALID_ENTROPY:
            self._status = "Contradiction"
            return

        weighted_actions = self.actions(next_element)
        just_actions, weights = zip(*weighted_actions)
        action = random.choices(just_actions, weights=weights)[0]
        self.state = action(self.state)
        self.propagate(next_element)

    def run_to_completion(self):
        while self.status == "Pending":
            self.step()
