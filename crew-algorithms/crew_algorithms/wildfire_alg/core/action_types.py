"""Simulator action data, independent of model clients and API credentials."""

from pydantic import BaseModel


class Action(BaseModel):
    done: bool
    action: int
    x: int
    y: int
    explanation: str

    def print_action(self) -> None:
        print([self.done, self.action, self.x, self.y, self.explanation])
