from dataclass import dataclass


@dataclass
class Prompt:
    """
    Prompt initated by human for task-oriented dialogue
    """
    domain: str
    task: str
    initiator: str
    role: str


@dataclass
class Feedback:
    """
    Feedback collected during robot-human interaction
    """
    domain: str
    task: str
    rate: str
    content: str


@dataclass
class Knowledge:
    """
    Knowledge shared by human  
    """
    domain: str
    content: str
    source: str
    sharee: str


@dataclass
class Conversation:
    "Human-human conversation"
    domain: str
    participant: str
    mesage: str
