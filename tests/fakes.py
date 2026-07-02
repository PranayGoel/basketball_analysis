"""
A minimal, stdlib-only fake replicating just the shape of the `openai` SDK client
(`.chat.completions.create(...).choices[0].message.content` /
`.choices[0].message.tool_calls`) that llm_client.py and game_qa.py depend on.

This exists so tests never need the real `openai` package installed -- everything
under test here talks to `client` via dependency injection (a plain parameter), so a
duck-typed fake is all that's needed to exercise the logic for real.
"""


class FakeToolCallFunction:
    def __init__(self, name, arguments_json):
        self.name = name
        self.arguments = arguments_json


class FakeToolCall:
    def __init__(self, id, name, arguments_json):
        self.id = id
        self.function = FakeToolCallFunction(name, arguments_json)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeCompletions:
    """
    Records every call it receives and returns responses from a pre-supplied queue,
    one per call -- lets a test script a multi-turn tool-calling exchange (e.g. "first
    call requests a tool, second call returns the final answer").
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeCompletions ran out of scripted responses")
        return self._responses.pop(0)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    """Duck-types the one surface llm_client.py/game_qa.py actually touch: client.chat.completions.create(...)."""

    def __init__(self, responses):
        self.chat = FakeChat(FakeCompletions(responses))

    @property
    def calls(self):
        return self.chat.completions.calls
