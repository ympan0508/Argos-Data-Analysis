import json
import re
from typing import AsyncGenerator, List, Literal, Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    MultiModalMessage,
    TextMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from pydantic import BaseModel


def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


class PlanningFormat(BaseModel):
    thought: str
    subtask_description: str
    team_for_subtask: Literal["visual_team", "programmatic_team"]


class ArgosTeamProxyAgent(BaseChatAgent):
    def __init__(
        self,
        name,
        description,
        *,
        visual_coder: BaseChatAgent,
        visual_executor: BaseChatAgent,
        visual_reflector: BaseChatAgent,
        program_coder: BaseChatAgent,
        program_executor: BaseChatAgent,
        program_reflector: BaseChatAgent,
        summarizer: BaseChatAgent,
        inner_termination=TextMentionTermination("APPROVE"),
        max_reflection_rounds=3,
        rework_flag="REWORK",
        EXECUTED_OUTPUT_LIMIT_THRESHOLD=2000,
    ):
        super().__init__(name, description)
        self._visual_coder = visual_coder
        self._visual_executor = visual_executor
        self._visual_reflector = visual_reflector
        self._program_coder = program_coder
        self._program_executor = program_executor
        self._program_reflector = program_reflector
        self._summarizer = summarizer

        self._rework_flag = rework_flag

        inner_max_turn = max_reflection_rounds * 3 - 1
        self._visual_team = RoundRobinGroupChat(
            participants=[
                self._visual_coder,
                self._visual_executor,
                self._visual_reflector,
            ],
            termination_condition=inner_termination,
            max_turns=inner_max_turn,
        )

        self._program_team = RoundRobinGroupChat(
            participants=[
                self._program_coder,
                self._program_executor,
                self._program_reflector,
            ],
            termination_condition=inner_termination,
            max_turns=inner_max_turn,
        )

        self._run_flag = {"visual_team": False, "programmatic_team": False}
        self._EXECUTED_OUTPUT_LIMIT_THRESHOLD = EXECUTED_OUTPUT_LIMIT_THRESHOLD

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage, MultiModalMessage)

    async def on_messages(
        self, messages: Sequence[ChatMessage],
        cancellation_token: CancellationToken
    ) -> Response:
        # Calls the on_messages_stream.
        response: Response | None = None
        async for message in self.on_messages_stream(messages,
                                                     cancellation_token):
            if isinstance(message, Response):
                response = message
        assert response is not None
        return response

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage],
        cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:

        # last message must be from planner
        plan = messages[-1].content
        plan_obj = json.loads(plan)
        assert isinstance(plan_obj, dict)
        assert (
            "thought" in plan_obj
            and "subtask_description" in plan_obj
            and "team_for_subtask" in plan_obj
            and plan_obj["team_for_subtask"] in ("visual_team",
                                                 "programmatic_team")
        )

        task_description = plan_obj["subtask_description"]
        task_team = plan_obj["team_for_subtask"]
        if task_team == "visual_team":
            team = self._visual_team
            coder = self._visual_coder
            executor = self._visual_executor
            reflector = self._visual_reflector
        else:
            team = self._program_team
            coder = self._program_coder
            executor = self._program_executor
            reflector = self._program_reflector

        stream = team.run_stream(
            task=task_description, cancellation_token=cancellation_token
        )
        task_message_flag = True
        coder_message = executor_message = reflector_message = None
        inner_messages = []
        async for inner_msg in stream:
            # skip the task message (always the first)
            if task_message_flag:
                task_message_flag = False
                continue
            if isinstance(inner_msg, TaskResult):
                result = inner_msg
                assert result is not None
            else:
                if inner_msg.source == coder.name:
                    coder_message = inner_msg
                elif inner_msg.source == executor.name:
                    executor_message = inner_msg
                elif inner_msg.source == reflector.name:
                    reflector_message = inner_msg  # noqa: F841
                yield inner_msg
                inner_messages.append(inner_msg)
        assert coder_message is not None and executor_message is not None

        self._run_flag[task_team] = True

        code_blocks = extract_markdown_code_blocks(coder_message.content)
        code = "\n".join([block.code for block in code_blocks])
        execute_output = executor_message.content

        if len(execute_output) > self._EXECUTED_OUTPUT_LIMIT_THRESHOLD:
            execute_output = (
                execute_output[: self._EXECUTED_OUTPUT_LIMIT_THRESHOLD]
                + f"\n\n...output exceeded limitation and there are "
                f"{len(execute_output)-self._EXECUTED_OUTPUT_LIMIT_THRESHOLD}"
                " more characters remaining"
            )

        if isinstance(executor_message, TextMessage):
            executed_output_content = (
                f"Given the task: {task_description}\n\n"
                f"The {task_team} executed the following script:\n"
                f"```python\n{code}\n```\n\n"
                f"The output of the executed script is:\n"
                f"```\n{execute_output}\n```"
            )
            executed_output_message = TextMessage(
                source=self.name, content=executed_output_content
            )

        elif isinstance(executor_message, MultiModalMessage):
            executed_output_content = (
                f"Given the task: {task_description}\n\n"
                f"The {task_team} executed the following script:\n\n"
                f"```python\n{code}\n```\n\n"
                f"The output of the executed script is:\n"
                f"```\n{execute_output[0]}\n```"
            )
            executed_output_message = MultiModalMessage(
                source=self.name, content=[
                    executed_output_content, *execute_output[1:]]
            )
        else:
            raise ValueError(
                f"Unexpected message type {type(executor_message)}")

        inner_messages.append(executed_output_message)
        response = await self._summarizer.on_messages(
            messages=[
                executed_output_message], cancellation_token=cancellation_token
        )
        await self._summarizer.on_reset(cancellation_token=cancellation_token)
        yield response

        if self._run_flag["visual_team"]:
            self._run_flag["visual_team"] = False
            await self._visual_team.reset()
        if self._run_flag["programmatic_team"]:
            self._run_flag["programmatic_team"] = False
            await self._program_team.reset()

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        if self._run_flag["visual_team"]:
            self._run_flag["visual_team"] = False
            await self._visual_team.reset()
        if self._run_flag["programmatic_team"]:
            self._run_flag["programmatic_team"] = False
            await self._program_team.reset()
