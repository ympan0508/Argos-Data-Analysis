import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import MultiModalMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo, UserMessage
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .autogen_argosext.code_executor_agent import (
    MyCodeExecutorAgent,
    MyCodeExecutorAgentWithMultiModalOutput,
)
from .autogen_argosext.team_proxy_agent import (
    ArgosTeamProxyAgent,
    PlanningFormat,
)
from .prompt import (
    DEFAULT_ADDITIONAL_REQUIREMENTS,
    DEFAULT_REPORT_FORMAT,
    PLANNING_AGENT_PROMPT,
    PROGRAMMATIC_CODING_AGENT_PROMPT,
    PROGRAMMATIC_REFLECTOR_AGENT_PROMPT,
    SUMMARIZING_AGENT_PROMPT,
    VISUAL_CODING_AGENT_PROMPT,
    VISUAL_REFLECTOR_AGENT_PROMPT,
)
from .report import (
    ArgosReportGenerator,
    DACOReportGenerator,
    InsightBenchReportGenerator,
)
from .util import (
    get_dataset_summary,
    multimodal_message_to_dict,
    prep_venv_context,
    task_result_to_dict,
    text_message_to_dict,
)

MONKEY_PATCH_TO_SAVE_CALLS = False
MONKEY_PATCH_TO_REMOVE_IMAGES = False
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_API_KEY = None
DEFAULT_BASE_URL = None
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_ACTION_ROUNDS = 5
DEFAULT_MAX_REFLECTION_ROUNDS = 3
DEFAULT_EXECUTOR_TYPE = "local"
DEFAULT_ADDITIONAL_DESCRIPTION = ""


@dataclass
class ModelConfig:
    model_name: str = None
    api_key: str = None
    base_url: str = None
    vision: bool = True
    json_output: bool = True
    family: str = "unknown"
    remove_images: bool = False
    openai_kwargs: Optional[Dict] = field(default_factory=dict)


@dataclass
class ArgosConfig:
    work_dir: str
    dataset_names: List[str]
    question: str
    venv_dir: str
    executor_type: str = DEFAULT_EXECUTOR_TYPE
    max_action_rounds: int = DEFAULT_MAX_ACTION_ROUNDS
    max_reflection_rounds: int = DEFAULT_MAX_REFLECTION_ROUNDS
    additional_description: str = DEFAULT_ADDITIONAL_DESCRIPTION

    default_model_name: str = DEFAULT_MODEL
    default_api_key: str = DEFAULT_API_KEY
    default_base_url: str = DEFAULT_BASE_URL
    default_openai_kwargs: Dict = field(
        default_factory=lambda: {
            "temperature": DEFAULT_TEMPERATURE,
        }
    )

    planning_agent: Optional[ModelConfig] = None
    # summarizing_agent: Optional[ModelConfig] = None
    visual_coding_agent: Optional[ModelConfig] = None
    visual_reflector_agent: Optional[ModelConfig] = None
    visual_summarizing_agent: Optional[ModelConfig] = None
    programmatic_coding_agent: Optional[ModelConfig] = None
    programmatic_reflector_agent: Optional[ModelConfig] = None
    programmatic_summarizing_agent: Optional[ModelConfig] = None
    data_report_agent: Optional[ModelConfig] = None

    data_report_preset: Literal["daco", "insightbench", "default"] = "default"
    data_report_requirements: str = DEFAULT_ADDITIONAL_REQUIREMENTS
    data_report_format: str = DEFAULT_REPORT_FORMAT

    def __post_init__(self):
        def fill(cfg: Optional[ModelConfig]) -> ModelConfig:
            if cfg is None:
                cfg = ModelConfig()
            if cfg.model_name is None:
                cfg.model_name = self.default_model_name
            if cfg.api_key is None:
                cfg.api_key = self.default_api_key
            if cfg.base_url is None:
                cfg.base_url = self.default_base_url
            if cfg.openai_kwargs is None or len(cfg.openai_kwargs) == 0:
                cfg.openai_kwargs = self.default_openai_kwargs.copy()
            return cfg

        self.planning_agent = fill(self.planning_agent)
        # self.summarizing_agent = fill(self.summarizing_agent)
        self.visual_coding_agent = fill(self.visual_coding_agent)
        self.visual_reflector_agent = fill(self.visual_reflector_agent)
        self.visual_summarizing_agent = fill(self.visual_summarizing_agent)
        self.programmatic_coding_agent = fill(self.programmatic_coding_agent)
        self.programmatic_reflector_agent = fill(
            self.programmatic_reflector_agent)
        self.programmatic_summarizing_agent = fill(
            self.programmatic_summarizing_agent)
        self.data_report_agent = fill(self.data_report_agent)

    def to_dict(self):
        return asdict(self)

    def save(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, filename: str) -> "ArgosConfig":
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        def create_model_config(subdata):
            if subdata is None:
                return None
            return ModelConfig(**subdata)

        data['planning_agent'] = create_model_config(
            data.get('planning_agent'))
        # data['summarizing_agent'] = create_model_config(
        #     data.get('summarizing_agent'))
        data['visual_coding_agent'] = create_model_config(
            data.get('visual_coding_agent'))
        data['visual_reflector_agent'] = create_model_config(
            data.get('visual_reflector_agent'))
        data['visual_summarizing_agent'] = create_model_config(
            data.get('visual_summarizing_agent'))
        data['programmatic_coding_agent'] = create_model_config(
            data.get('programmatic_coding_agent'))
        data['programmatic_reflector_agent'] = create_model_config(
            data.get('programmatic_reflector_agent'))
        data['programmatic_summarizing_agent'] = create_model_config(
            data.get('programmatic_summarizing_agent'))
        data['data_report_agent'] = create_model_config(
            data.get('data_report_agent'))
        return cls(**data)


class ArgosAgent:
    _clients = {}
    _client_calls = defaultdict(list)

    @staticmethod
    def _client(
        name: str,
        model_config: Optional[ModelConfig] = None,
        response_format=None,
        monkey_patch_to_save_calls: Optional[bool] = None,
        monkey_patch_to_remove_images: Optional[bool] = None
    ):
        if monkey_patch_to_save_calls is None:
            monkey_patch_to_save_calls = MONKEY_PATCH_TO_SAVE_CALLS
        if monkey_patch_to_save_calls:
            print(
                "Monkey patching openai client "
                f"to save api calls for client: {name}"
            )

        if monkey_patch_to_remove_images is None:
            monkey_patch_to_remove_images = model_config.remove_images or \
                MONKEY_PATCH_TO_REMOVE_IMAGES
        if monkey_patch_to_remove_images:
            print(
                "Monkey patching openai client "
                f"to remove images for client: {name}"
            )

        def monkey_patch_wrapper_for_save_calls(orig):
            async def wrapper(
                self,
                messages,
                *,
                tools=[],
                json_output=None,
                extra_create_args={},
                cancellation_token=None,
            ):
                result = await orig(
                    messages=messages,
                    tools=tools,
                    json_output=json_output,
                    extra_create_args=extra_create_args,
                    cancellation_token=cancellation_token,
                )
                info = {
                    "messages": messages,
                    "result": result,
                }
                ArgosAgent._client_calls[name].append(info)
                return result
            return wrapper

        def monkey_patch_wrapper_for_remove_images(orig):

            def remove_images_in_multimodal_user_message(msg):
                content = msg.content
                if isinstance(content, str):
                    return msg
                else:
                    assert isinstance(content, list), "Content must be a list"
                    new_content_list = [str(e) for e in content]
                    new_content = "\n".join(new_content_list)
                    msg.content = new_content
                    return msg

            async def wrapper(
                self,
                messages,
                *,
                tools=[],
                json_output=None,
                extra_create_args={},
                cancellation_token=None,
            ):
                messages = [remove_images_in_multimodal_user_message(
                    msg) for msg in messages]
                return await orig(
                    messages=messages,
                    tools=tools,
                    json_output=json_output,
                    extra_create_args=extra_create_args,
                    cancellation_token=cancellation_token,
                )
            return wrapper

        if name not in ArgosAgent._clients:
            assert model_config is not None, \
                "Model config must be provided for creating a new client."
            cli = OpenAIChatCompletionClient(
                model=model_config.model_name,
                api_key=model_config.api_key,
                response_format=response_format,
                base_url=model_config.base_url,
                model_info=ModelInfo(
                    vision=model_config.vision,
                    function_calling=False,
                    json_output=model_config.json_output,
                    family=model_config.family
                ),
                **model_config.openai_kwargs
            )
            if monkey_patch_to_save_calls:
                cli.create = monkey_patch_wrapper_for_save_calls(
                    cli.create).__get__(cli, type(cli))
            if monkey_patch_to_remove_images:
                cli.create = monkey_patch_wrapper_for_remove_images(
                    cli.create).__get__(cli, type(cli))
            ArgosAgent._clients[name] = cli
        return ArgosAgent._clients[name]

    async def test_client(self, client_name):
        client = ArgosAgent._client(
            client_name,
            model_config=ModelConfig(
                model_name=DEFAULT_MODEL,
                api_key=DEFAULT_API_KEY,
                base_url=DEFAULT_BASE_URL
            )
        )
        result = await client.create(
            messages=[UserMessage(content="1+1=3?", source="user")]
        )
        print(result)
        return result

    @staticmethod
    def print_client_usage():
        for key, value in ArgosAgent._clients.items():
            print(key, value.actual_usage())

    def _setup_code_executor(self, exe_type, venv_dir, work_dir):
        if exe_type == "local":
            venv_context = prep_venv_context(venv_dir=venv_dir)
            local_executor = LocalCommandLineCodeExecutor(
                work_dir=work_dir, virtual_env_context=venv_context
            )
            self._code_executor = local_executor
        elif exe_type == "docker":
            # You can setup a docker code executor here, please refer to:
            # https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.code_executors.docker.html
            raise NotImplementedError(
                "Docker code executor is not implemented yet. "
            )
        else:
            raise ValueError(
                f"Executor type {exe_type} is not supported. "
                "Please choose from 'local' or 'docker'."
            )

    def __init__(self, config: ArgosConfig):
        self.config = config
        self._work_dir = config.work_dir
        self._dataset_names = config.dataset_names
        self._question = config.question
        self._max_action_rounds = config.max_action_rounds
        self._max_reflection_rounds = config.max_reflection_rounds
        self._additional_description = config.additional_description

        self._code_executor = None
        self._setup_code_executor(
            exe_type=config.executor_type,
            work_dir=self._work_dir,
            venv_dir=config.venv_dir
        )

        self._dataset_summary = get_dataset_summary(
            self._work_dir, self._dataset_names, self._additional_description)

        self.planning_agent = AssistantAgent(
            name="planning_agent",
            description="Plans and delegates tasks for data analysis tasks.",
            model_client=ArgosAgent._client(
                "planning_agent",
                model_config=config.planning_agent,
                response_format=PlanningFormat,
            ),
            system_message=PLANNING_AGENT_PROMPT.format(
                dataset_summary=self._dataset_summary,
                question=self._question,
                max_action_rounds=self._max_action_rounds,
            ),
        )

        self.visual_summarizing_agent = AssistantAgent(
            name="visual_summarizing_agent",
            description=("Summarizes key insights from visualizations."),
            model_client=ArgosAgent._client(
                "visual_summarizing_agent",
                # model_config=config.summarizing_agent,
                model_config=config.visual_summarizing_agent
            ),
            system_message=SUMMARIZING_AGENT_PROMPT,
        )

        self.visual_coding_agent = AssistantAgent(
            name="visual_coding_agent",
            description="Generates Python code for data visualizations.",
            model_client=ArgosAgent._client(
                "visual_coding_agent",
                model_config=config.visual_coding_agent,
            ),
            system_message=VISUAL_CODING_AGENT_PROMPT.format(
                dataset_names=self._dataset_names,
                dataset_summary=self._dataset_summary,
            ),
        )

        self.visual_executor_agent = MyCodeExecutorAgentWithMultiModalOutput(
            name="visual_executor_agent",
            code_executor=self._code_executor,
            working_dir=self._work_dir,
            execute_last_msg=True,
        )

        self.visual_reflector_agent = AssistantAgent(
            name="visual_reflector_agent",
            description=(
                "Reviews output of data visualization programs, "
                "provides feedback to improve the program if necessary."
            ),
            model_client=ArgosAgent._client(
                "visual_reflector_agent",
                model_config=config.visual_reflector_agent,
            ),
            system_message=VISUAL_REFLECTOR_AGENT_PROMPT,
        )

        self.programmatic_summarizing_agent = AssistantAgent(
            name="programmatic_summarizing_agent",
            description=("Summarizes key insights from programmatic output."),
            model_client=ArgosAgent._client(
                "programmatic_summarizing_agent",
                # model_config=config.summarizing_agent,
                model_config=config.programmatic_summarizing_agent
            ),
            system_message=SUMMARIZING_AGENT_PROMPT,
        )

        self.programmatic_coding_agent = AssistantAgent(
            name="programmatic_coding_agent",
            description=("Generates Python code for numerical analysis "
                         "and hypothesis testing."),
            model_client=ArgosAgent._client(
                "programmatic_coding_agent",
                model_config=config.programmatic_coding_agent,
            ),
            system_message=PROGRAMMATIC_CODING_AGENT_PROMPT.format(
                dataset_names=self._dataset_names,
                dataset_summary=self._dataset_summary,
            ),
        )

        self.programmatic_executor_agent = MyCodeExecutorAgent(
            name="programmatic_executor_agent",
            code_executor=self._code_executor,
            execute_last_msg=True,
        )

        self.programmatic_reflector_agent = AssistantAgent(
            name="programmatic_reflector_agent",
            description=(
                "Reviews output of numerical analysis programs, "
                "provides feedback to improve the program if necessary."
            ),
            model_client=ArgosAgent._client(
                "programmatic_reflector_agent",
                model_config=config.programmatic_reflector_agent,
            ),
            system_message=PROGRAMMATIC_REFLECTOR_AGENT_PROMPT,
        )

        self.team_proxy_agent = ArgosTeamProxyAgent(
            name="visualization_or_programming_team",
            description="The proxy agent for visual and programmatic team",
            visual_coder=self.visual_coding_agent,
            visual_executor=self.visual_executor_agent,
            visual_reflector=self.visual_reflector_agent,
            visual_summarizer=self.visual_summarizing_agent,
            program_coder=self.programmatic_coding_agent,
            program_executor=self.programmatic_executor_agent,
            program_reflector=self.programmatic_reflector_agent,
            program_summarizer=self.programmatic_summarizing_agent,
            max_reflection_rounds=self._max_reflection_rounds
        )

        self.brain_trust = RoundRobinGroupChat(
            participants=[
                self.planning_agent,
                self.team_proxy_agent,
            ],
            termination_condition=TextMentionTermination("TERMINATE"),
            max_turns=2 * self._max_action_rounds,
        )

        self.data_report_agent = ArgosAgent._client(
            "data_report_agent",
            model_config=config.data_report_agent,
        )
        # self.data_report_generator = self.load_data_report_generator()
        self.data_report_generator = None

        self.task_result = None

    def load_data_report_generator(self):
        if self.config.data_report_preset.lower() == "daco":
            return DACOReportGenerator(
                client=self.data_report_agent,
                work_dir=self._work_dir,
                question=self._question,
            )
        elif self.config.data_report_preset.lower() in (
                "insightbench", "insight-bench", "insight_bench"):
            return InsightBenchReportGenerator(
                client=self.data_report_agent,
                work_dir=self._work_dir,
                question=self._question,
            )
        else:
            return ArgosReportGenerator(
                client=self.data_report_agent,
                work_dir=self._work_dir,
                question=self._question,
                report_format=self.config.data_report_format,
                additional_requirements=self.config.data_report_requirements,
            )

    def get_task_message(self):
        return TextMessage(
            content=(
                f"Here is my primary objective: {self._question}\n"
                "Please start your plan."
            ),
            source="user",
        )

    async def run(self, print_to_console=False):
        task_message = self.get_task_message()
        if print_to_console:
            stream = self.brain_trust.run_stream(task=task_message)
            res = await Console(stream)
        else:
            res = await self.brain_trust.run(task=task_message)
        self.task_result = res
        return res

    async def run_stream(self):
        task_message = self.get_task_message()
        stream = self.brain_trust.run_stream(task=task_message)
        async for message in stream:
            if isinstance(message, TextMessage):
                yield text_message_to_dict(message)
            elif isinstance(message, MultiModalMessage):
                yield multimodal_message_to_dict(message)
            elif isinstance(message, TaskResult):
                yield {
                    "source": "system",
                    "usage": None,
                    "content": f"Stop reason: {message.stop_reason}",
                    "images": None,
                    "type": "TaskResult"
                }
                self.task_result = message
            else:
                raise NotImplementedError(
                    f"Message type {type(message)} is not supported.")

    async def save_task_result(self, filename="task_result.json"):
        if self.task_result is None:
            print("Task result is None. Running the task now.")
            task_result = await self.run()
            self.task_result = task_result

        dico = task_result_to_dict(self.task_result)
        filepath = os.path.join(self._work_dir, filename)
        with open(filepath, "w") as f:
            json.dump(dico, f, indent=4)
        return dico

    async def save_data_report(self,
                               raw_filename="report.md",
                               json_filename="report.json",
                               print_to_console=False):
        try:
            self.data_report_generator = self.load_data_report_generator()
            if print_to_console:
                print('---------- data_report ----------')
                print(self.data_report_generator.request_messages[0].content)
                print(self.data_report_generator.request_messages[1].content)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"File not found when loading data report generator: {e}"
                "You should probably run the task before saving the result."
            )
        await self.data_report_generator.generate_report(
            raw_filename=raw_filename,
            json_filename=json_filename
        )
