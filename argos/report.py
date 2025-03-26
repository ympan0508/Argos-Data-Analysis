import json
import os

from autogen_core import CancellationToken
from autogen_core.models import SystemMessage, UserMessage

from .prompt import (
    DACO_ADDITIONAL_REQUIREMENTS,
    DACO_REPORT_FORMAT,
    FINALIZED_DATA_REPORT_PROMPT,
    INSIGHTBENCH_ADDITIONAL_REQUIREMENTS,
    INSIGHTBENCH_REPORT_FORMAT,
    WORK_LOG_TEMPLATE,
)


class ArgosReportGenerator:

    def _load_summarizer_messages(self):
        _summarizer_messages = []
        for _message in self.task_result:
            if _message['source'] == 'summarizing_agent':
                _summarizer_messages.append(_message)
        return _summarizer_messages

    def __init__(self, client, work_dir, question,
                 report_format, additional_requirements):
        _task_result_filename = os.path.join(work_dir, 'task_result.json')
        with open(_task_result_filename, 'r') as f:
            task_result = json.load(f)

        self.client = client
        self.task_result = task_result
        self.work_dir = work_dir
        self.summarizer_messages = self._load_summarizer_messages()

        _sys_prompt = FINALIZED_DATA_REPORT_PROMPT.format(
            question=question,
            additional_requirements=additional_requirements,
            report_format=report_format
        )
        _work_log_fillin = '\n'.join(
            f"## Work Log for Action-{i}\n{e['content']}\n"
            for i, e in enumerate(self.summarizer_messages, 1)
        )
        _work_log = WORK_LOG_TEMPLATE.format(
            work_logs=_work_log_fillin)

        self.request_messages = [
            SystemMessage(content=_sys_prompt),
            UserMessage(content=_work_log, source='brain_trust')
        ]

    async def generate_report(self, raw_filename="report.md",
                              json_filename="report.json"):
        response = await self.client.create(
            messages=self.request_messages,
            cancellation_token=CancellationToken()
        )
        response_dico = {
            "content": response.content,
            "usage": {"prompt": response.usage.prompt_tokens,
                      "completion": response.usage.completion_tokens},
            "finish_reason": response.finish_reason
        }
        with open(os.path.join(self.work_dir, json_filename), 'w') as f:
            json.dump(response_dico, f, indent=4)
        with open(os.path.join(self.work_dir, raw_filename), 'w') as f:
            f.write(response.content)


class DACOReportGenerator(ArgosReportGenerator):

    def __init__(self, client, work_dir, question):
        super().__init__(
            client=client,
            work_dir=work_dir,
            question=question,
            report_format=DACO_REPORT_FORMAT,
            additional_requirements=DACO_ADDITIONAL_REQUIREMENTS
        )


class InsightBenchReportGenerator(ArgosReportGenerator):

    def __init__(self, client, work_dir, question):
        super().__init__(
            client=client,
            work_dir=work_dir,
            question=question,
            report_format=INSIGHTBENCH_REPORT_FORMAT,
            additional_requirements=INSIGHTBENCH_ADDITIONAL_REQUIREMENTS
        )
