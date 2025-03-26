import code
import re
from email import message
from io import BytesIO
from typing import List, Sequence

import PIL
import requests
from autogen_agentchat.agents._base_chat_agent import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, MultiModalMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core import Image as AGImage
from autogen_core.code_executor import CodeBlock, CodeExecutor


def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


"""
add `execute_last_message` regarding https://github.com/microsoft/autogen/issues/5283;

which can also be resolved by https://github.com/microsoft/autogen/pull/5259
"""


class MyCodeExecutorAgent(BaseChatAgent):
    def __init__(
        self,
        name: str,
        code_executor: CodeExecutor,
        *,
        description: str = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks).",
        execute_last_msg: bool = False,
    ) -> None:
        super().__init__(name=name, description=description)
        self._code_executor = code_executor
        self._execute_last_msg = execute_last_msg

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        """The types of messages that the code executor agent produces."""
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        # Extract code blocks from the messages.
        code_blocks: List[CodeBlock] = []
        if self._execute_last_msg:  # modified with issue:5283
            messages = messages[-1:]
        for msg in messages:
            if isinstance(msg, TextMessage):
                code_blocks.extend(extract_markdown_code_blocks(msg.content))
        if code_blocks:
            # Execute the code blocks.
            result = await self._code_executor.execute_code_blocks(
                code_blocks, cancellation_token=cancellation_token
            )

            code_output = result.output
            if code_output.strip() == "":
                # No output
                code_output = f"The script ran but produced no output to console. The POSIX exit code was: {result.exit_code}. If you were expecting output, consider revising the script to ensure content is printed to stdout."
            elif result.exit_code != 0:
                # Error
                code_output = f"The script ran, then exited with an error (POSIX exit code: {result.exit_code})\nIts output was:\n{result.output}"

            return Response(
                chat_message=TextMessage(content=code_output, source=self.name)
            )
        else:
            return Response(
                chat_message=TextMessage(
                    content="No code blocks found in the thread. Please provide at least one markdown-encoded code block to execute (i.e., quoting code in ```python or ```sh code blocks).",
                    source=self.name,
                )
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """It it's a no-op as the code executor agent has no mutable state."""
        pass


def get_autogen_agimage(image_path, remote=False):
    """
    Example usage:
    MultiModalMessage(
        content=[
            "Can you describe the content of this image?",
            get_autogen_agimage("https://picsum.photos/300/200", remote=True)
        ], source="User"
    )
    """
    if remote:
        response = requests.get(image_path)
        pil_image = PIL.Image.open(BytesIO(response.content))
    else:
        pil_image = PIL.Image.open(image_path)
    return AGImage(pil_image)


class MyCodeExecutorAgentWithMultiModalOutput(BaseChatAgent):

    def __init__(
        self,
        name: str,
        code_executor: CodeExecutor,
        working_dir: str,
        *,
        description: str = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks).",
        execute_last_msg: bool = False,
        # image_suffix: str = ".png",
    ) -> None:
        super().__init__(name=name, description=description)
        self._code_executor = code_executor
        self._execute_last_msg = execute_last_msg
        self._working_dir = working_dir
        # self._image_suffix = image_suffix

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        """The types of messages that the code executor agent produces."""
        return (TextMessage, MultiModalMessage)

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        # Extract code blocks from the messages.
        code_blocks: List[CodeBlock] = []
        if self._execute_last_msg:  # modified with issue:5283
            messages = messages[-1:]
        for msg in messages:
            if isinstance(msg, TextMessage):
                code_blocks.extend(extract_markdown_code_blocks(msg.content))
        if code_blocks:
            # Execute the code blocks.
            result = await self._code_executor.execute_code_blocks(
                code_blocks, cancellation_token=cancellation_token
            )

            code_output = result.output
            # if code_output.strip() == "":
            #     # No output
            #     code_output = f"The script ran but produced no output to console. The POSIX exit code was: {result.exit_code}. If you were expecting output, consider revising the script to ensure content is printed to stdout."
            if result.exit_code != 0:
                # Error
                code_output = f"The script ran, then exited with an error (POSIX exit code: {result.exit_code})\nIts output was:\n{result.output}"
                return Response(
                    chat_message=TextMessage(content=code_output, source=self.name)
                )

            # search for image files in the working directory generated by plt.savefig
            image_files = []
            for code_block in code_blocks:
                if code_block.language == "python":
                    # image_files.extend(re.findall(rf"\"(.+{self._image_suffix})\"", code_block.code))
                    pattern = r"plt\.savefig\s*\(\s*['\"]([^'\"]+)['\"]"
                    image_files.extend(re.findall(pattern, code_block.code))

            # create MultiModalMessage with text and images
            content = [code_output]
            if image_files:
                for image_file in image_files:
                    image_path = f"{self._working_dir}/{image_file}"
                    image = get_autogen_agimage(image_path)
                    content.append(f"The image of {image_file}: ")
                    content.append(image)

            return Response(
                chat_message=MultiModalMessage(content=content, source=self.name)
            )
        else:
            return Response(
                chat_message=TextMessage(
                    content="No code blocks found in the thread. Please provide at least one markdown-encoded code block to execute (i.e., quoting code in ```python or ```sh code blocks).",
                    source=self.name,
                )
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """It it's a no-op as the code executor agent has no mutable state."""
        pass
