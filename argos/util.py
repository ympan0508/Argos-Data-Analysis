import math
import venv
from pathlib import Path
from typing import List

import pandas as pd
from autogen_agentchat.messages import MultiModalMessage, TextMessage


def get_dataset_summary(
    work_dir: str, dataset_names: List[str], additional_description: str = ""
):
    def remove_nan_values(data):
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                cleaned_subdict = remove_nan_values(value)
                cleaned_data[key] = cleaned_subdict
            elif not (isinstance(value, float) and math.isnan(value)):
                cleaned_data[key] = value

        return cleaned_data

    n = len(dataset_names)
    summary = ""

    if additional_description:
        summary += f"Description of dataset: {additional_description}\n\n"

    summary += (
        f"Meta information summary of {n} dataframe{'s' if n > 1 else ''} "
        "(generated by pd.describe(include='all'), including statistics: "
        "[count, unique, top (the most common value), freq (frequency of the "
        "most common value), mean, std, min, 25%, 50%, 75%, max] of each "
        "column, all the nan statsistics have been excluded):\n"
    )

    for i, dataset in enumerate(dataset_names, 1):
        assert dataset.endswith(".csv"), "only .csv files are supported"
        dataset_path = Path(f"{work_dir}/{dataset}")
        df = pd.read_csv(dataset_path)
        df_info = df.describe(include="all").round(3).to_dict()
        df_info = remove_nan_values(df_info)
        summary += f"{i}. Dataframe {dataset} summary: {df_info}\n"

    return summary


def prep_venv_context(venv_dir):
    venv_dir = Path(venv_dir)
    venv_builder = venv.EnvBuilder(with_pip=True)
    # venv_builder.create(venv_dir)  # venv should be manually created now
    venv_context = venv_builder.ensure_directories(venv_dir)

    return venv_context


def task_result_to_dict(task_result):
    messages = task_result.messages
    dico = []
    for message in messages:
        if isinstance(message, TextMessage):
            dico.append(text_message_to_dict(message))
        elif isinstance(message, MultiModalMessage):
            dico.append(multimodal_message_to_dict(message))
        else:
            raise NotImplementedError(
                f"Message type {type(message)} is not supported.")
    return dico


def text_message_to_dict(message):
    dico = {}
    dico["source"] = message.source
    dico["usage"] = None
    if message.models_usage is not None:
        dico["usage"] = {
            "prompt": message.models_usage.prompt_tokens,
            "completion": message.models_usage.completion_tokens,
        }
    dico["content"] = message.content
    dico["images"] = None
    dico["type"] = "TextMessage"
    return dico


def multimodal_message_to_dict(message):

    def process_multimodal_content(content):
        n = len(content)
        assert n % 2 == 1

        texts = [content[0]]
        images = []
        for i in range(1, n, 2):
            assert content[i].startswith(
                "The image of ") and content[i].endswith(": ")
            filename = content[i][len("The image of "): -2]
            images.append(filename)
            texts.append(f"<{filename}>")
        return "".join(texts), images

    dico = {}
    dico["source"] = message.source
    dico["usage"] = None
    if message.models_usage is not None:
        dico["usage"] = {
            "prompt": message.models_usage.prompt_tokens,
            "completion": message.models_usage.completion_tokens,
        }

    text_content, images = process_multimodal_content(message.content)
    dico["content"] = text_content
    dico["images"] = images
    dico["type"] = "MultiModalMessage"

    return dico
