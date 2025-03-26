import re
import time

import evaluate
import numpy as np
import openai
from openai import OpenAI


def compute_rouge_score(answer, gt_answer, **kwargs):
    """Compute ROUGE-1 between answer and gt_answer"""
    ROUGE_SCORE = evaluate.load("rouge")

    return ROUGE_SCORE.compute(
        predictions=[answer],
        references=[gt_answer],
        rouge_types=["rouge1"],
    )["rouge1"]


def compute_g_eval(answer, gt_answer, model_name="gpt-4o-2024-08-06", top_logprobs=5):
    client = OpenAI()
    G_EVAL_BASIC_SYSTEM_MESSAGE = """You are a high school teacher evaluating student responses to a question. You are tasked with grading the response based on how well it answers the question. You are to provide a numerical rating for how well the response answers the question based on the ground truth answer."""
    G_EVAL_BASIC_TEMPLATE = f"""
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        Provided Answer:
        {answer}

        Ground Truth Answer:
        {gt_answer}

        Follow these instructions when writing your response:
        * On a scale of 1-10, provide a numerical rating for how close the provided answer is to the ground truth answer, with 10 denoting that the provided answer is the same as ground truth answer.
        * Your response should contain only the numerical rating. DONOT include anything else like the provided answer, the ground truth answer, or an explanation of your rating scale in your response.
        * Wrap your numerical rating inside <rating></rating> tags.
        * Check very carefully before answering.
        * Follow the output format as shown in the example below:
        Example response:
        <rating>7</rating>

        ### Response:
    """
    template, system_message = G_EVAL_BASIC_TEMPLATE, G_EVAL_BASIC_SYSTEM_MESSAGE

    prompt = template.format(answer=answer, gt_answer=gt_answer)
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=50,
                top_p=1,
                logprobs=bool(top_logprobs),
                top_logprobs=top_logprobs,
            )
            if not top_logprobs:
                score = response.choices[0].message.content
            else:
                # get the index in response where we have the rating
                rating_str = re.findall(
                    r"<rating>(\d+)</rating>", response.choices[0].message.content
                )[0]
                tokens = [o.token for o in response.choices[0].logprobs.content]
                rating_idx_in_response = tokens.index(rating_str)
                response = (
                    response.choices[0]
                    .logprobs.content[rating_idx_in_response]
                    .top_logprobs
                )
                # convert logprobs to probs
                probs = [np.exp(obj.logprob) for obj in response]
                # renormalize probs to sum to 1
                probs = [obj / sum(probs) for obj in probs]
                ratings = [
                    float(obj.token) if obj.token.isdigit() else 0 for obj in response
                ]
                # final score
                score = sum([a * b for a, b in zip(ratings, probs)])
            try:
                score = float(score)
            except ValueError:
                score = float(score.splitlines()[0])
            except:
                score = 0
            return score
        except openai.RateLimitError as e:
            print("RateLimitError, Sleeping for 100 seconds...")
            time.sleep(100)
        except openai.APIError as e:
            print(f"APIError, {e}\nSleeping for 100 seconds...")
            time.sleep(100)
        except Exception as e:
            print(f"{e}, Sleeping for 100 seconds...")


def compute_g_eval_with_cot(answer, gt_answer, goal, model_name="gpt-4o-2024-08-06", top_logprobs=10):
    client = OpenAI()

    system_message = "You are a data scientist assessing a data analysis team. Given the analysis target, the expected ground truth, and the analysis report submitted by the team, you should provide a numerical rating for the quality of the report."
    prompt = f"""
        Please rate the quality of the analysis report based on the analysis target and ground truth.
        ### Instruction:
        Goal:
        {goal}

        Provided Answer:
        {answer}

        Ground Truth Answer:
        {gt_answer}

        * On a scale of 0-9, provide a numerical rating for the quality of the provided analysis report based on the ground truth. A score of 9 means the provided report perfectly covers all key aspects of the ground truth without introducing major errors. Additional correct information that contributes to the analysis target and does not contradict the ground truth should also be considered valuable. However, irrelevant, misleading, or incorrect additions should lower the score.
        * Your response should include three parts: 1) why-not-higher, 2) why-not-lower, and 3) final-score. You should not provide the final score until you have provided the why-not-higher and why-not-lower parts.
        * Wrap your numerical rating inside <rating></rating> tags.
        * Check very carefully before answering.
        * Follow the output format as shown in the example below:
        Example response:
        <rating>7</rating>

        ### Response:
    """
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=1000,
                top_p=1,
                logprobs=bool(top_logprobs),
                top_logprobs=top_logprobs,
            )
            if not top_logprobs:
                score = response.choices[0].message.content
            else:
                # get the index in response where we have the rating
                rating_str = re.findall(
                    r"<rating>(\d+)</rating>", response.choices[0].message.content
                )[0]
                tokens = [o.token for o in response.choices[0].logprobs.content]
                rating_idx_in_response = tokens.index(rating_str)
                response = (
                    response.choices[0]
                    .logprobs.content[rating_idx_in_response]
                    .top_logprobs
                )
                # convert logprobs to probs
                probs = [np.exp(obj.logprob) for obj in response]
                # renormalize probs to sum to 1
                probs = [obj / sum(probs) for obj in probs]
                ratings = [
                    float(obj.token) if obj.token.isdigit() else 0 for obj in response
                ]
                # final score
                score = sum([a * b for a, b in zip(ratings, probs)])
            try:
                score = float(score)
            except ValueError:
                score = float(score.splitlines()[0])
            except:
                score = 0
            return score
        except openai.RateLimitError as e:
            print("RateLimitError, Sleeping for 100 seconds...")
            import time
            time.sleep(100)
        except openai.APIError as e:
            print(f"APIError, {e}\nSleeping for 100 seconds...")
            import time
            time.sleep(100)
        except Exception as e:
            print(f"{e}, Sleeping for 100 seconds...")
