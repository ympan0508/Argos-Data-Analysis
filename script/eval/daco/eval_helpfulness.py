# Modified from:
# https://github.com/shirley-wu/daco/blob/main/evaluation/eval_helpfulness.py

import argparse
import json
import string
import time

import anthropic
import openai
import tqdm

if True:
    import os
    os.environ['http_proxy'] = ""
    os.environ['https_proxy'] = ""


def make_request(query, args):
    params = {"max_tokens": 400, "model": args.model, "temperature": 1.0, }

    if args.model_type == 'openai':
        client_func = openai.OpenAI().chat.completions.create
        messages = [{'role': 'user', 'content': query}, ]
    elif args.model_type == 'vllm':
        client = openai.OpenAI(
            base_url=args.vllm_base_url,
            api_key=args.api_key,
        )
        client_func = client.chat.completions.create
        messages = [{'role': 'user', 'content': query}, ]
    else:
        assert args.model_type == 'anthropic'  # e.g. claude-3-5-sonnet
        client_func = anthropic.Anthropic(api_key=args.api_key).messages.create
        messages = [{"role": "user", "content": [
            {"type": "text", "text": query}]}, ]

    for _ in range(5):
        try:
            completion = client_func(messages=messages, **params)
            if args.model_type in ['openai', 'vllm', ]:
                return completion.choices[0].message.content
            else:
                return completion.content[0].text
        except Exception as e:
            print("Error")
            print(e)
            time.sleep(3)

    # return None
    raise RuntimeError("Failed to make request, please check the environment")


QUERY_PROMPT = """{INTENTION_DESC}

I have hired two data analysts to perform the analysis, and they gave me two different reports (listed below). Each report consists of two lists, one for findings and one for suggestions. Which one is more helpful to my analysis? When evaluating helpfulness, you should consider the following three rubrics in decreasing priority: (1) relevance to my analysis goal; (2) insightfulness; and (3) diversity of perspectives, especially for suggestions.

Your response should be in the following format. Note: <answer> should be either Report-1 or Report-2
* Answer: <answer>
* Reasoning: <explain your reasoning here>

The reports are as follows:

# Report-1

{REPORT_1}

# Report-2

{REPORT_2}"""  # noqa: E501


def parse_response(text):
    try:
        assert 'Answer:' in text
        text = text.split('Answer:')[1].strip().split()[0]
        text = ''.join(
            [
                t for t in text
                if t in string.ascii_letters + string.digits + '-'
            ]
        ).lower()
        if text == 'report-1':
            return 0
        elif text == 'report-2':
            return 1
        else:
            return None
    except Exception:
        return None


def compare_final_report(gen1, gen2, args):
    intention_desc = gen1['messages'][0]['content'].splitlines()[0][1:].strip()
    report_1 = gen1['messages'][-1]['content'].strip()
    report_2 = gen2['messages'][-1]['content'].strip()

    query = QUERY_PROMPT.format(
        INTENTION_DESC=intention_desc, REPORT_1=report_1, REPORT_2=report_2)
    response = make_request(query, args)

    return parse_response(response)


def convert_format(data):
    result = []
    for d in data:
        d["data_id"] = d["id"][:-3] + "||" + d["id"][-1]
        d["table_id"] = d["data_id"][:-3]
        messages = []
        messages.append({"role": "assistant", "content": d["report"]})
        d["messages"] = messages
        result.append(d)

    return result


def main(args):
    with open(args.comparison) as f:
        comparison = json.load(f)

    with open(args.pred) as f:
        pred = json.load(f)
        pred = convert_format(pred)

    pred = {item['data_id']: item for item in pred}
    comparison = [
        item for item in comparison if item["data_id"] in pred.keys()]
    pred = [pred[item['data_id']] for item in comparison]

    def is_valid(gen):
        if "success" in gen.keys():
            return gen["success"]
        else:
            return True

    # final report
    # 0: first win; 1: second win; None: invalid answer
    counter = {0: 0, 1: 0, None: 0, }
    result = []

    for i in tqdm.trange(len(comparison)):
        if is_valid(pred[i]) and is_valid(comparison[i]):

            ans = compare_final_report(pred[i], comparison[i], args)
            counter[ans] += 1
            ans1 = ans

            ans = compare_final_report(comparison[i], pred[i], args)
            if ans is not None:
                ans = 1 - ans
            counter[ans] += 1
            ans2 = ans
            result.append([ans1, ans2])
        else:
            result.append([None, None])
            continue

    print("Win rate of {} against {}: {} wins {} loses ({}%)".format(
        args.pred, args.comparison, counter[0], counter[1], counter[0] / (
            counter[0] + counter[1]) * 100,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='example.json')
    parser.add_argument(
        '--comparison', default='test_h.json')
    parser.add_argument('--model_type', default='openai')
    parser.add_argument('--model', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--api_key', default='EMPTY')
    parser.add_argument('--vllm_base_url', default='http://localhost:8000/v1')

    args = parser.parse_args()

    main(args)
