from openai import OpenAI
import argparse
from dotenv import load_dotenv
import os
import numpy as np
from load_data import load_dataset, load_dataset_labels, normalize_data
import matplotlib.pyplot as plt
import base64
import io

load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

def get_response(prompt: list[dict], client: OpenAI):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages= [
            {"role": "user", "content": prompt} #type: ignore
        ]
    )
    return response.choices[0].message.content

def build_prompt(images, k_idx) -> list[dict]:
    prompt = [
        {"type": "text", "text": "Given the following time series with its corresponding class.."}
    ]
    images_dict = [{"type": "image_url", "image_url": {"url": data_url}} for data_url in images]
    prompt.extend(images_dict)
    return prompt


def get_idx_per_cls(labels_ts: np.ndarray, k_cls:int) -> dict[str,list[int]]:
    labels = np.unique(labels_ts)
    idx_labels = {}
    for label in labels:
        labels_idx = np.where(labels_ts == label)[0]
        rand_idx = np.asanyarray(np.random.randint(labels_idx.shape[0], size=(k_cls)))
        idx_labels[label] = rand_idx

    return idx_labels

def get_k_examples(dataset_ts: np.ndarray, k_idx:dict) -> np.ndarray:
    labels = k_idx.keys()
    k_examples = []
    for label in labels:
        idx = k_idx[label]
        k_examples_label = dataset_ts[idx]
        k_examples.append(k_examples_label)

    return np.array(k_examples)

def ts_to_image(ts: np.ndarray, show_fig: bool = False):
    plt.figure(figsize=(4,3))
    plt.plot(ts); plt.title("My Time Series"); plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    if show_fig:
        plt.show(block=False)
        plt.pause(1)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_b64}"

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to feed samples from.")
    parser.add_argument('--classifier', type=str, default="cnn", help="Classifier to compare with." )
    parser.add_argument('--llm', type=str, default="",help="LLM within the OpenAI API.")
    parser.add_argument('--k', type=int, default=10, help="Number of examples to use.")
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    print(f"Testing {args.dataset} for classifier {args.classifier} on LLM {args.llm}")

    dataset_ts = load_dataset(args.dataset, data_type='TEST')
    dataset_ts_norm = normalize_data(args.dataset, data_type='TEST')
    dataset_ts_labels = load_dataset_labels(args.dataset, data_type='TEST')
    
    assert len(np.unique(dataset_ts_labels)) < args.k, "More labels than examples, use a bigger number of k."
    k_cls = int(args.k / len(np.unique(dataset_ts_labels)))
    k_idx = get_idx_per_cls(dataset_ts_labels, k_cls)
    k_examples = get_k_examples(dataset_ts, k_idx)
    
    k_img = [ts_to_image(ts, show_fig=False) for ts_list in k_examples for ts in ts_list]
    
    prompt = build_prompt(k_img, k_idx)
    #print(prompt)

    client = OpenAI()
    response = get_response(prompt, client)
    print(response)





