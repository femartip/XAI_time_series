from openai import OpenAI
import argparse
from dotenv import load_dotenv
import os
import numpy as np
from Utils.load_data import load_dataset, load_dataset_labels, normalize_data
import matplotlib.pyplot as plt
import base64
import io
import re
from collections import Counter
from Utils.load_models import model_batch_classify
from simplifications import get_OS_simplification, get_RDP_simplification, get_bottom_up_simplification, \
    get_VW_simplification, get_LSF_simplification, get_bottom_up_lsf_simplification
import logging
from Utils.selectPrototypes import select_prototypes

#logging.basicConfig()
load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
DEBUG =False

def get_response(prompt: list[dict], client: OpenAI):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages= [
            {"role": "user", "content": prompt} #type: ignore
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

def build_prompt(images: list[str], test_samples: list[str], num_labels: int, print_prompt:bool=False) -> list[dict]:
    labels = range(0, num_labels)
    num_images = len(images)
    num_images_per_label = int(num_images/2)
    
    classes = ", ".join(map(str, labels))
    prompt = [
        {"type": "text", "text": f"You are a time-series classification expert. Your goal is to learn from a small set of labeled examples (classes {classes}) and then assign the correct class to a new, unlabeled time series. Follow these steps:\n1.Carefully examine the {classes} examples of classes {classes} and identify their common patterns.\n2.Compare the new instance to your learned characteristics of classes {classes}.\n3.Provide a brief rationale for your decision.\n4.Conclude with a single line stating only the predicted class (Predicted class: {classes}).\nThe examples:"}
    ]
    for label in labels:
        images_label = images[:num_images_per_label]
        images = images[num_images_per_label:]
        images_dict = [{"type": "text", "text": f"Class {label} examples ({num_images_per_label} time-series plots labeled “{label}”) "}] + \
                      [{"type": "image_url", "image_url": {"url": data_url}} for data_url in images_label]
        prompt.extend(images_dict) #type:ignore

    test = [{"type": "text", "text": "New instances to classify (unlabeled time-series plot)"}] + [{"type": "image_url", "image_url": {"url": data_url}} for data_url in test_samples] + [{"type": "text", "text": "Now, classify the new instances."}]
    prompt.extend(test) #type: ignore
    
    if INTERACTIVE:
        print("----------------------------------------------")
        for text in prompt:
            if text["type"] == "text":
                print(text["text"])
            elif text["type"] == "image_url":
                print("$IMAGE$")
        print("----------------------------------------------")
        input("Press Enter to continue...")
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

def ts_to_image(ts: np.ndarray, show_fig: bool = False, name: str = ""):
    plt.figure(figsize=(4,3))
    plt.plot(ts); plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf)
    if show_fig:
        plt.savefig(f"./llm_tests/{name}")
        plt.pause(1)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_b64}"


def get_and_test_examples(dataset_ts: np.ndarray, dataset_ts_labels: list[int], test_ts: np.ndarray, test_ts_label: list[int], labels: int) -> float:
    dataset_ts = dataset_ts
    dataset_ts_labels = dataset_ts_labels
    test_ts = test_ts
    test_ts_label = test_ts_label
    
    k_img = [ts_to_image(ts, show_fig=DEBUG, name=f"train_{i}") for i, ts in enumerate(dataset_ts)]
    test_sample = [ts_to_image(ts, show_fig=DEBUG, name=f"test_{i}") for i, ts in enumerate(test_ts)]

    prompt = build_prompt(k_img, test_samples= test_sample, num_labels=labels, print_prompt=DEBUG)

    client = OpenAI()
    response = get_response(prompt, client)
    
    pattern = r"Predicted class:\s+(\d+)"
    predicted_labels_str = re.findall(pattern, str(response))
    predicted_labels_int = [int(label) for label in predicted_labels_str]
    acc_length = min(len(test_ts_label), len(predicted_labels_int))
    accuracy = sum([1 if test_ts_label[i] == predicted_labels_int[i] else 0 for i in range(acc_length)])/acc_length
    
    if INTERACTIVE:
        print(f"------------------------------------------\n")
        print(response)
        print(f"Real label {test_ts_label}, predicted label {predicted_labels_int} = {accuracy}")
        print("\n------------------------------------------")
        input("Press Enter to continue...")
    print(f"Real label {test_ts_label}, predicted label {predicted_labels_int} = {accuracy}")
    return accuracy

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to feed samples from.")
    parser.add_argument('--classifier', type=str, default="cnn", help="Classifier to compare with." )
    parser.add_argument('--llm', type=str, default="",help="LLM within the OpenAI API.")
    parser.add_argument('--k', type=int, default=3, help="Number of total examples to use.")
    parser.add_argument('--interactive', action='store_true', help='Make code interactive')
    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()
    print(f"Testing {args.dataset } for classifier {args.classifier} on LLM {args.llm}")

    global INTERACTIVE
    INTERACTIVE = True if args.interactive else False
    steps = 10

    results = []
    results_simp = []
    for i in range(steps):
        print("Testing without simplifications:")
        dataset_ts_norm = select_prototypes(args.dataset, num_instances=args.k, data_type="TRAIN_normalized") 
        labels = np.array(load_dataset_labels(args.dataset, data_type='TEST_normalized'))
        
        test_ts_norm = load_dataset(args.dataset, data_type="TEST_normalized")
        test_ts_norm = test_ts_norm[np.random.randint(0, test_ts_norm.shape[0], size=(10))]
        
        classifier_file = f"{args.classifier}_norm.pth" if args.classifier == "cnn" else f"{args.classifier}_norm.pkl"
        dataset_ts_labels = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", dataset_ts_norm, len(set(labels)))   #type: ignore
        test_ts_label = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", test_ts_norm, len(set(labels)))  #type: ignore
        
        out = get_and_test_examples(dataset_ts_norm, dataset_ts_labels, test_ts_norm, test_ts_label, len(set(labels))) 
        results.append(out)

        print("Testing with simplifications:")
        dataset_ts_norm_simp = get_OS_simplification(dataset_ts_norm, alpha=0.2)
        dataset_ts_norm_simp = np.array([ts.line_version for ts in dataset_ts_norm_simp])

        test_ts_norm_simp = get_OS_simplification(test_ts_norm, alpha=0.2)
        test_ts_norm_simp = np.array([ts.line_version for ts in test_ts_norm_simp])
        
        dataset_ts_simp_labels = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", dataset_ts_norm_simp, len(set(labels)))     #type: ignore
        test_ts_simp_labels = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", test_ts_norm_simp, len(set(labels)))       #type: ignore
        
        out = get_and_test_examples(dataset_ts_norm_simp, dataset_ts_simp_labels, test_ts_norm_simp, test_ts_simp_labels, len(set(labels)))
        results_simp.append(out)
    
    print(results)
    print(results_simp)







