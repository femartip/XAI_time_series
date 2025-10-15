import argparse
import numpy as np
from Utils.load_data import load_dataset, load_dataset_labels, normalize_data
import matplotlib.pyplot as plt
import io
from Utils.load_models import model_batch_classify
from Utils.metrics import get_alpha_by_loyalty
from simplifications import get_OS_simplification, get_RDP_simplification 
import logging
from Utils.selectPrototypes import select_prototypes
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Frame, FrameBreak,PageTemplate,BaseDocTemplate,PageBreak,Paragraph)
from reportlab.platypus.flowables import Flowable
from reportlab.platypus import Image

DEBUG =False

def build_pdf(prot_images: list[io.BytesIO], test_samples: list[io.BytesIO], test_ts_simp_labels: list[int], num_labels: int) -> list[Flowable]:
    labels = range(0, num_labels)
    num_images = len(prot_images)
    num_images_per_label = int(num_images/num_labels)
    
    images_pdf = [] 
    for label in labels:
        images_label = prot_images[:num_images_per_label]
        prot_images = prot_images[num_images_per_label:]
        colour = 'pink' if label == 0 else 'blue' if label == 1 else 'green'
        images_pdf.append(Paragraph(f"<b>{colour} class</b>", BODY_TEXT_STYLE))
        for img in images_label:
            buf_img = Image(img, width=3.5 * inch, height=2 * inch)
            images_pdf.append(buf_img)
        images_pdf.append(FrameBreak())
        #images_pdf.append(PageBreak())
        
    images_pdf.append(Paragraph(f"<b>Test Samples</b>", BODY_TEXT_STYLE))
    images_pdf.append(Paragraph("Assign the class (pink or blue) to each of the following test samples.", BODY_TEXT_STYLE))

    for img in test_samples:
        buf_img = Image(img, width=3 * inch, height=1.5 * inch)
        images_pdf.append(buf_img)
        images_pdf.append(Paragraph("Class: ", BODY_TEXT_STYLE))

    images_pdf.append(PageBreak())
    images_pdf.append(Paragraph("Results", BODY_TEXT_STYLE))
    labels = [f"Image {i+1}: class {label}" for i, label in enumerate(test_ts_simp_labels)]
    images_pdf.append(Paragraph(", ".join(labels), BODY_TEXT_STYLE))

    if INTERACTIVE:
        print("----------------------------------------------")
        print(images_pdf)
        print("----------------------------------------------")
        input("Press Enter to continue...")
    return images_pdf


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

def image_to_buf(ts: np.ndarray, class_num: int = -1) -> io.BytesIO:
    plt.figure(figsize=(6, 3))
    colour_dict = {0: 'pink', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple', -1: 'black'}
    linestyle_dict = {0: '-', 1: '-', 2: '-', 3: '-', 4: '-', -1: '--'}
    colour = colour_dict.get(class_num, 'gray')
    linestyle = linestyle_dict.get(class_num, '-')

    plt.plot(ts, color=colour, linestyle=linestyle); plt.ylabel("Domain Specific Y Label"); plt.grid(); plt.ylim(0, 1); plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf)
    plt.close()
    buf.seek(0)
    return buf


def ts_to_image(dataset_ts: np.ndarray, k: int, test_ts: np.ndarray, num_labels: int) -> tuple[list[io.BytesIO], list[io.BytesIO]]:
    dataset_ts = dataset_ts
    labels = [label for label in range(num_labels) for _ in range(k)]
    
    dataset_with_labels = zip(dataset_ts, labels)
    test_ts = test_ts
    k_img = [image_to_buf(ts, class_num=ts_y) for ts, ts_y in dataset_with_labels]
    test_sample = [image_to_buf(ts) for i, ts in enumerate(test_ts)]

    return k_img, test_sample

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Chinatown", help="Dataset to feed samples from.")
    parser.add_argument('--classifier', type=str, default="cnn", help="Classifier to compare with." )
    parser.add_argument('--k', type=int, default=3, help="Number of total examples to use.")
    parser.add_argument('--method', type=str, default="OS", help="Get results selected explainability methods. Can be: OS, RDP, SAX")
    parser.add_argument('--loyalty', type=float, default=0.8, help="If a simplification method selected. Set values of loyalty to iterate over.")
    parser.add_argument('--interactive', action='store_true', help='Make code interactive')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = argparser()
    print(f"Generating PDF for {args.dataset } using {args.classifier}")

    global INTERACTIVE
    INTERACTIVE = True if args.interactive else False
    steps = 5

    alpha = get_alpha_by_loyalty(args.dataset, args.classifier, args.loyalty, args.method)
    print(f"Alpha value for {args.method} with loyalty {args.loyalty} is {alpha}")
        
    train_ts_norm = load_dataset(args.dataset, data_type="TRAIN_normalized")
    
    prototipes_ts_norm = select_prototypes(args.dataset, num_instances=args.k, data_type="TRAIN_normalized") 
    labels = np.array(load_dataset_labels(args.dataset, data_type='TEST_normalized'))

    test_ts_norm = load_dataset(args.dataset, data_type="TEST_normalized")
    rand_ts_idx = np.random.randint(0, test_ts_norm.shape[0], size=(10))
    test_ts_norm = test_ts_norm[rand_ts_idx]
    
    classifier_file = f"{args.classifier}_norm.pth" if args.classifier == "cnn" else f"{args.classifier}_norm.pkl"
    dataset_ts_labels = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", prototipes_ts_norm, len(set(labels)))   #type: ignore
    test_ts_label = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", test_ts_norm, len(set(labels)))  #type: ignore
        
    simp_methods = {}
    if "OS" in args.method:
        simp_methods = get_OS_simplification
    if "RDP" in args.method:
        simp_methods = get_RDP_simplification

    doc = BaseDocTemplate(f"./human_study/{args.dataset}_{args.method}_{args.loyalty}.pdf", pagesize=letter)
    
    ## -----------------  For two column layout ----------------- 
    gutter = 12  # pt spacing between columns
    col_width = (doc.width - gutter) / 2

    frames = [
        Frame(doc.leftMargin, doc.bottomMargin, col_width, doc.height, id="col1"),
        Frame(doc.leftMargin + col_width + gutter, doc.bottomMargin, col_width, doc.height, id="col2"),
    ]

    doc.addPageTemplates(PageTemplate(id="TwoCol", frames=frames))

    ## ------------------- For single column layout -------------------
    #frame = Frame(1 * inch, 1 * inch, 6.8 * inch, 9 * inch, id="normal")
    #template = PageTemplate(id="test", frames=frame)
    #doc.addPageTemplates([template])

    styles = getSampleStyleSheet()
    global BODY_TEXT_STYLE
    BODY_TEXT_STYLE = styles["BodyText"]
    BODY_TEXT_STYLE.fontSize = 12

    if args.loyalty == 1.0:
        prot_img_simp, test_img_simp = ts_to_image(prototipes_ts_norm, args.k, test_ts_norm, len(set(labels)))
        out = build_pdf(prot_img_simp, test_img_simp, test_ts_label, len(set(labels)))
    else:
        prototipes_ts_norm_simp = simp_methods(prototipes_ts_norm, alpha)           #type: ignore
        prototipes_ts_norm_simp = np.array([ts.line_version for ts in prototipes_ts_norm_simp])
        test_ts_norm_simp = simp_methods(test_ts_norm, alpha)           #type: ignore
        num_segments = np.mean([(len(ts.x_pivots) - 1) for ts in test_ts_norm_simp])
        test_ts_norm_simp = np.array([ts.line_version for ts in test_ts_norm_simp])
        dataset_ts_simp_labels = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", prototipes_ts_norm_simp, len(set(labels)))     #type: ignore
        test_ts_simp_labels = test_ts_label     # Labels compares with original lables given by model
        
        prot_img_simp, test_img_simp = ts_to_image(prototipes_ts_norm_simp, args.k, test_ts_norm_simp, len(set(labels)))        #Even tough we use the simplified version, we want the labels from the original ts
        out = build_pdf(prot_img_simp, test_img_simp, test_ts_simp_labels, len(set(labels)))
    
    doc.build(out)

    





