import argparse
from random import shuffle
from matplotlib.table import Table
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
from reportlab.lib import colors
from reportlab.platypus import Frame, FrameBreak,PageTemplate,BaseDocTemplate,PageBreak,Paragraph, Table, TableStyle
from reportlab.platypus.flowables import Flowable
from reportlab.platypus import Image


DEBUG =False

class ClassSwatch(Flowable):
    def __init__(self, class_name: str, fill_color: colors.Color, box=12, gap=4):
        super().__init__()
        self.class_name = class_name
        self.fill_color = fill_color
        self.box = box
        self.gap = gap
        self.height = box
        self.width = box + gap + 45

    def draw(self):
        self.canv.setFillColor(self.fill_color)
        self.canv.rect(0, 0, self.box, self.box, fill=1, stroke=1)
        self.canv.setFont("Helvetica", 10)
        self.canv.setFillColor(colors.black)
        self.canv.drawString(self.box + self.gap, 2, self.class_name.capitalize())


def build_pdf(prot_images: list[io.BytesIO], test_samples: list[io.BytesIO], test_ts_simp_labels: list[int], num_labels: int) -> list[Flowable]:
    labels = range(0, num_labels)
    num_images = len(prot_images)
    num_images_per_label = int(num_images/num_labels)
    class_names = ("pink", "blue", "green", "orange", "purple")[:num_labels]

    IMG_WIDTH = 2.9 * inch
    IMG_HEIGHT = 1.4 * inch

    images_pdf = [] 
    images_pdf.append(Paragraph(f"Configuration: {CONFIG}", BODY_TEXT_STYLE)) 
    for label in labels:
        images_label = prot_images[:num_images_per_label]
        prot_images = prot_images[num_images_per_label:]
        colour = class_names[label]
        #images_pdf.append(Paragraph(f"<b>{colour} class</b>", BODY_TEXT_STYLE))
        for img in images_label:
            buf_img = Image(img, width=IMG_WIDTH, height=IMG_HEIGHT)
            images_pdf.append(buf_img)

            data = [[ClassSwatch(c, getattr(colors, "white")) if c != colour else ClassSwatch(c, getattr(colors, colour)) for c in class_names]]
            table = Table(data, colWidths=[60]*len(class_names))
            table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            images_pdf.append(table)
       
        if num_labels == 2 or label == num_labels -1:
            images_pdf.append(FrameBreak())
        #images_pdf.append(PageBreak())

    images_pdf.append(Paragraph(f"Configuration: {CONFIG}", BODY_TEXT_STYLE))
    
    images_pdf.append(Paragraph(f"<b>Test Samples</b> - Assign the colour you think corresponds to each of the following samples.", BODY_TEXT_STYLE))
    #images_pdf.append(Paragraph("Assign the colour you think corresponds to each of the following samples.", BODY_TEXT_STYLE))

    for j, img in enumerate(test_samples):
        buf_img = Image(img, width=IMG_WIDTH, height=IMG_HEIGHT)
        images_pdf.append(buf_img)
        #images_pdf.append(Paragraph("Class: ", BODY_TEXT_STYLE))

        data = [[f"{j+1})"] + [ClassSwatch(c, getattr(colors, "white")) for c in class_names]]
        table = Table(data, colWidths=[60]*(len(class_names)+1))
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        images_pdf.append(table)
       
    images_pdf.append(Paragraph("Participant:", BODY_TEXT_STYLE))
    images_pdf.append(PageBreak())
    images_pdf.append(Paragraph(f"Results for {CONFIG}:", BODY_TEXT_STYLE))
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
    parser.add_argument('--init_letter', type=str, default="A", help="Initial letter of the naming of the configurations.")
    parser.add_argument('--dataset', type=str, default="Chinatown", help="Dataset to feed samples from.")
    parser.add_argument('--classifier', type=str, default="cnn", help="Classifier to compare with." )
    parser.add_argument('--k', type=int, default=3, help="Number of total examples to use.")
    parser.add_argument('--method', type=str, default="OS", help="Get results selected explainability methods. Can be: OS, RDP, SAX")
    parser.add_argument('--loyaltys', type=float, nargs='+', default=[0.6, 0.85, 0.96, 1.00], help="If a simplification method selected. Set values of loyalty to iterate over as a list of floats.")
    ## If loyalty is 1.0 == No simplification applied
    parser.add_argument('--interactive', action='store_true', help='Make code interactive')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = argparser()
    print(f"Generating PDF for {args.dataset } using {args.classifier}")

    global INTERACTIVE
    INTERACTIVE = True if args.interactive else False
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    configs = [c for c in alphabet if c >= args.init_letter]
    steps = 5


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

    for i, loyalty in enumerate(args.loyaltys):
        global CONFIG
        CONFIG = configs[i]
        alpha = get_alpha_by_loyalty(args.dataset, args.classifier, loyalty, args.method)
        print(f"Alpha value for {args.method} with loyalty {loyalty} is {alpha}")

        doc = BaseDocTemplate(f"./human_study/{args.dataset}_{args.method}_{loyalty}.pdf", pagesize=letter)
    
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
        BODY_TEXT_STYLE.fontSize = 9

        if loyalty == 1.0:
            prot_img_simp, test_img_simp = ts_to_image(prototipes_ts_norm, args.k, test_ts_norm, len(set(labels)))
            test_img_idx = list(range(len(test_img_simp)))
            shuffle(test_img_idx)
            test_img_simp = [test_img_simp[i] for i in test_img_idx]
            test_ts_label = [test_ts_label[i] for i in test_img_idx]
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
            test_img_idx = list(range(len(test_img_simp)))
            shuffle(test_img_idx)
            test_img_simp = [test_img_simp[i] for i in test_img_idx]
            test_ts_simp_labels = [test_ts_simp_labels[i] for i in test_img_idx]
            out = build_pdf(prot_img_simp, test_img_simp, test_ts_simp_labels, len(set(labels)))
        
        doc.build(out)

        





