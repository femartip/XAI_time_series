from typing import Dict, List, Tuple

import PIL
import numpy as np
import io

from matplotlib import pyplot as plt
from reportlab.platypus import (Frame, FrameBreak,PageTemplate,BaseDocTemplate,PageBreak,Paragraph,Image, Spacer)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Frame, FrameBreak,PageTemplate,BaseDocTemplate,PageBreak,Paragraph, Table, TableStyle

from generate_user_survey.configurations import get_dataset_and_loyalty_from_config, get_config_of_group
from generate_user_survey.get_train_and_test_instances import get_train_and_test_instances

styles = getSampleStyleSheet()
global BODY_TEXT_STYLE
BODY_TEXT_STYLE = styles["BodyText"]
BODY_TEXT_STYLE.fontSize = 10

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


def build_pdf_one_dataset(prot_images: Dict[int,list[io.BytesIO]], test_images:  Dict[str,list[io.BytesIO]],config_for_student,group,studentnr) -> list[Flowable]:
    colour_names = ['pink', 'blue']
    
    prototype_pdf = []
    prototype_pdf.append(Paragraph(f"Configuration: {config_for_student}", BODY_TEXT_STYLE))
    for i,label in enumerate(prot_images.keys()):
        one_class_images = prot_images[label]
        colour = 'pink' if label == 0 else 'blue'
        #prototype_pdf.append(Paragraph(f"<font color='{colour}'>{colour}</font>", BODY_TEXT_STYLE))
        for img in one_class_images:
            buf_img = Image(img, width=3.5 * inch, height=1.3 * inch)
            prototype_pdf.append(buf_img)

            data = [[ClassSwatch(c, getattr(colors, "white")) if c != colour else ClassSwatch(c, getattr(colors, colour)) for c in colour_names]]
            table = Table(data, colWidths=[60]*len(colour_names))
            table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            prototype_pdf.append(table)

        prototype_pdf.append(FrameBreak())
        if i == 0:
            prototype_pdf.append(Paragraph(f"Group:{group} Student:{studentnr}", BODY_TEXT_STYLE))

        # images_pdf.append(PageBreak())
    test_pdf = []
    test_pdf.append(Paragraph(f"Configuration: {config_for_student}", BODY_TEXT_STYLE))

    #test_pdf.append(Paragraph("Which class do you think this is (pink or blue).", BODY_TEXT_STYLE))
    test_pdf.append(Paragraph("<b>Test Samples</b> - Assign the colour you think corresponds to each of the following samples.", BODY_TEXT_STYLE))

    for i,img in enumerate(test_images):
        if i == 5:
            test_pdf.append(FrameBreak())
            test_pdf.append(Paragraph(f"Group:{group} Student:{studentnr}", BODY_TEXT_STYLE))
            test_pdf.append(Spacer( 0, BODY_TEXT_STYLE.leading))


        buf_img = Image(img, width=3.5 * inch, height=1.3 * inch)
        test_pdf.append(buf_img)
        #test_pdf.append(Paragraph("Color: ", BODY_TEXT_STYLE))
        data = [[f"{i+1})"] + [ClassSwatch(c, getattr(colors, "white")) for c in colour_names]]
        table = Table(data, colWidths=[60]*(len(colour_names)+1))
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        test_pdf.append(table)

    test_pdf.append(PageBreak())
    full_pdf = prototype_pdf + test_pdf

    return full_pdf


def image_to_buf(ts: np.ndarray,y_lim:Tuple[float,float], class_num: str = -1,) -> io.BytesIO:
    plt.figure(figsize=(6, 3))
    colour_dict = {0: 'pink', 1: 'blue'}
    colour = colour_dict.get(class_num, 'gray')
    linestyle = '-' if colour != 'gray' else '--'

    plt.plot(ts, color=colour, linestyle=linestyle)
    plt.ylabel("Domain Specific Y Label")
    plt.grid()
    plt.ylim(y_lim)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf)
    plt.close()
    buf.seek(0)
    from PIL import Image
    return buf
def convert_to_images(proto_types:Dict[str,np.ndarray],test_instances:np.ndarray)->Tuple[Dict[str,List[io.BytesIO]],List[io.BytesIO]]:
    all_values_proto = np.concatenate([lst.flatten() for lst in proto_types.values()])
    all_values_test = test_instances.flatten()
    all_values = np.concatenate([all_values_proto, all_values_test])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    offset = ((max_val - min_val)/10)
    y_lim = (min_val-offset, max_val+offset)
    images_proto_types = {}
    for c in proto_types.keys():
        images_proto_types[c] = [image_to_buf(ts=ts,y_lim=y_lim,class_num=c) for ts in proto_types[c]]

    images_test= [image_to_buf(ts=ts,y_lim=y_lim) for ts in test_instances]
    return images_proto_types, images_test

def make_pdf_for_dataset_loyalty(dataset, loyalty_level,config_for_student, group, student_number):
    full_train_instances_simplified,full_test_instances_simplified =get_train_and_test_instances(dataset, loyalty_level)
    train_images, test_images = convert_to_images(full_train_instances_simplified,full_test_instances_simplified)
    return build_pdf_one_dataset(train_images,test_images,config_for_student,group,student_number)

def make_pdf_for_all_datasets(dataset_loyalty_pairs, group, student_number):
    pdf = []
    for i,(dataset, loyalty_level,config) in enumerate(dataset_loyalty_pairs):
        pdf.extend(make_pdf_for_dataset_loyalty(dataset, loyalty_level,config, group,student_number))

    doc = BaseDocTemplate(f"generate_user_survey/user_surveyes/{group}_{student_number}.pdf", pagesize=letter)
    ## -----------------  For two column layout -----------------
    gutter = 12  # pt spacing between columns
    col_width = (doc.width - gutter) / 2

    frames = [
        Frame(doc.leftMargin, doc.bottomMargin, col_width, doc.height, id="col1"),
        Frame(doc.leftMargin + col_width + gutter, doc.bottomMargin, col_width, doc.height, id="col2"),
    ]

    doc.addPageTemplates(PageTemplate(id="TwoCol", frames=frames))
    doc.build(pdf)

    return pdf



def make_pdf_for_current_student(group, student_number):
    config_for_student = get_config_of_group(group)
    datasets_and_loyalty_for_student = [(*get_dataset_and_loyalty_from_config(config),config) for config in config_for_student]
    print(datasets_and_loyalty_for_student)
    make_pdf_for_all_datasets(datasets_and_loyalty_for_student, group, student_number)
    print(group,student_number,datasets_and_loyalty_for_student)


def build_all_surveys_full_survey():
    num_of_student_in_each_group = 4
    for gi in range(1,5):
        group =  "G{}".format(gi)
        for si in range(1,1+num_of_student_in_each_group):
            student_number = "S{}".format(si)

            make_pdf_for_current_student(group, student_number)



if __name__ == "__main__":
    build_all_surveys_full_survey()
