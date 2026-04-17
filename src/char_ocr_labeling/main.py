from ocr_module.labeling import Labeler

def run_ocr_pipeline(segments):

    labeler = Labeler()

    labeled = labeler.label_segments(segments)

    grouped = labeler.group_by_label(labeled)

    return grouped