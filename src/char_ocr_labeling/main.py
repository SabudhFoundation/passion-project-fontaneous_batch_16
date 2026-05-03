from .labeling import Labeler

def run_ocr_pipeline(char_crops):
    """
    Accepts list of character images (np.ndarray)
    Returns labeled data grouped for UI selection
    """

    labeler = Labeler()

    labeled = labeler.label_segments(char_crops)

    # group by label
    groups = {}
    for item in labeled:
        groups.setdefault(item["label"], []).append(item)

    # sort by score
    for label in groups:
        groups[label].sort(key=lambda x: x["score"], reverse=True)

    return {
        "labeled": labeled,
        "groups": groups
    }