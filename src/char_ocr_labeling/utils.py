def get_label_folder(label):

    if label.isalpha():
        return f"capital_{label.lower()}" if label.isupper() else f"small_{label}"

    if label.isdigit():
        return label

    return f"other_{label}"