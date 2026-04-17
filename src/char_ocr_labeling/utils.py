def get_label_folder(label):

    if label.isalpha():
        if label.isupper():
            return f"capital_{label.lower()}"
        else:
            return f"small_{label}"

    if label.isdigit():
        return label

    return f"other_{label}"