ISI_SEVERITY = {
    (0, 7): "None",
    (8, 14): "Mild",
    (15, 21): "Moderate",
    (22, 28): "Severe",
}

PHQ9_SEVERITY = {
    (0, 4): "None",
    (5, 9): "Mild",
    (10, 14): "Moderate",
    (15, 19): "Moderately Severe",
    (20, 27): "Severe",
}

def get_severity(score: int, mapping: dict):
    for (low, high), label in mapping.items():
        if low <= score <= high:
            return label
    return "Unknown"
