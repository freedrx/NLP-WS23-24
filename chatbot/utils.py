def extract_substring(text: str, start_marker: str, end_marker: str):
    start_index = text.find(start_marker)
    if start_index != -1:
        text = text[start_index + len(start_marker):]
    
    end_index = text.find(end_marker)
    if end_index != -1:
        text = text[:end_index]
    return text