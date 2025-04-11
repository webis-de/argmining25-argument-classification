from pathlib import Path

def load_prompt_files(references):
    final_prompts = []
    if isinstance(references,str):
        references = [references]
    for reference in references:
        if not reference.endswith(".txt"):
            reference = reference + ".txt"
        reference = Path(reference)
        if not reference.is_absolute():
            reference = Path(__file__).parent / reference
        with open(reference, 'r') as file:
            final_prompts.append(file.read())
    final_prompts = "\n\n".join(final_prompts)
    return final_prompts