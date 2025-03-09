

def create_messages(questions, instructions):
    if isinstance(questions, str):
        questions = list([questions])
    messages = []
    for q in questions:
        messages.append({"role":"user", "content":q })
    if instructions is not None:
        messages.append(instructions)
    return messages