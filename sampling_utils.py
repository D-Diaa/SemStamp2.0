
hash_key = 15485863
PUNCTS = '!.?'



def extract_prompt_from_text(text, len_prompt):
    tokens = text.split(' ')
    tokens = tokens[:len_prompt]
    new_text = ' '.join(tokens)
    prompts = []
    for p in PUNCTS:
        idx = new_text.find(p)
        if idx != -1:
            tokens = new_text[:idx + 1].split(" ")
            # has to be greater than a minimum prompt
            if len(tokens) > 3:
                prompts.append(new_text[:idx + 1])
    if len(prompts) == 0:
        prompts.append(new_text + ".")
    # select first (sub)sentence, deliminated by any of PUNCTS
    prompt = list(sorted(prompts, key=lambda x: len(x)))[0]
    return prompt



