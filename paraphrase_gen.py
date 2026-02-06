from itertools import groupby
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from nltk import sent_tokenize
import os
import torch
import openai
import pickle
from transformers import AutoTokenizer
from paraphrase_gen_utils import accept_by_bigram_overlap, SParrot, query_openai, query_openai_bigram, gen_prompt, gen_bigram_prompt, extract_list
from nltk.tokenize import sent_tokenize
from string import punctuation
from itertools import groupby

PUNCTS = '!.?'

def well_formed_sentence(sent, end_sent=False):
    sent = first_upper(sent)
    sent = sent.replace('  ', ' ')
    sent = sent.replace(' i ', " I ")
    if end_sent and len(sent) > 0 and sent[-1] not in PUNCTS:
        sent += "."
    return clean_text(sent)

def clean_text(s):
    punc = set(punctuation) - set('.')
    punc.add("\n")
    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return ''.join(newtext)

def first_upper(s):
    if len(s) == 0:
        return s
    else:
        return s[0].upper() + s[1:]

device = 'cuda' if torch.cuda.is_available() else "cpu"
num_beams = 25

def pegasus_paraphrase(texts, 
                       tokenizer, 
                       paraphraser_name="tuner007/pegasus_paraphrase", 
                       device='cuda', 
                       num_beams = 10, 
                       temp=2, 
                       bsz=-1,
                       bigram=False, 
                       bert_threshold=0.03
                       ):
    paraphraser = PegasusForConditionalGeneration.from_pretrained(
        paraphraser_name).to(device)
    paraphraser_tokenizer = PegasusTokenizer.from_pretrained(paraphraser_name)

    def paraphrase(sents):
        '''
        Arguments:
            sents: list of sentences (max len under 60!)
        Returns:
            paraphrased: list of paraphrased sents
        '''
        batch = paraphraser_tokenizer(
            sents, truncation=True, padding='longest', return_tensors="pt", max_length=60).to(device)
        
        paraphrased_ids = paraphraser.generate(
            **batch, max_length=60, num_beams=num_beams, num_return_sequences=num_beams, temperature=temp, repetition_penalty=1.03)
        # batch decode and return the first one
        paraphrased = [paraphraser_tokenizer.decode(paraphrased_ids[i*num_beams], skip_special_tokens=True) for i in range(len(paraphrased_ids) // num_beams)]       
            # breakpoint()    
        
        return paraphrased
    
    # dataset has to be a text list
    sents, data_len = [], []
    for text in tqdm(texts, desc="Tokenizer"):
        sent_list = sent_tokenize(text)
        sents.extend(sent_list)
        data_len.append(len(sent_list))
    paras = []

    if bsz != -1:
        batched_sents = [sents[i:i+bsz] for i in range(0, len(sents), bsz)]
        for batch in tqdm(batched_sents, desc="Paraphrasing"):
            paraphrased = paraphrase(batch)
            paras.extend(paraphrased)

    else:
        for sent in tqdm(sents):
            paraphrased = paraphrase([sent])
            paraphrased = [well_formed_sentence(para) for para in paraphrased]
            if bigram:
                para = accept_by_bigram_overlap(sent, paraphrased, tokenizer, bert_threshold)
            else: 
                para = paraphrased[0]
            paras.append(para)
    
    start_pos = 0
    output = []
    new_texts = []
    for l in data_len:
        output.append(paras[start_pos: start_pos+l])
        new_texts.append(sents[start_pos: start_pos+l])
        start_pos+=l
    new_dataset = Dataset.from_dict(
        {'text': new_texts, 'para_text': output})
    name = args.data_path + \
        f'-pegasus-bigram={bigram}-threshold={bert_threshold}'
    new_dataset.save_to_disk(name)
    return output

def parrot_paraphrase(parrot, texts, tokenizer, num_beams=10, bigram=False, save_to_disk=True, avg_sent_len=20, save_by_sents=False, bert_threshold=0.03):
    # modified parrot source code to have the num_beams argument
    def paraphrase(sent):
        para_phrases = parrot.augment(input_phrase=sent,
                                      use_gpu=True,
                                      diversity_ranker="levenshtein",
                                      do_diverse=True,
                                      max_return_phrases=num_beams,
                                      max_length=60,
                                      adequacy_threshold=0.8,
                                      fluency_threshold=0.8)
        return para_phrases

    sents, data_len = [], []
    for text in tqdm(texts, desc="Tokenizer"):
        sent_list = sent_tokenize(text)
        sents.extend(sent_list)
        data_len.append(len(sent_list))
    start_pos = 0
    paras = []
    total_paraphrased = []
    for sent in tqdm(sents):
        paraphrased = paraphrase(sent)
        paraphrased = [well_formed_sentence(
            para, end_sent=True) for para in paraphrased]
        total_paraphrased.append(paraphrased)
        if bigram:
            para = accept_by_bigram_overlap(sent, paraphrased, tokenizer, bert_threshold=bert_threshold)
        else:
            para = paraphrased[0]
        paras.append(para)
    start_pos = 0
    output = []
    new_texts = []
    if save_by_sents:
        for l in data_len:
            output.append(paras[start_pos: start_pos+l])
            new_texts.append(sents[start_pos: start_pos+l])
            start_pos += l
    elif save_to_disk:
        new_texts = texts
        for l in data_len:
            output.append(" ".join(paras[start_pos: start_pos+l]))
            start_pos += l
    new_dataset = Dataset.from_dict({'text': new_texts, 'para_text': output})   
    name = args.data_path + \
        f'-parrot-bigram={bigram}-threshold={bert_threshold}'
    new_dataset.save_to_disk(name)
    pkl_name = args.data_path + f'-parrot-bigram={bigram}-threshold={bert_threshold}-all_beams.pkl'
    with open(pkl_name, 'wb') as f:
        pickle.dump(total_paraphrased, f)
        f.close()
    return output

def paraphrase_openai(client, texts, num_beams, bigram=False):
    new_texts = []
    all_paras = []
    MAX_ITER = 10
    for text in tqdm(texts, desc="Paraphrasing with OpenAI"):
        sents = sent_tokenize(text)
        para_sents = []
        fail = False
        for i in range(len(sents)):
            sent = sents[i]
            context = sents[:i]
            num_iter = 0
            if bigram:
                para_ls = []
                prompt = gen_bigram_prompt(sent, context, num_beams)
                # if insufficient number of para_sents generated, try again
                while(len(para_ls) < 5 and num_iter < MAX_ITER):
                    para_str = query_openai_bigram(client, prompt)
                    # use regex to extract list from string
                    para_ls = extract_list(para_str)
                    num_iter += 1
                    # openai refuses to paraphrase, thendiscard
                if num_iter <= MAX_ITER:
                    para_sents.append(para_ls)
                else: 
                    fail = True
            else:
                prompt = gen_prompt(sent, context)
                para = query_openai(client, prompt)
                para_sents.append(para)
        if not fail:
            new_texts.append(sents)
            all_paras.append(para_sents) 
    
    save_path = args.data_path + f'-openai-num_beams={num_beams}-bigram={bigram}'
    Dataset.from_dict({'text': new_texts, 'para_text': all_paras}).save_to_disk(save_path)
    return new_texts, all_paras

STANDARD_PROMPT = (
    "You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences.\n"
    "Ensure that the final output contains the same information as the original text and has roughly the same length.\n"
    "Do not leave out any important details when rewriting in your own voice.\n"
    "Do not include any information that is not present in the original text.\n"
    "Do not respond with a greeting or any other extraneous information.\n"
    "Skip the preamble. Just rewrite the text directly."
)

SHUFFLE_PROMPT = (
    "Rewrite the following text. Paraphrase every sentence, shuffle sentence order where possible, "
    "and preserve all original information exactly—no additions, no omissions, and roughly the same length. "
    "Output only the rewritten text."
)

COMBINE_PROMPT = (
    "You are an expert copy-editor. Please rewrite the following text in your own voice, merging every pair of consecutive sentences into one sentence where possible, and paraphrasing all sentences.\n"
    "Ensure that the final output contains the same information as the original text and has roughly the same length.\n"
    "Do not leave out any important details when rewriting in your own voice.\n"
    "Do not include any information that is not present in the original text.\n"
    "Do not respond with a greeting or any other extraneous information.\n"
    "Skip the preamble. Just rewrite the text directly."
)

def custom_paraphrase(texts, custom_model, device='cuda', bsz=1):
    # Auto-detect whether custom_model is a LoRA adapter or a full model
    try:
        from peft import PeftConfig, PeftModel
        adapter_config = PeftConfig.from_pretrained(custom_model)
        base_model_name = adapter_config.base_model_name_or_path
        print(f"Detected LoRA adapter, loading base model: {base_model_name}")
        para_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.float16).to(device)
        model = PeftModel.from_pretrained(model, custom_model).to(device)
    except Exception:
        print(f"No adapter config found, loading as full model: {custom_model}")
        para_tokenizer = AutoTokenizer.from_pretrained(custom_model)
        model = AutoModelForCausalLM.from_pretrained(custom_model, dtype=torch.float16).to(device)
    model.eval()

    if para_tokenizer.pad_token_id is None:
        para_tokenizer.pad_token_id = para_tokenizer.eos_token_id
    # Left-padding is required for batched generation with causal LMs
    para_tokenizer.padding_side = "left"

    def build_prompt(text):
        return para_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": COMBINE_PROMPT},
                {"role": "user", "content": f"\n[[START OF TEXT]]\n{text}\n[[END OF TEXT]]"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ) + "[[START OF PARAPHRASE]]\n"

    def paraphrase_batch(batch_texts):
        prompts = [build_prompt(text) for text in batch_texts]
        inputs = para_tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=1.0,
                do_sample=True,
                pad_token_id=para_tokenizer.pad_token_id,
            )
        results = []
        for i in range(len(prompts)):
            # Decode only the newly generated tokens (skip the input)
            new_tokens = outputs[i][inputs['input_ids'].shape[1]:]
            decoded = para_tokenizer.decode(new_tokens, skip_special_tokens=True)
            if "[[END OF" in decoded:
                decoded = decoded.split("[[END OF")[0]
            results.append(decoded.strip())
        return results

    output = []
    batched_texts = [texts[i:i+bsz] for i in range(0, len(texts), bsz)]
    for batch in tqdm(batched_texts, desc=f"Paraphrasing with {custom_model}"):
        paras = paraphrase_batch(batch)
        output.extend(paras)

    model_short = custom_model.split("/")[-1]
    name = args.data_path + f'-custom-{model_short}'
    new_dataset = Dataset.from_dict({'text': texts, 'para_text': output})
    new_dataset.save_to_disk(name)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--model_path', type=str, default='AbeHou/opt-1.3b-semstamp')
    parser.add_argument('--bsz', type=int, default=-1)
    parser.add_argument('--paraphraser', type=str,
                        default="parrot", choices=['pegasus',
                                                    'parrot',
                                                    'openai',
                                                    'parrot-bigram',
                                                    'pegasus-bigram',
                                                    'openai-bigram',
                                                    'custom'])
    parser.add_argument('--custom_model', type=str, default=None,
                        help='HuggingFace model path for custom paraphraser (PEFT/LoRA adapter)')
    parser.add_argument('--temp', type=float, default=2.0, help='decode temperature')
    parser.add_argument('--bert_threshold', type=float, default=0.0, help='threshold for bert similarity between original and paraphrased')
    parser.add_argument('--num_beams', type=int, default=10, help='number of beams for beam-search')
    args = parser.parse_args()

    batch_size = args.bsz if args.bsz > 0 else 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()

    dataset = load_from_disk(args.data_path)
    texts = dataset['text']

    if args.paraphraser == 'parrot':
        parrot = SParrot()
        parrot_paraphrase(parrot, texts, tokenizer, num_beams=args.num_beams, 
                            bert_threshold=args.bert_threshold)
    elif args.paraphraser == 'parrot-bigram':
        parrot = SParrot()
        parrot_paraphrase(parrot, texts, tokenizer, bigram=True, num_beams=args.num_beams,
                            bert_threshold=args.bert_threshold)
    elif args.paraphraser == 'pegasus-bigram':
        pegasus_paraphrase(texts, tokenizer, bigram=True, num_beams=args.num_beams, bert_threshold=args.bert_threshold, bsz=batch_size)
    elif args.paraphraser == 'pegasus':
        pegasus_paraphrase(texts, tokenizer, num_beams = args.num_beams, bert_threshold=args.bert_threshold, bsz=batch_size)
    elif args.paraphraser == 'openai':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        new_texts, paras = paraphrase_openai(client, texts, args.num_beams, bigram=False)
    elif args.paraphraser == 'openai-bigram':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        new_texts, paras = paraphrase_openai(client, texts, args.num_beams, bigram=True)
    elif args.paraphraser == 'custom':
        assert args.custom_model is not None, "--custom_model is required when using --paraphraser custom"
        custom_paraphrase(texts, args.custom_model, device=device, bsz=batch_size)
    else:
        raise NotImplementedError
