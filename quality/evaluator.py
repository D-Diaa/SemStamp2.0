from .utils import eval_perplexity
from .metrics import run_entropy, run_ngrams, run_mauve, run_bertscore, run_sem_ent


def eval_quality(model, gens, corpus_texts, ref_texts, tokenizer, args):

    if (type(gens[0]) == list):
        gens_text = [" ".join(g) for g in gens]
    else:
        gens_text = gens
    print("Evaluating perplexity...")
    gen_ppl = eval_perplexity(model, tokenizer, gens_text)

    print("Evaluating semantic entropy")
    sem_ent = run_sem_ent(model, gens, tokenizer, corpus_texts, args)

    print("Evaluating n-gram repetition...")
    rep_scores = run_ngrams(gens_text, tokenizer)

    print("Evaluating entropy...")
    gen_entros = run_entropy(gens_text, tokenizer)

    print("Evaluating MAUVE...")
    mauve_score = run_mauve(list(gens_text), list(ref_texts))

    print("Evaluating BERTScore...")
    bert_P, bert_R, bert_F1 = run_bertscore(list(gens_text), list(ref_texts))

    return gen_ppl, gen_entros[0], gen_entros[1], rep_scores[0], rep_scores[1], rep_scores[2], sem_ent, mauve_score, bert_P, bert_R, bert_F1
