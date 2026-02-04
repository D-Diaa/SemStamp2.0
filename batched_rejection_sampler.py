"""
Dedicated batched sentence-wise rejection sampler.

This module provides a BatchedRejectionSampler class that performs rejection sampling
on batches of generated sentences, accepting the first sample that passes a given
filtering/acceptance criterion.
"""

import torch
from typing import Callable, Tuple, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteriaList
from nltk.tokenize import sent_tokenize


class BatchedSentenceStoppingCriteria:
    """
    Stopping criteria that stops generation when all samples in the batch
    have generated at least one complete extra sentence.
    """

    def __init__(self, tokenizer, input_length: int, batch_size: int, max_new_tokens: int = 512):
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.stopped = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.BoolTensor:
        if self.stopped is None:
            self.stopped = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for i in range(input_ids.size(0)):
            if self.stopped[i]:
                continue

            new_tokens = input_ids[i, self.input_length:]
            if new_tokens.numel() == 0:
                continue

            # Safety: stop if we've generated too many tokens
            if new_tokens.numel() >= self.max_new_tokens:
                self.stopped[i] = True
                continue

            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if not text:
                continue

            # Stop when we detect at least 1 complete sentence (2 sentences means 1 is complete)
            sentences = sent_tokenize(text)
            if len(sentences) >= 2:
                self.stopped[i] = True

        # Return True (stop) only when ALL sequences have stopped
        return self.stopped.all()


def extract_first_sentence(text: str) -> str:
    """
    Extract only the first complete sentence from text.
    Returns the sentence with a preceding space for proper concatenation.
    """
    if not text.strip():
        return ""

    sentences = sent_tokenize(text)
    if not sentences:
        return ""

    return " " + sentences[0].strip()


class BatchedRejectionSampler:
    """
    A batched sentence-wise rejection sampler.

    This sampler generates multiple candidate sentences in parallel and accepts
    the first one that passes the given acceptance criterion. Generation is
    sentence-wise: tokens are sampled until all candidates have at least one
    complete sentence, then each candidate is truncated to exactly one sentence.

    Args:
        model_path: Path or name of the HuggingFace model to use.
        device: Device to run the model on (e.g., 'cuda', 'cuda:0', 'cpu').
        batch_size: Number of candidate sentences to generate per batch.
        max_batches: Maximum number of batches to try before giving up.
        dtype: Data type for the model (default: torch.bfloat16).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 16,
        max_batches: int = 8,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.dtype = dtype

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()

    def _generate_batch(
        self,
        text_ids: torch.LongTensor,
        gen_config: GenerationConfig,
    ) -> Tuple[List[str], torch.LongTensor]:
        """
        Generate a batch of candidate sentences.

        Args:
            text_ids: Input token IDs, shape (1, seq_len).
            gen_config: Generation configuration.

        Returns:
            new_texts: List of generated sentence strings (truncated to 1 sentence).
            outputs: Tensor of all generated token IDs, shape (batch_size, seq_len).
        """
        prompt_len = text_ids.size(1)

        # Create config with num_return_sequences for batched output
        batch_gen_config = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens if gen_config.max_new_tokens else 128,
            min_new_tokens=gen_config.min_new_tokens if gen_config.min_new_tokens else 1,
            do_sample=True,
            temperature=gen_config.temperature if gen_config.temperature is not None else 1.0,
            top_k=gen_config.top_k if gen_config.top_k is not None else 50,
            top_p=gen_config.top_p if gen_config.top_p is not None else 1.0,
            repetition_penalty=gen_config.repetition_penalty if gen_config.repetition_penalty is not None else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=self.batch_size,
        )

        # Create stopping criteria
        stopping_criteria = StoppingCriteriaList([
            BatchedSentenceStoppingCriteria(
                self.tokenizer,
                prompt_len,
                batch_size=self.batch_size,
                max_new_tokens=gen_config.max_new_tokens if gen_config.max_new_tokens else 128,
            )
        ])

        # Create attention mask
        attention_mask = torch.ones_like(text_ids)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                text_ids,
                generation_config=batch_gen_config,
                attention_mask=attention_mask,
                stopping_criteria=stopping_criteria,
            )

        # Extract first sentence from each generated sequence
        new_texts = []
        for i in range(outputs.size(0)):
            generated_text = self.tokenizer.decode(
                outputs[i, prompt_len:], skip_special_tokens=True
            )
            first_sentence = extract_first_sentence(generated_text)
            new_texts.append(first_sentence)

        return new_texts, outputs

    def generate(
        self,
        prompt: str,
        gen_config: GenerationConfig,
        acceptance_fn: Callable[[str], bool],
    ) -> Tuple[str, bool]:
        """
        Generate a single sentence using batched rejection sampling.

        Samples tokens until all samples in each batch have at least one extra
        sentence, then truncates every candidate to exactly one extra sentence.
        Returns the first accepted sample, or a random sample if max_batches is reached.

        Args:
            prompt: The input prompt text.
            gen_config: Generation configuration for the model.
            acceptance_fn: A function that takes a candidate sentence (str) and
                          returns True if the candidate should be accepted.

        Returns:
            A tuple of (sample, accepted) where:
                - sample: The generated sentence string with proper preceding whitespace.
                - accepted: True if the sample passed the acceptance criterion,
                           False if max_batches was reached without finding an acceptable sample.
        """
        # Encode the prompt
        text_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        fallback_sample: Optional[str] = None

        for _ in range(self.max_batches):
            # Generate a batch of candidate sentences
            candidate_texts, _ = self._generate_batch(text_ids, gen_config)

            # Check each candidate against the acceptance criterion
            for candidate in candidate_texts:
                # Skip empty candidates
                if not candidate.strip():
                    continue

                # Save first non-empty candidate as fallback
                if fallback_sample is None:
                    fallback_sample = candidate

                # Check if candidate passes acceptance criterion
                if acceptance_fn(candidate):
                    return candidate, True

        # Max batches reached without finding an acceptable sample
        # Return the fallback sample (or empty string if all were empty)
        if fallback_sample is not None:
            return fallback_sample, False
        else:
            return "", False

    def generate_continuation(
        self,
        prompt: str,
        gen_config: GenerationConfig,
        acceptance_fn: Callable[[str, str], bool],
        max_sentences: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        """
        Generate a multi-sentence continuation using iterative rejection sampling.

        Generates sentences one at a time, each using batched rejection sampling,
        until the maximum number of sentences or tokens is reached.

        Args:
            prompt: The input prompt text.
            gen_config: Generation configuration for the model.
            acceptance_fn: A function that takes (context, candidate_sentence) and
                          returns True if the candidate should be accepted.
                          The context is the full text generated so far (including prompt).
            max_sentences: Maximum number of sentences to generate (default: None = no limit).
            max_tokens: Maximum number of new tokens to generate (default: uses gen_config).

        Returns:
            A tuple of (text, accepted_count, total_count) where:
                - text: The full generated text (prompt + all generated sentences).
                - accepted_count: Number of sentences that passed the acceptance criterion.
                - total_count: Total number of sentences generated.
        """
        text = prompt
        text_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_length = text_ids.size(1)

        max_new_tokens = max_tokens if max_tokens else (gen_config.max_new_tokens or 256)
        sentence_count = 0
        accepted_count = 0

        while True:
            # Create a context-aware acceptance function
            current_context = text

            def contextualized_acceptance(candidate: str) -> bool:
                return acceptance_fn(current_context, candidate)

            # Generate next sentence
            new_sentence, accepted = self.generate(
                text,
                gen_config,
                contextualized_acceptance,
            )

            if not new_sentence.strip():
                # No valid sentence generated, stop
                break

            # Append the new sentence
            text += new_sentence
            sentence_count += 1
            if accepted:
                accepted_count += 1

            # Re-encode to check token count
            text_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            current_new_tokens = text_ids.size(1) - prompt_length

            # Check stopping conditions
            if max_sentences is not None and sentence_count >= max_sentences:
                break

            if current_new_tokens >= max_new_tokens:
                break

        return text, accepted_count, sentence_count


def create_sampler(
    model_path: str,
    device: str = "cuda",
    batch_size: int = 16,
    max_batches: int = 8,
    dtype: torch.dtype = torch.bfloat16,
) -> BatchedRejectionSampler:
    """
    Factory function to create a BatchedRejectionSampler.

    Args:
        model_path: Path or name of the HuggingFace model to use.
        device: Device to run the model on (e.g., 'cuda', 'cuda:0', 'cpu').
        batch_size: Number of candidate sentences to generate per batch.
        max_batches: Maximum number of batches to try before giving up.
        dtype: Data type for the model (default: torch.bfloat16).

    Returns:
        An initialized BatchedRejectionSampler instance.
    """
    return BatchedRejectionSampler(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
        dtype=dtype,
    )


def main():
    """Example usage of the BatchedRejectionSampler."""
    import argparse

    parser = argparse.ArgumentParser(description="Batched Rejection Sampler Example")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model path or name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Number of candidates per batch",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=8,
        help="Maximum batches before giving up",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate",
    )
    args = parser.parse_args()

    print(f"Initializing sampler with model: {args.model}")
    sampler = create_sampler(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=1,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
        pad_token_id=sampler.tokenizer.pad_token_id,
    )

    # Example 1: Simple acceptance criterion (accept all non-empty)
    print("\n" + "=" * 60)
    print("Example 1: Accept any non-empty sentence")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")

    def accept_all(candidate: str) -> bool:
        return len(candidate.strip()) > 0

    sentence, accepted = sampler.generate(args.prompt, gen_config, accept_all)
    print(f"Generated: {sentence}")
    print(f"Accepted: {accepted}")

    # Example 2: Acceptance criterion based on length
    print("\n" + "=" * 60)
    print("Example 2: Accept sentences with at least 10 words")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")

    def accept_long_sentences(candidate: str) -> bool:
        return len(candidate.split()) >= 10

    sentence, accepted = sampler.generate(args.prompt, gen_config, accept_long_sentences)
    print(f"Generated: {sentence}")
    print(f"Accepted: {accepted}")

    # Example 3: Multi-sentence continuation
    print("\n" + "=" * 60)
    print("Example 3: Generate multi-sentence continuation, not repeating words from context")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")

    def context_aware_accept(context: str, candidate: str) -> bool:
        # Accept if candidate doesn't repeat words from context too much
        context_words = set(context.lower().split())
        candidate_words = candidate.lower().split()
        if not candidate_words:
            return False
        overlap = sum(1 for w in candidate_words if w in context_words)
        overlap_ratio = overlap / len(candidate_words)
        return overlap_ratio < 0.25

    full_text, accepted_count, total_count = sampler.generate_continuation(
        args.prompt,
        gen_config,
        context_aware_accept,
        max_sentences=3,
    )
    print(f"Full text:\n{full_text}")
    print(f"Accepted: {accepted_count}/{total_count} sentences")

    # Example 4: Accept only sentences containing specific punctuation/structure
    print("\n" + "=" * 60)
    print("Example 4: Accept sentences containing a comma (complex sentences)")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")

    def accept_complex_sentences(candidate: str) -> bool:
        # Accept sentences that contain a comma (indicating more complex structure)
        return "," in candidate and len(candidate.split()) >= 5

    sentence, accepted = sampler.generate(args.prompt, gen_config, accept_complex_sentences)
    print(f"Generated: {sentence}")
    print(f"Accepted: {accepted}")


if __name__ == "__main__":
    main()
