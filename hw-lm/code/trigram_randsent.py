#!/usr/bin/env python3
import argparse
import logging
import math
from pathlib import Path
import torch
from probs import EOS, BOS, Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", 
        type=Path, 
        help="path to the trained model")
    parser.add_argument(
        "num_sentences", 
        type=int, 
        help="number of sentences to sample")
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=20, 
        help="maximum sentence length")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="device to use for PyTorch (cpu, cuda, or mps)",
    )

    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()




def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    log.info("Loading model...")
    lm = LanguageModel.load(args.model, device=args.device)
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("Sampling...")
    for i in range(args.num_sentences):
        sentence = lm.sample(lm, args.max_length)
        print(sentence)
        
if __name__ == "__main__":
    main()

