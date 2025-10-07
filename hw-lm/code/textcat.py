#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import torch
import sys

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the trained model 1",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the trained model 2",
    )
    parser.add_argument(
        "prior_probability",
        type=float,
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )
    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy) 

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)
        
    return log_prob


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
        
    lm1 = LanguageModel.load(args.model1, device = args.device)
    lm2 = LanguageModel.load(args.model2, device = args.device)
    
    if ("OOV" not in lm1.vocab) or ("OOV" not in lm2.vocab) or ("EOS" not in lm1.vocab) or ("EOS" not in lm2.vocab):
        log.critical("where's oov and eos")
        sys.exit(1)

    if lm1.vocab != lm2.vocab:
        log.critical("vocabs don't match")
        sys.exit(2)

    name1 = args.model1.name
    name2 = args.model2.name

    pgen = float(args.prior_probability)
    pspam = 1 - pgen

    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    total_1 = 0.0
    total_2 = 0.0

    for file in args.test_files:
        log_prob_1: float = file_log_prob(file, lm1)
        log_prob_2: float = file_log_prob(file, lm2)

        if pgen == 0.0:
            prior1 = -math.inf
        elif pgen == 1.0:
            prior1 = 0.0
        else:
            prior1 = math.log(pgen)

        if pgen == 0.0:
            prior2 = 0.0
        elif pgen == 1.0:
            prior2 = -math.inf
        else:
            prior2 = math.log(pspam)
         
        post_1 = prior1 + log_prob_1
        post_2 = prior2 + log_prob_2

        if post_1 >= post_2:
            print(f"{name1} {file}")
            total_1 += 1
        else:
            print(f"{name2} {file}")
            total_2 += 1
        
    n_pred = int(total_1 + total_2)
    pct1 = (total_1 / n_pred * 100.0) if n_pred > 0 else 0.0
    pct2 = (total_2 / n_pred * 100.0) if n_pred > 0 else 0.0

    print(f"{int(total_1)} files were more probably from {name1} ({pct1:.2f}%)")
    print(f"{int(total_2)} files were more probably from {name2} ({pct2:.2f}%)")

    
    # But cross-entropy is conventionally measured in bits: so when it's
    # time to print cross-entropy, we convert log base e to log base 2, 
    # by dividing by log(2).

    #bits = - file_log_prob/ math.log(2)   # convert to bits of surprisal

    # We also divide by the # of tokens (including EOS tokens) to get
    # bits per token.  (The division happens within the print statement.

    #tokens = sum(num_tokens(test_file) for test_file in args.test_files)
    #print(f"Overall cross-entropy:\t{bits / tokens:.5f} bits per token")


if __name__ == "__main__":
    main()
