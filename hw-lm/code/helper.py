from probs import num_tokens
from pathlib import Path

gen_files  = list(Path("../data/gen_spam/dev/gen").glob("*"))
spam_files = list(Path("../data/gen_spam/dev/spam").glob("*"))

T_gen  = sum(num_tokens(p) for p in gen_files)
T_spam = sum(num_tokens(p) for p in spam_files)

print("GEN_TOKENS:", T_gen)
print("SPAM_TOKENS:", T_spam)
print("TOTAL:", T_gen + T_spam)
