#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
import logging
import math
import tqdm
import heapq
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple, Set

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
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


class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar.filtered_for_tokens(tokens)
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self._run_earley()    # run Earley's algorithm to construct self.cols

    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        for item in self.cols[-1].all():    # the last column
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0
                    return True
        return False   # we didn't find any appropriate item

    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]
        for col in self.cols:
            col.set_beam(getattr(self.grammar, "_beam_width_bits", float("inf")))

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            while column:    # while agenda isn't empty
                item = column.pop()   # dequeue the next unprocessed item
                next = item.next_symbol()
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)   
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)                 

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        column = self.cols[position]
        if column.already_predicted(nonterminal):
            return
        column.mark_predicted(nonterminal)

        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position)
            if self._push(position, new_item, score=rule.weight, bp=('predict', None)):
                log.debug(f"\tPredicted: {new_item} in column {position}")
                self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced()
            if self._push(position + 1, new_item, score=self._score(position, item), bp=(1, self.tokens[position])):
                log.debug(f"\tScanned to get: {new_item} in column {position+1}")
                self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        mid = item.start_position   # start position of this item = end position of item to its left
        child_cost = self._score(position, item)
        for customer in self.cols[mid].all():  # could you eliminate this inefficient linear search?
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced()
                parent_so_far = self._score(mid, customer)
                new_cost = parent_so_far + child_cost
                if self._push(position, new_item, score=new_cost, bp=(0, customer, item)): 
                    log.debug(f"\tAttached to get: {new_item} in column {position}")
                    self.profile["ATTACH"] += 1

    def _push(self, position: int, item: Item, score: float, bp=None) -> None:
        return self.cols[position].push(item, score, bp)

    def _score(self, position: int, item: Item) -> float:
        return self.cols[position].score(item)

    def _bp(self, position: int, item: Item):
        return self.cols[position].bp(item)

    def best_final_item(self) -> Optional["Item"]:
        best_item, best_cost = None, float("inf")
        last_col = self.cols[-1]
        for it in last_col.all():
            if (it.rule.lhs == self.grammar.start_symbol
                and it.next_symbol() is None
                and it.start_position == 0):
                cost = last_col.score(it)
                if cost < best_cost:
                    best_cost = cost
                    best_item = it
        return best_item

    def best_parse(self):
        final = self.best_final_item()
        if final is None:
            return None
        cost = self.cols[-1].score(final)
        tree = self._reconstruct_as_sexpr(final, end_pos=len(self.tokens))
        return cost, tree


    def _reconstruct_as_sexpr(self, item: Item, end_pos: int) -> str:
        kids = []
        cur = item
        cur_pos = end_pos
        while cur.dot_position > 0:
            kind, *rest = self.cols[cur_pos].bp(cur)
            if kind == 1:
                word = rest[0]
                kids.append(word)
                cur = Item(cur.rule, cur.dot_position - 1, cur.start_position)
                cur_pos -= 1
            elif kind == 0:
                prev_parent, child = rest
                child_tree = self._reconstruct_as_sexpr(child, end_pos=cur_pos)
                kids.append(child_tree)
                cur = prev_parent
                cur_pos = child.start_position
            else:
                raise RuntimeError("Unknown backpointer")
        kids.reverse()

        inside = " ".join(kids)
        return f"({item.rule.lhs} {inside})"

class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.  

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.

    #>>> a = Agenda()
    #>>> a.push(3)
    #>>> a.push(5)
    #>>> a.push(3)   # duplicate ignored
    #>>> a
    Agenda([]; [3, 5])
    #>>> a.pop()
    3
    #>>> a
    Agenda([3]; [5])
    #>>> a.push(3)   # duplicate ignored
    #>>> a.push(7)
    #>>> a
    Agenda([3]; [5, 7])
    #>>> while a:    # that is, while len(a) != 0
    ...    print(a.pop())
    5
    7

    """

    def __init__(self) -> None:
        self._items: List[Item] = []       # list of all items that were *ever* pushed
        self._index: Dict[Item, int] = {}  # stores index of an item if it was ever pushed

        self._todo_heap: List[Tuple[float, int, Item]] = []
        self._inqueue: Set[Item] = set()
        self._counter: int = 0  # tie-breaker for heap
        
        self._backpointer: Dict[Item, Tuple[str, object]] = {}
        self._score: Dict[Item, float] = {}

        self._best_score: float = float("inf")
        self.set_beam(12.0)
        self._beam_width: float = float("inf")

        self._predicted_nts: Set[str] = set()
        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.  
        # 
        # However, we provided this defsign because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._todo_heap)

    def already_predicted(self, nonterminal: str) -> bool:
        return nonterminal in self._predicted_nts

    def mark_predicted(self, nonterminal: str) -> None:
        self._predicted_nts.add(nonterminal)     
    
    def push(self, item: Item, score: float, bp: Optional[Tuple[str, object]] = None) -> bool:
        """Add (enqueue) the item, unless it was previously added."""
        old = self._score.get(item)

        if old is not None and score >= old- 1e-12 :
            return False
            
        if item not in self._index and item not in self._items:
            self._index[item] = len(self._items)
            self._items.append(item)
        
        self._score[item] = score
        if score < self._best_score:
            self._best_score = score

        if score > self._best_score + self._beam_width:
            return False

        if bp is not None:
            self._backpointer[item] = bp

        if item not in self._inqueue:
            heapq.heappush(self._todo_heap, (score, self._counter, item))
            self._counter += 1
            self._inqueue.add(item)
        
        return True

    def set_beam(self, beam_width: float) -> None:
        self._beam_width = beam_width
            
    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if not self._todo_heap:
            raise IndexError
        _, _, item = heapq.heappop(self._todo_heap)
        self._inqueue.remove(item)
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items

    def score(self, item: Item) -> float:
        return self._score[item]

    def bp(self, item: Item) -> Optional[Tuple[str, object]]:
        return self._backpointer.get(item)

    def __repr__(self):
        """Provide a human-readable string REPResentation of this Agenda."""
        return f"{self.__class__.__name__}({list(self._items)}; queue_size={len(self._todo_heap)})"


class Grammar:
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def _terminals_in_rule(self, rule: Rule) -> Tuple[str, ...]:
         return tuple(sym for sym in rule.rhs if not self.is_nonterminal(sym))

    def filtered_for_tokens(self, tokens: Iterable[str]) -> "Grammar":
        token_set = set(tokens)
        g = Grammar(self.start_symbol)
        kept = 0
        dropped = 0
        for lhs, rules in self._expansions.items():
            for r in rules:
                terms = self._terminals_in_rule(r)
                if all(t in token_set for t in terms):
                    if lhs not in g._expansions:
                        g._expansions[lhs] = []
                    g._expansions[lhs].append(r)
                    kept += 1
                else:
                    dropped += 1
        if dropped:
            logging.getLogger(Path(__file__).stem).debug(
                f"Filtered grammar for sentence: kept {kept} rules, dropped {dropped} rules not matching terminals"
            )
        return g

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited line of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())  
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us declare that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.

    #>>> r = Rule('S',('NP','VP'),3.14)
    #>>> r
    S → NP VP
    #>>> r.weight
    3.14
    #>>> r.weight = 2.718
    Traceback (most recent call last):
    dataclasses.FrozenInstanceError: cannot assign to field 'weight'
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        # Note: You might want to modify this to include the weight.

        return f"{self.weight} {self.lhs} → {' '.join(self.rhs)}"

    
# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse chart, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position)

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        # Note: If you revise this class to change what an Item stores, you'll probably want to change this method too.
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"  # matches notation on slides


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f:
            sentence = sentence.strip()
            if not sentence:
                continue

            #log.debug("=" * 70)
            #log.debug(f"Parsing sentence: {sentence}")

            chart = EarleyChart(sentence.split(), grammar, progress=args.progress)

            if chart.accepted():
                best = chart.best_parse() 
                if best is None:
                    print("NONE")
                else:
                    cost, tree = best
                    print(tree)
                    print(cost)
            else:
                print(f"NONE")

            log.debug(f"Profile of work done: {chart.profile}")

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)   # run tests
    main()
