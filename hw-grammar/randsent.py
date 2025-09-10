#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import os
import sys
import random
import argparse
from collections import defaultdict

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG")
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str, required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file
        
        Returns:
            self
        """
        # Parse the input grammar file
        self.rules = None
        self.count = None
        self._load_rules_from_file(grammar_file)

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file 
        """

        self.rules = defaultdict(lambda: {"definitions": [], "weights": []}) # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        
        with open(grammar_file, 'r') as file:
            for line in file:

                stripped_line = line.strip()

                if not stripped_line or stripped_line.startswith("#"): # skip empty lines and lines starting with with #
                    continue 

                if '#' in stripped_line: # remove inline commnets
                    stripped_line = stripped_line.split("#")[0].strip() 
    
                pts = stripped_line.split()

                prob = float(pts[0])

                entity = pts[1]

                phrase = pts[2:]

                self.rules[entity]["definitions"].append(phrase) #format: terminal: [{nonterminal: [terminal],weights:[weights] }]
                self.rules[entity]["weights"].append(prob)

    def sample(self, derivation_tree = False, max_expansions = 450, start_symbol = "ROOT"):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent 
                the tree (using bracket notation) that records how the sentence 
                was derived
                               
            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """
        self.count = max_expansions

        sentence, tree = self.generate(start_symbol)

        if derivation_tree:
            return tree
        
        return " ".join(sentence)
    
    def generate(self, start_symbol):

        '''
            Pseudocode planning:

            1. if word is terminal, return word
            2. if run out of expansions, return word
            3. if non-terminal, step into a subphrase 
            4. randomly choose one phrase
            5. step through all words in this phrase
            6. recurse with each word, add returns to sentence
        
        '''

        sentence = []
        tree = []

        if start_symbol not in self.rules: # terminal, not a key
            return [start_symbol], start_symbol

        if self.count <= 0: # reached expansion limit
            return "...", ""
        
        self.count -= 1 # decrease number of max expansions left

        phrase = random.choices(self.rules[start_symbol]["definitions"],weights=self.rules[start_symbol]["weights"])[0] # https://docs.python.org/3/library/random.html

        for word in phrase:
            add_sent, add_term = self.generate(start_symbol = word)
            sentence.extend(add_sent)
            tree.append(add_term)


        final_tree = f"({start_symbol} {' '.join(tree)})" # formatting, adding () where new start symbol occurs
        return sentence, final_tree

####################
### Main Program
####################
def main():

    #random.seed(42) # for testing purposes, remove on submission

    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
            derivation_tree=args.tree,
            max_expansions=args.max_expansions,
            start_symbol=args.start_symbol
        )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
            t = os.system(f"echo '{sentence}' | perl {prettyprint_path}")
        else:
            print(sentence)


if __name__ == "__main__":
    main()
