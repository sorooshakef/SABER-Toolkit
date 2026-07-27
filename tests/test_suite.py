"""Precision/recall harness for the pattern matchers.

Runs every matcher in ``saber/matchers`` against the positive and negative
example sentences in ``tests.yaml`` and writes the results to
``test_results.csv``. Each matcher is measured twice: once with all of its
patterns active ("Full"), and once with only its first pattern registered
("First Only"), which shows how much of the matcher's coverage that one pattern
accounts for.

    python tests/test_suite.py [tests.yaml]
"""

import os
import sys
import yaml
import csv
import spacy.matcher
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Allow running the file directly from a checkout, without installing.
sys.path.insert(0, str(HERE.parent))

# Import the NLP pipelines and the matcher registry from the package
try:
    from saber.nlp import nlp_stanza, nlp_small
    from saber.registry import load_matchers
except ImportError as e:
    print(f"Failed to import SABER: {e}")
    print("Install the package (pip install -e .) or run this from the repository root.")
    sys.exit(1)


def load_all_matchers():
    """Return ``{function_name: matcher_function}`` for every matcher."""
    return dict(load_matchers())

class LimitedMatcherProxy:
    """
    A proxy object that wraps a spaCy matcher instance and limits it 
    to only registering the very first pattern added.
    """
    def __init__(self, real_matcher):
        self._real_matcher = real_matcher
        self._added_once = False

    def add(self, match_id, patterns, **kwargs):
        # We only ever allow ONE call to .add(), and only use the FIRST pattern from that call.
        if not self._added_once:
            self._added_once = True
            if patterns:
                # Only add the first pattern in the list
                self._real_matcher.add(match_id, patterns[0:1], **kwargs)

    def __call__(self, *args, **kwargs):
        return self._real_matcher(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real_matcher, name)

    @property
    def pipe(self):
        return self._real_matcher.pipe

def with_first_pattern_only(func):
    """
    Decorator that patches the globals of the function and the spacy.matcher module
    to return LimitedMatcherProxy objects.
    """
    def wrapper(doc):
        # Save original constructors from spacy.matcher
        orig_spacy_matcher = spacy.matcher.Matcher
        orig_spacy_dep = spacy.matcher.DependencyMatcher
        orig_spacy_phrase = spacy.matcher.PhraseMatcher

        # Save original globals from the function's module
        orig_globals = {}
        for name in ["Matcher", "DependencyMatcher", "PhraseMatcher"]:
            if name in func.__globals__:
                orig_globals[name] = func.__globals__[name]

        # Define factory wrappers using the original (saved) classes
        def limited_matcher_factory(*args, **kwargs):
            return LimitedMatcherProxy(orig_spacy_matcher(*args, **kwargs))
        
        def limited_dep_matcher_factory(*args, **kwargs):
            return LimitedMatcherProxy(orig_spacy_dep(*args, **kwargs))

        def limited_phrase_matcher_factory(*args, **kwargs):
            return LimitedMatcherProxy(orig_spacy_phrase(*args, **kwargs))

        try:
            # Patch spacy.matcher class references
            spacy.matcher.Matcher = limited_matcher_factory
            spacy.matcher.DependencyMatcher = limited_dep_matcher_factory
            spacy.matcher.PhraseMatcher = limited_phrase_matcher_factory
            
            # Patch the function's global namespace
            if "Matcher" in orig_globals: func.__globals__["Matcher"] = limited_matcher_factory
            if "DependencyMatcher" in orig_globals: func.__globals__["DependencyMatcher"] = limited_dep_matcher_factory
            if "PhraseMatcher" in orig_globals: func.__globals__["PhraseMatcher"] = limited_phrase_matcher_factory
            
            # Execute the function
            return func(doc)
        finally:
            # Restore spacy.matcher
            spacy.matcher.Matcher = orig_spacy_matcher
            spacy.matcher.DependencyMatcher = orig_spacy_dep
            spacy.matcher.PhraseMatcher = orig_spacy_phrase
            
            # Restore the function's global namespace
            for name, orig_val in orig_globals.items():
                func.__globals__[name] = orig_val
            
    return wrapper

def run_tests(yaml_file):
    if not os.path.exists(yaml_file):
        print(f"Error: {yaml_file} not found.")
        print("Please create it (e.g., tests.yaml) with your test cases.")
        sys.exit(1)

    with open(yaml_file, "r", encoding="utf-8") as f:
        test_data = yaml.safe_load(f)

    if not test_data:
        print("No test data found in the YAML file.")
        sys.exit(1)

    print("Loading matchers...")
    matcher_funcs = load_all_matchers()
    print(f"Loaded {len(matcher_funcs)} matcher functions.\n")
    
    results = []

    print("Starting tests...")
    for matcher_name, data in test_data.items():
        if not data:
             continue
             
        if matcher_name not in matcher_funcs:
            print(f"Warning: Matcher '{matcher_name}' not found. Skipping.")
            continue
            
        matcher_func = matcher_funcs[matcher_name]
        matcher_func_first_only = with_first_pattern_only(matcher_func)
        
        positives = data.get("positive", []) or []
        negatives = data.get("negative", []) or []
        
        tp_full = fp_full = fn_full = tn_full = 0
        tp_first = fp_first = fn_first = tn_first = 0
        
        for sentence in positives:
            if getattr(matcher_func, "REQUIRES_SPACY", False):
                doc = nlp_small(sentence)
            else:
                doc = nlp_stanza(sentence)
            
            # Full run
            matches_full = matcher_func(doc)
            if matches_full:
                tp_full += 1
            else:
                fn_full += 1
                
            # First pattern only run
            matches_first = matcher_func_first_only(doc)
            if matches_first:
                tp_first += 1
            else:
                fn_first += 1
                
        for sentence in negatives:
            if getattr(matcher_func, "REQUIRES_SPACY", False):
                doc = nlp_small(sentence)
            else:
                doc = nlp_stanza(sentence)
            
            # Full run
            matches_full = matcher_func(doc)
            if matches_full:
                fp_full += 1
            else:
                tn_full += 1
                
            # First pattern only run
            matches_first = matcher_func_first_only(doc)
            if matches_first:
                fp_first += 1
            else:
                tn_first += 1
                
        # Calculate full metrics
        prec_full = tp_full / (tp_full + fp_full) if (tp_full + fp_full) > 0 else 0.0
        rec_full = tp_full / (tp_full + fn_full) if (tp_full + fn_full) > 0 else 0.0
        
        # Calculate first-only metrics
        prec_first = tp_first / (tp_first + fp_first) if (tp_first + fp_first) > 0 else 0.0
        rec_first = tp_first / (tp_first + fn_first) if (tp_first + fn_first) > 0 else 0.0
        
        results.append({
            "Matcher": matcher_name,
            "Type": "Full",
            "Precision": round(prec_full, 4),
            "Recall": round(rec_full, 4),
            "TP": tp_full,
            "FP": fp_full,
            "FN": fn_full,
            "TN": tn_full
        })
        
        results.append({
            "Matcher": matcher_name,
            "Type": "First Only",
            "Precision": round(prec_first, 4),
            "Recall": round(rec_first, 4),
            "TP": tp_first,
            "FP": fp_first,
            "FN": fn_first,
            "TN": tn_first
        })
        
        print(f"[{matcher_name} | Full] Precision: {prec_full:.2f} | Recall: {rec_full:.2f} (TP:{tp_full} FP:{fp_full} FN:{fn_full} TN:{tn_full})")
        print(f"[{matcher_name} | First Only] Precision: {prec_first:.2f} | Recall: {rec_first:.2f} (TP:{tp_first} FP:{fp_first} FN:{fn_first} TN:{tn_first})")

    if not results:
        print("\nNo valid tests were run.")
        return

    # Write results to a CSV file next to this script
    csv_filename = HERE / "test_results.csv"

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Matcher", "Type", "Precision", "Recall", "TP", "FP", "FN", "TN"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nTests complete. Results written to {csv_filename}")

if __name__ == "__main__":
    start_time = time.time()
    yaml_file = HERE / "tests.yaml"
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    run_tests(yaml_file)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
