template_list = {
    ("super_glue", "rte"): [
        "MNLI crowdsource",
        "guaranteed true",
        "can we infer",
        "GPT-3 style",
        "does this imply",
        "should assume",
        "does it follow that",
        "based on the previous passage",
        "justified in saying",
        "must be true",
    ],
    ("super_glue", "cb"): [
        "can we infer",
        "based on the previous passage",
        "claim true/false/inconclusive",
        "does it follow that",
        "justified in saying",
        "always/sometimes/never",
        "GPT-3 style",
        "consider always/sometimes/never",
        "guaranteed true",
        "must be true",
        "guaranteed/possible/impossible",
        "does this imply",
        "MNLI crowdsource",
        "should assume",
        "take the following as truth",
    ],
    ("anli", None): [
        "MNLI crowdsource",
        "should assume",
        "does it follow that",
        "GPT-3 style",
        "based on the previous passage",
        "justified in saying",
        "take the following as truth",
        "must be true",
        "can we infer",
        "guaranteed/possible/impossible",
        "always/sometimes/never",
        "does this imply",
        "consider always/sometimes/never",
        "claim true/false/inconclusive",
        "guaranteed true",
    ],
    ("super_glue", "wsc.fixed"): [
        "does the pronoun refer to",
        "by p they mean",
        "in other words",
        "I think they mean",
        "does p stand for",
        "GPT-3 Style",
        "replaced with",
        "p is/are r",
        "the pronoun refers to",
        "Who or what is/are",
    ],
    ("winogrande", "winogrande_xl"): [
        "does underscore refer to",
        "stand for",
        "underscore refer to",
        "fill in the blank",
        "Replace",
    ],
    ("story_cloze", "2016"): [
        "Answer Given options",
        "Choose Story Ending",
        "Movie What Happens Next",
        "Story Continuation and Options",
        "Novel Correct Ending",
    ],
    ("super_glue", "wic"): [
        "question-context-meaning-with-label",
        "question-context-meaning",
        "grammar_homework",
        "affirmation_true_or_false",
        "GPT-3-prompt",
        "same_sense",
        "question-context",
        "GPT-3-prompt-with-label",
        "polysemous",
        "similar-sense",
    ],
    ("hellaswag", None): [
        "Predict ending with hint",
        "Randomized prompts template",
        "complete_first_then",
        "if_begins_how_continues",
    ],
    ("super_glue", "copa"): [
        "exercise",
        "…What could happen next, C1 or C2?",
        "i_am_hesitating",
        "plausible_alternatives",
        "C1 or C2? premise, so/because…",
        "…As a result, C1 or C2?",
        "best_option",
        "…which may be caused by",
        "more likely",
        "cause_effect",
        "…why? C1 or C2",
        "choose",
    ],
    ("super_glue", "boolq"):[
        "after_reading",
        "GPT-3 Style",
        "I wonder…",
        "yes_no_question",
        "could you tell me…",
        "exam",
        "based on the following passage",
        "exercise",
        "based on the previous passage",
        "valid_binary",
    ],
    ("super_glue", "multirc"):[
        "found_this_answer",
        "is… a correct answer?",
        "grading",
        "Would it be good to answer…",
        "paragraph… question… is it… ?",
        "decide_valid",
        "is the correct answer…",
        "correct",
        "confirm",
        "I was going to say…",
    ],
    ("piqa", None):[
        "what_is_the_correct_ending",
        "pick_correct_choice_with_choice_given_before_goal",
        "pick_correct_choice_index",
        "finish_sentence_with_correct_choice",
        "choose the most appropriate solution",
    ],
    ("openbookqa", "main"):[
        "choose_an_answer_with_options",
        "which_correct",
        "pick_using_id",
        "choices",
        "only_options",
        "which_correct_inverse",
        "pick_answer_with_options",
    ],
    ("cos_e","v1.11"):[
        "question_description_option_text",
        "question_description_option_id",
        "question_option_description_text",
        "description_question_option_id",
        "description_question_option_text",
        "question_option_description_id"
    ],
    ("cosmos_qa", None):[
        "description_context_question_answer_text",
        "description_context_question_text",
        "description_context_question_answer_id",
        "context_description_question_answer_text",
        "no_prompt_id",
        "no_prompt_text",
        "context_description_question_answer_id",
        "context_question_description_answer_id",
        "context_description_question_text",
        "context_question_description_answer_text"
    ],
    ("dream",None):[
        "answer-to-dialogue",
        "baseline",
        "read_the_following_conversation_and_answer_the_question"
    ],
    ("qasc",None):[
        "qa_with_separated_facts_1",
        "qa_with_separated_facts_3",
        "qa_with_separated_facts_4",
        "qa_with_separated_facts_5",
        "qa_with_separated_facts_2"
    ],
    ("quail",None):[
        "context_question_answer_description_id",
        "context_question_answer_description_text",
        "description_context_question_answer_id",
        "context_question_description_answer_text",
        "context_question_description_answer_id",
        "no_prompt_id",
        "context_description_question_answer_id",
        "no_prompt_text",
        "context_description_question_answer_text",
        "description_context_question_answer_text"
    ],
    ("quarel",None):[
        "do_not_use",
        "logic_test",
        "heres_a_story",
        "choose_between",
        "testing_students"
    ],
    ("quartz",None):[
        "use_info_from_question_paragraph",
        "paragraph_question_plain_concat",
        "use_info_from_paragraph_question",
        "answer_question_based_on",
        "answer_question_below",
        "read_passage_below_choose",
        "having_read_above_passage",
        "given_the_fact_answer_the_q"
    ],
    ("race","high"):[
        "Taking a test",
        "Select the best answer",
        "Select the best answer (generate span)",
        "Select the best answer (no instructions)",
        "Read the article and answer the question (no option)" 
    ],
    ("race","middle"):[
        "Select the best answer",
        "Read the article and answer the question (no option)",
        "Select the best answer (no instructions)",
        "Select the best answer (generate span)",
        "Taking a test"
    ],
    ("sciq", None):[
        "Direct Question (Closed Book)",
        "Multiple Choice Question First",
        "Multiple Choice",
        "Direct Question"
    ],
    ("social_i_qa",None):[
        "I was wondering",
        "Show choices and generate answer",
        "Check if a random answer is valid or not",
        "Generate answer",
        "Show choices and generate index"
    ],
    ("wiki_hop","original"):[
        "choose_best_object_interrogative_1",
        "choose_best_object_affirmative_1",
        "choose_best_object_affirmative_3",
        "choose_best_object_affirmative_2",
        "choose_best_object_interrogative_2"
    ],
    ("wiqa",None):[
        "effect_with_string_answer",
        "effect_with_label_answer",
    ]

}



from transformers import GPT2Tokenizer, OPTForCausalLM, GPTJForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import os
import numpy as np
import torch.nn.functional as F
import json

from collections import defaultdict
from tqdm import tqdm
from promptsource.templates import DatasetTemplates
import copy

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning T0 in PyTorch, optionally few-shot.")
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="dataset name"
    )
    parser.add_argument(
        "-dc",
        "--dataset_config_name",
        type=str,
        default=None,
        required=False,
        help="dataset config name"
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    template_name = None

    if dataset_name == "anli":
        raw_datasets = load_dataset(dataset_name, split=dataset_config_name)
    else:
        raw_datasets = load_dataset(dataset_name, dataset_config_name, split="validation")

    prompts_wrapper = DatasetTemplates(
        f"{dataset_name}"
        if dataset_config_name is None
        else f"{dataset_name}/{dataset_config_name}"
    )


    dummy_example = copy.deepcopy(raw_datasets[0])
    dummy_example['word'] = "N/A"
    dummy_example['sentence1'] = "N/A"
    dummy_example['sentence2'] = "N/A"

    # example = raw_datasets[0]
    example = dummy_example

    # print(example)
    tempalte_names = []
    prompts = []
    print()
    for t in list(prompts_wrapper.templates.values()):
        # print("template name:", t.name)
        input, target = t.apply(example)
        # print("INPUT:", input, "\nTARGET:", target, "\n")
        prompts.append(input)
        tempalte_names.append(t.name)
    all_prompts = [prompts]
    vis_dataset = {f'{dataset_name}_{dataset_config_name}':tempalte_names}
    print(json.dumps(vis_dataset, indent=4))
