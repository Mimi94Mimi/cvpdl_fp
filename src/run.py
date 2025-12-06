import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
import asyncio
import random

from utils import get_options, get_chunk

from policies import policy_map  # Use full import path for policy_map

async def create_sample(args):
    row, method_args, round_idx = args
    QuestionSample = policy_map[method_args.method_name]
    return QuestionSample(row, method_args, round_idx)

async def process_sample(sample):
    return await sample.process()

async def eval_model(args):
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    # Prepare sample arguments
    sample_args = []
    rows_as_dicts = questions.to_dict(orient="records")
    for row in rows_as_dicts:
        if args.all_rounds:
            # do not use this 
            num_rounds = len(get_options(row, ['A', 'B', 'C', 'D']))
        else:
            num_rounds = 1
            
        for round_idx in range(num_rounds):
            sample_args.append((row, args, round_idx))
    
    # Generate samples using coroutines
    samples = []
    if args.debug: 
        sample_args = [random.choice(sample_args)]
    else:
        random.seed(42)  # Fix random seed
        sample_args = [random.choice(sample_args) for _ in range(1)]
        
    tasks = [create_sample(args) for args in sample_args]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Creating samples"):
        sample = await coro
        samples.append(sample)
    
    # Process samples using coroutines
    results = []
    tasks = [process_sample(sample) for sample in samples]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing samples"):
        result = await coro
        results.append(result)

    # Write results
    with open(answers_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Collect and write flat tree info if tree_file is specified
    if args.tree_file:
        tree_file = os.path.expanduser(args.tree_file)
        os.makedirs(os.path.dirname(tree_file), exist_ok=True)
        
        # Collect all flat tree info from samples
        all_flat_nodes = []
        for sample in samples:
            # Check if sample has root node (MCTS-based methods)
            if hasattr(sample, 'root') and sample.root is not None:
                # Get flat tree info
                flat_tree_info = sample.serialize_tree_flat(sample.root)
                # Extend the list (not append) to create one big flat list
                all_flat_nodes.extend(flat_tree_info)
        
        # Write all nodes to file
        if all_flat_nodes:
            with open(tree_file, "w") as f:
                json.dump(all_flat_nodes, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--tree-file", type=str, default=None, help="Output file for MCTS tree nodes (flat list)")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--method_name", type=str, default="common")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--iou-threshold", type=float, default=-1, help="IoU threshold for node similarity detection in MCTS. Set to -1 to disable similarity check (default: -1)")
    parser.add_argument("--max-consecutive-repeats", type=int, default=-1, help="Maximum consecutive repeat_question actions allowed. Set to -1 to disable this limit (default: -1)")
    
    def str2bool(v):
        return v.lower() == 'true'
    
    parser.add_argument("--debug", type=str2bool, help="debug mode", default=False)
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugpy connection...")
        debugpy.wait_for_client()
        print("Breakpoint stopped here, ready for debugging...")
        debugpy.breakpoint()
    
    asyncio.run(eval_model(args))