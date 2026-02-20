import argparse
import json
import os
import numpy as np
import multiprocessing as mp
from typing import Iterator
from tqdm import tqdm

from .tokenizer import Tokenizer

def data_generator(file_path: str, start_byte: int, end_byte: int) -> Iterator[str]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(start_byte)
        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break
            yield line

def worker_fn(worker_id, start_byte, end_byte, args, progress_queue):
    with open(args.config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, **config['tokenizer'])
    
    token_buffer = []
    with open(args.input_path, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(start_byte)
        last_pos = start_byte
        
        while f.tell() < end_byte:
            line = f.readline()
            if not line: break

            token_buffer.extend(tokenizer.encode(line))
            
            current_pos = f.tell()
            delta = current_pos - last_pos
            progress_queue.put((worker_id, delta, None))
            last_pos = current_pos

    progress_queue.put((worker_id, None, np.array(token_buffer, dtype=np.uint32)))

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--vocab_path', type=str, required=True)
parser.add_argument('--merges_path', type=str, required=True)
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--n_proc', type=int, default=4)
args = parser.parse_args()


with open(args.input_path, "rb") as f:
    from .train_bpe import find_chunk_boundaries, SPLIT_SPECIAL_TOKEN
    boundaries = find_chunk_boundaries(f, args.n_proc, SPLIT_SPECIAL_TOKEN.encode('utf-8'))

manager = mp.Manager()
progress_queue = manager.Queue()
processes, bars = [], []

for i in range(args.n_proc):
    p = mp.Process(target=worker_fn, args=(i, boundaries[i], boundaries[i+1], args, progress_queue))
    p.start()
    processes.append(p)
    
    bar = tqdm(total=boundaries[i+1] - boundaries[i], unit='B', unit_scale=True, desc=f"Worker {i:02d}", position=i, leave=True)
    bars.append(bar)

finished_count = 0
results = {}

while finished_count < args.n_proc:
    worker_id, delta, arr = progress_queue.get()
    if arr is not None:  
        results[worker_id] = arr
        finished_count += 1
    elif delta is not None:
        bars[worker_id].update(delta)

for p in processes:
    p.join()
    
for bar in bars:
    bar.close()
    
output_dir = os.path.dirname(args.output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

total_length = sum(len(arr) for arr in results.values())
mmap_arr = np.lib.format.open_memmap(args.output_path, mode='w+', dtype=np.uint32, shape=(total_length,))

current_idx = 0
for i in range(args.n_proc):
    arr_len = len(results[i])
    mmap_arr[current_idx : current_idx + arr_len] = results[i]
    current_idx += arr_len
    del results[i]

mmap_arr.flush()
del mmap_arr

print("\n" * args.n_proc)
print("Done!")