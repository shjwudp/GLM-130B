import os
import torch
import stat
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from functools import partial
from typing import List, Tuple

from SwissArmyTransformer import mpu
from evaluation.model import batch_filling_sequence
from generation import BeamSearchStrategy, BaseStrategy, ConstraintBeamSearchStrategy
from initialize import initialize, initialize_model_and_tokenizer

from flask import Flask, request, jsonify, current_app, make_response
from flask_restful import Resource, Api


class GPTGenerate(Resource):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.executor = ThreadPoolExecutor(1)

    def generate_gpt_result(self, sentences, max_len, temperature, top_p, do_sample):
        if not isinstance(sentences, str):
            return "sentences must be str", 400

        start_timestamp = time.time()
        prompt = sentences
        input_ids = self.tokenizer(
            prompt, return_tensors="pt").input_ids.cuda()
        generate_ids = self.model.generate(
            input_ids,
            max_length=max_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        result = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        resp = {
            "prompt": prompt,
            "text": result,
            "compute_time": time.time() - start_timestamp,
        }

        return resp, 200

    def put(self):
        try:
            j = request.get_json()

            start_timestamp = time.time()
            torch.distributed.broadcast_object_list([j, False])
            answers, code = process(self.model, self.tokenizer, j)
            if code != 200:
                return answers, code

            print("answers", answers)

            resp = {
                "prompt": j["context"],
                "otuput": answers,
                "compute_time": time.time() - start_timestamp,
            }

            return make_response(jsonify(resp), 200)
        except:
            traceback.print_exc()
            return "error", 400


def index():
    return current_app.send_static_file('index.html')


class GenerateServer(object):
    def __init__(self, model, tokenizer):
        self.app = Flask(__name__)
        self.app.add_url_rule('/', 'index', index)
        api = Api(self.app)
        api.add_resource(GPTGenerate, '/generate',
                         resource_class_args=[model, tokenizer])

    def run(self, url):
        self.app.run(url, debug=False)


def add_generation_specific_args(parser):
    parser.add_argument("--sampling-strategy", type=str, default="BaseStrategy", help="Type of sampling strategy.")
    parser.add_argument("--min-gen-length", type=int, default=0, help="The minimum length each blank should generate.")
    parser.add_argument(
        "--print-all-beams", action="store_true", help="Print all output generated by beam search strategy."
    )


def isEnglish(s):
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_masks_and_position_ids(seq, mask_position, max_gen_length, gmask=False):
    context_length = seq.shape[1]
    tokens = torch.nn.functional.pad(seq, (0, max_gen_length), mode='constant', value=-1)
    attention_mask = torch.ones((1, tokens.shape[-1], tokens.shape[-1]), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., : context_length - 1] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    position_ids = torch.arange(tokens.shape[-1], dtype=torch.long, device=tokens.device)
    if not gmask:
        position_ids[context_length - 1 :] = mask_position

    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids


def fill_blanks(raw_text: str, model, tokenizer, strategy) -> Tuple[List[str], List[str], List[List[str]]]:
    # add MASK
    generation_mask = "[MASK]" if "[MASK]" in raw_text else "[gMASK]"
    use_gmask = "[MASK]" not in raw_text

    mask_pattern = r"\[g?MASK\]"
    text_list = re.split(mask_pattern, raw_text)
    pattern_list = re.compile(mask_pattern).findall(raw_text)
    seq = []
    for i in range(len(pattern_list)):
        pattern = pattern_list[i]
        sub_text = text_list[i]
        seq.extend(tokenizer.tokenize(sub_text))
        seq.append(tokenizer.get_command(pattern))

    seq.extend(tokenizer.tokenize(text_list[-1]))

    if "MASK]" not in raw_text:
        seq += [tokenizer.get_command(generation_mask)]
        raw_text += " " + generation_mask
    if not raw_text.endswith("MASK]"):
        seq = seq + [tokenizer.get_command("eos")]
    if mpu.get_model_parallel_rank() == 0:
        print("\nInput: {}\n".format(raw_text))
    if len(seq) > args.max_sequence_length or len(seq) > args.out_seq_length:
        raise ValueError("text too long.")

    # generation
    is_english = isEnglish(raw_text)
    output_list = [seq]
    num_output = args.num_beams if args.sampling_strategy == "BeamSearchStrategy" or args.sampling_strategy == "ConstraintBeamSearchStrategy" else 1
    last_pos, answers, answers_with_style, blanks = (
        [0] * num_output,
        ["" for _ in range(num_output)],
        ["" for _ in range(num_output)],
        [[] for _ in range(num_output)],
    )

    # continually detect the first mark position
    while True:
        seq = output_list[0]
        # detect mask position
        mask_token = tokenizer.get_command(generation_mask)
        if mask_token not in seq:
            break
        mask_position = seq.index(mask_token)

        output_list = []

        input_seq = torch.cuda.LongTensor(
            [seq + [tokenizer.get_command("sop")]],
            device=args.device,
        )
        output, _ = batch_filling_sequence(
            model,
            input_seq,
            torch.cuda.LongTensor([input_seq.shape[-1]], device=args.device),
            strategy=strategy,
            get_masks_and_position_ids=partial(
                get_masks_and_position_ids,
                mask_position=mask_position,
                max_gen_length=args.out_seq_length - input_seq.shape[-1],
                gmask=use_gmask,
            ),
        )
        if isinstance(output, torch.Tensor):  # different strategies
            output = output.tolist()
        output = output[0]  # batch_size = 1
        output_list.extend(output)

        # clip -1s and fill back generated things into seq
        for i in range(len(output_list)):
            output = output_list[i].tolist() if isinstance(output_list[i], torch.Tensor) else output_list[i]
            try:
                unfinished = output.index(-1)
            except ValueError:
                unfinished = len(output)
            if output[unfinished - 1] in strategy.end_tokens:
                unfinished -= 1
            bog = output.index(tokenizer.get_command("sop"))

            prefix = tokenizer.detokenize(output[last_pos[i] : mask_position])
            blank = tokenizer.detokenize(output[bog + 1 : unfinished])
            answers_with_style[i] += (
                prefix
                + (" " if is_english else "")
                + ("\033[4m" if use_gmask else "\x1b[0;32m\033[4m")
                + blank
                + ("\033[0m" if use_gmask else "\033[0m\x1b[0m")
                + (" " if is_english else "")
            )
            blanks[i].append(blank)
            last_pos[i] = mask_position + unfinished - (bog + 1)
            output_list[i] = output[:mask_position] + output[bog + 1 : unfinished] + output[mask_position + 1 : bog]

    for i, output in enumerate(output_list):
        if output[-1] == tokenizer.get_command("eos"):
            output = output[:-1]
        answers_with_style[i] += tokenizer.detokenize(output[last_pos[i] :])
        answers[i] = tokenizer.detokenize(output)

    return answers, answers_with_style, blanks


def process(model, tokenizer, request):
    j = request
    args.sampling_strategy = j["sampling_strategy"]
    sentences = j["context"]
    args.out_seq_length = j.get("out_seq_length", 256)

    end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
    if args.sampling_strategy == "BaseStrategy":
        max_length = j["max_len"]
        args.max_sequence_length = max_length
        temperature = j["temperature"]
        top_k = j["top_k"]
        top_p = j["top_p"]
        strategy = BaseStrategy(temperature=temperature, top_k=top_k, top_p=top_p, end_tokens=end_tokens)
    elif args.sampling_strategy == "BeamSearchStrategy":
        batch_size = j.get("batch_size", 1)
        args.num_beams = j["num_beams"]
        length_penalty = j["length_penalty"]
        no_repeat_ngram_size = j["no_repeat_ngram_size"]
        min_gen_length = j["min_gen_length"]
        strategy = BeamSearchStrategy(
            batch_size,
            args.num_beams,
            length_penalty=length_penalty,
            consider_end=True,
            end_tokens=end_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_gen_length=min_gen_length,
        )
    elif args.sampling_strategy == "ConstraintBeamSearchStrategy":
        batch_size = j.get("batch_size", 1)
        args.num_beams = j["num_beams"]
        length_penalty = j["length_penalty"]
        no_repeat_ngram_size = j["no_repeat_ngram_size"]
        min_gen_length = j["min_gen_length"]
        deterministic = j.get("deterministic", False)
        forces_output = j.get("forces_output", [])
        record_the_inference_process = j.get("record_the_inference_process", False)
        forces_output = [
            tokenizer.tokenize(word) + [end_tokens[0]] for word in forces_output
        ]
        strategy = ConstraintBeamSearchStrategy(
            batch_size,
            args.num_beams,
            length_penalty=length_penalty,
            consider_end=True,
            end_tokens=end_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_gen_length=min_gen_length,
            forces_output=forces_output,
            deterministic=deterministic,
            record_the_inference_process=record_the_inference_process,
        )
    else:
        return f"unknown strategy {args.sampling_strategy}", 400

    try:
        answers, answers_with_style, blanks = fill_blanks(sentences, model, tokenizer, strategy)
    except Exception as ex:
        return f"fill_blanks failed, ex={ex}", 400

    try:
        beam_scores = torch.tensor(strategy.returned_beam_scores[0]) # batch_idx = 0
        beam_probs = torch.nn.functional.softmax(beam_scores, dim=-1).tolist()
        answers = list(zip(answers, beam_probs))
    except:
        pass

    return answers, 200


def main(args):
    model, tokenizer = initialize_model_and_tokenizer(args)

    if torch.distributed.get_rank() == 0:
        generate_server = GenerateServer(model, tokenizer)

        generate_server.run("0.0.0.0")
    else:
        while True: 
            request, is_stop = {}, False
            info = [request, is_stop]
            try:
                torch.distributed.broadcast_object_list(info)
            except:
                continue
            request, is_stop = info

            if is_stop:
                return

            process(model, tokenizer, request)


if __name__ == "__main__":
    args = initialize(extra_args_provider=add_generation_specific_args)

    with torch.no_grad():
        main(args)
