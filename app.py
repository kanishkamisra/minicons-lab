from flask import Flask, render_template, request, jsonify
from threading import Lock
from nltk.tokenize import TweetTokenizer
import traceback
import os
import argparse

app = Flask(__name__, template_folder="templates", static_folder="static")

# Lazy model cache
cache = {"model": None, "scorer": None}
cache_lock = Lock()

def determine_bos(model_name: str) -> bool:
    if not model_name:
        return False
    return any(x in model_name for x in ("gpt2", "pythia", "SmolLM"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/score", methods=["POST"])
def score():
    data = request.get_json()
    sequences_text = data.get("sequences", "")
    model_name = data.get("model", "")
    device = data.get("device", "cpu")
    granularity = data.get("granularity", "word")
    surprisal = bool(data.get("surprisal", True))
    base_two = bool(data.get("base_two", True))
    rank = bool(data.get("rank", False))
    parsed_agg = data.get("parsed_agg", "sum")
    bos_explicit = data.get("bos", None)

    sequences = [s.strip() for s in sequences_text.splitlines() if s.strip()]
    if not sequences:
        return jsonify({"error": "No sequences provided"}), 400

    try:
        from minicons import scorer
    except Exception:
        return jsonify({"error": "could not import minicons.scorer", "trace": traceback.format_exc()}), 500

    # Determine BOS if not provided explicitly
    if bos_explicit is None:
        BOS = determine_bos(model_name)
    else:
        BOS = bool(bos_explicit)

    # Load or reuse scorer
    with cache_lock:
        if cache["model"] != model_name or cache["scorer"] is None:
            cache["model"] = model_name
            cache["scorer"] = scorer.IncrementalLMScorer(model_name, device=device)
        lm = cache["scorer"]

    word_tokenizer = TweetTokenizer().tokenize

    results = []
    for seq in sequences:
        if granularity == "token":
            # token-level: pass a list of raw strings so scorer returns per-sequence lists
            seqs = [seq]
            out = lm.token_score(seqs, bos_token=BOS, surprisal=surprisal, base_two=base_two, rank=rank)
            seq_res = []
            if isinstance(out, list) and len(out) > 0:
                seq_out = out[0]
                for i, t in enumerate(seq_out):
                    if isinstance(t, dict):
                        token = t.get("token") or t.get("text") or str(i)
                        value = t.get("surprisal") if surprisal else t.get("logprob")
                        r = {"token": token, "value": value, **{k:v for k,v in t.items() if k not in ("token","surprisal","logprob")}}
                    elif isinstance(t, (list, tuple)) and len(t) >= 2:
                        token, value = t[0], t[1]
                        r = {"token": token, "value": value}
                        # include positional extras (common: rank at index 2)
                        if len(t) >= 3:
                            try:
                                r["rank"] = t[2]
                            except Exception:
                                r["extra_2"] = t[2]
                        # include any further extras as extra_N
                        if len(t) > 3:
                            for j in range(3, len(t)):
                                r[f"extra_{j}"] = t[j]
                    else:
                        r = {"token": str(i), "value": None}
                    seq_res.append(r)
            else:
                seq_res = []
        elif granularity == "parsed":
            # parsed granularity: combine bracketed spans into single items by summing word scores
            # remove bracket markers before scoring so brackets are not treated as tokens
            seq_clean = seq.replace("[", "").replace("]", "")
            seqs_tokenized = [seq_clean]
            out = lm.word_score_tokenized(seqs_tokenized, bos_token=BOS, tokenize_function=word_tokenizer, surprisal=surprisal, base_two=base_two)
            seq_res = []
            if isinstance(out, list) and len(out) > 0:
                seq_out = out[0]
                # extract token texts and values
                token_texts = []
                token_values = []
                token_extras = []
                for t in seq_out:
                    if isinstance(t, dict):
                        text = t.get("token") or t.get("text") or ""
                        value = t.get("surprisal") if surprisal else t.get("logprob")
                        extras = {k:v for k,v in t.items() if k not in ("token","surprisal","logprob")}
                    elif isinstance(t, (list, tuple)) and len(t) >= 2:
                        text, value = t[0], t[1]
                        extras = {}
                        if len(t) >= 3:
                            extras["rank"] = t[2]
                    else:
                        text, value, extras = str(t), None, {}
                    token_texts.append(text)
                    token_values.append(value)
                    token_extras.append(extras)

                # find bracketed substrings in original seq (raw content between brackets)
                bracket_ranges = []  # list of (start_idx, end_idx, raw_substring)
                idx = 0
                while True:
                    start = seq.find('[', idx)
                    if start == -1:
                        break
                    end = seq.find(']', start+1)
                    if end == -1:
                        break
                    raw = seq[start+1:end]
                    bracket_ranges.append((start, end, raw))
                    idx = end+1

                # for each bracket raw substring, tokenize and locate in token_texts
                matched_token_ranges = []  # list of (tok_start, tok_end, raw)
                used = [False]*len(token_texts)
                for (_, _, raw) in bracket_ranges:
                    if not raw.strip():
                        continue
                    btokens = word_tokenizer(raw)
                    if not btokens:
                        continue
                    # normalize for matching
                    norm_b = [b.lower() for b in btokens]
                    norm_tokens = [t.lower() for t in token_texts]
                    found = False
                    for i in range(0, len(norm_tokens)-len(norm_b)+1):
                        if any(used[i+j] for j in range(len(norm_b))):
                            continue
                        match = True
                        for j in range(len(norm_b)):
                            if norm_tokens[i+j] != norm_b[j]:
                                match = False
                                break
                        if match:
                            # store raw exactly as inside brackets (no brackets included)
                            matched_token_ranges.append((i, i+len(norm_b)-1, raw))
                            for j in range(len(norm_b)):
                                used[i+j] = True
                            found = True
                            break
                    # if not found, skip

                # build seq_res merging matched ranges
                i = 0
                while i < len(token_texts):
                    m = next((r for r in matched_token_ranges if r[0] == i), None)
                    if m is not None:
                        s_idx, e_idx, raw = m
                        # sum values for the range
                        total = 0.0
                        any_val = False
                        for k in range(s_idx, e_idx+1):
                            v = token_values[k]
                            if isinstance(v, (int, float)):
                                total += float(v)
                                any_val = True
                        if any_val:
                            if parsed_agg == "mean":
                                count = (e_idx - s_idx + 1)
                                val = (total / count) if count > 0 else None
                            else:
                                val = total
                        else:
                            val = None
                        # label without bracket markers (brackets are only indicators)
                        seq_res.append({"token": raw, "value": val})
                        i = e_idx+1
                    else:
                        # single token
                        v = token_values[i]
                        seq_res.append({"token": token_texts[i], "value": v, **token_extras[i]})
                        i += 1
            else:
                seq_res = []
        else:
            # word-level: pass raw strings (scorer will call tokenize_function on each)
            seqs_tokenized = [seq]
            out = lm.word_score_tokenized(seqs_tokenized, bos_token=BOS, tokenize_function=word_tokenizer, surprisal=surprisal, base_two=base_two)
            # out assumed to be list of sequences
            seq_res = []
            if isinstance(out, list) and len(out) > 0:
                seq_out = out[0]
                for t in seq_out:
                    if isinstance(t, dict):
                        token = t.get("token") or t.get("text")
                        value = t.get("surprisal") if surprisal else t.get("logprob")
                        seq_res.append({"token": token, "value": value, **{k:v for k,v in t.items() if k not in ("token","surprisal","logprob")}})
                    elif isinstance(t, (list, tuple)) and len(t) >= 2:
                        token, value = t[0], t[1]
                        seq_res.append({"token": token, "value": value})
                    else:
                        seq_res.append({"token": str(t), "value": None})
            else:
                seq_res = []

        results.append({"sequence": seq, "scores": seq_res})

    return jsonify({"results": results})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Minicons Lab Flask app")
    parser.add_argument('--port', '-p', type=int, default=int(os.environ.get('PORT', 8888)), help='Port to listen on')
    parser.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'), help='Host to bind to')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Enable debug mode (default: disabled)')
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
