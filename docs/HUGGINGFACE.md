# Host BeerLLM on Hugging Face

These steps use this laptop's saved Helicious Helles model. The standalone model has
been prepared locally at `outputs/helles-hf-v1` (about 6.2 GB). Nothing has been
uploaded and no paid endpoint has been created.

## 1. Open the project and prepare the upload folder

```bash
cd "/Users/nh/Desktop/Naisarg/Github/Fine Tune LLM"
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m scripts.prepare_hf_model
```

This command is safe to repeat: it verifies the completed export and returns without
merging again. It refuses changed or incomplete exports. The original adapter and
checkpoints remain untouched. Rebuilding from the saved adapter needs at least 8 GiB
free disk space; a different export requires a new `--output` directory.

The export uses the pinned Qwen base and merges our LoRA into full float32 weights.
Three next-token logit comparisons pass, including unchanged predicted next tokens.
This numerical check is not the twelve-case hosted evaluation.

## 2. Create an account and access token

Sign in at [Hugging Face](https://huggingface.co/join), then open
[Access Tokens](https://huggingface.co/settings/tokens). Create a **Write** token for
uploading your model. Paste it into Terminal when this command asks:

```bash
.venv/bin/hf auth login
```

You can answer `n` if asked to add the token to Git credentials. The upload and API
test use Hugging Face's saved login. No token belongs in the project files or chat.

## 3. Upload the model

Replace `YOUR_USERNAME` with your Hugging Face username, then run:

```bash
.venv/bin/hf upload YOUR_USERNAME/beerllm-helles outputs/helles-hf-v1 \
  --repo-type model --private \
  --commit-message "Upload merged Helicious Helles model"
```

Allow the several-gigabyte upload to finish. This creates a private model repository
if the name is new. For an existing repository, `--private` does not change its
visibility; check the repository settings before using an existing name.
The upload folder contains full weights, tokenizer, base-model license, model card,
the saved teaching instruction and export hashes. It contains no training corpus
or credentials.

## 4. Start the hosted model

Open your new model page, then choose **Deploy → Inference Endpoints (dedicated)**.
Add the payment method/credits Hugging Face requests. Use these settings:

| Setting | Value |
|---|---|
| Model repository | `YOUR_USERNAME/beerllm-helles` |
| Task | Text Generation |
| Inference engine/container | vLLM |
| Hardware | One NVIDIA L4 GPU, in an available region |
| Access | Private/authenticated; requires your HF token |
| Maximum replicas | 1 |
| Minimum replicas | 0 |
| Scale to zero after inactivity | 15 minutes |

Under advanced container arguments, set `max_model_len` to `2048` and
`max_num_seqs` to `1`. Leave dtype on `auto` and quantization unset. The runtime may
use a different precision from local float32, so hosted answers need checking.

**Review the displayed hourly price before clicking Create Endpoint.** This starts
paid hosting. Wait until the endpoint says **Running**. On **Overview**, copy the
**Endpoint URL**.

The zero-replica setting reduces idle cost. After sleeping, the endpoint can take
minutes to wake up. Keeping a minimum of one replica instead keeps it warm and
billed while running. You can also pause the endpoint after testing.

## 5. Ask the API a question

Replace `YOUR_ENDPOINT_URL` with the HTTPS URL copied from Overview:

```bash
.venv/bin/python -m scripts.query_hf_endpoint \
  --endpoint "YOUR_ENDPOINT_URL" \
  --prompt "Explain the complete Helicious Helles recipe in simple language."
```

The helper discovers the engine's model name from `/v1/models`, sends a request to
`/v1/chat/completions`, and prints the answer. It uses your saved HF login and
automatically includes our original system instruction with greedy decoding.
It also accepts an Endpoint URL ending in `/v1`.

This is a single-question API test. For a conversation, an application must send
earlier user/assistant messages with each new question. The engine does not read
our custom `beerllm.json` instruction automatically; clients must send it.

## 6. Check the hosted behavior

Use the questions in `configs/helles-fresh-confirmation.jsonl` and the six-turn
protocol in `configs/helles-fresh-dialogue.json`. Compare complete answers against
[the rubric](../configs/helles-rubric.json) and [the local report](../reports/single-recipe-run.md).
Those prompts are now regression cases; they are no longer untouched final tests.
The local adapter's 12/12 score does not automatically transfer to the merged GPU
deployment. Record hosted output and timing before claiming the same quality or speed.

Local verification: 84 software tests pass; the export's three numerical checks
pass and its original adapter files remain unchanged. The API helper has mocked
HTTP tests. An actual hosted request remains untested until an endpoint is created.

Official references: [LoRA merging](https://huggingface.co/docs/peft/v0.13.0/en/package_reference/lora#peft.LoraModel.merge_and_unload),
[create an endpoint](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint),
[vLLM engine settings](https://huggingface.co/docs/inference-endpoints/en/engines/vllm),
[idle scaling](https://huggingface.co/docs/inference-endpoints/en/guides/autoscaling),
[OpenAI-compatible serving](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).
