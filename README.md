## Logit threshold sampling for coherent creativity
(with many thanks to [exllamav2](https://github.com/turboderp/exllamav2/tree/master))

In language model text generation, adjusting the temperature parameter affects the randomness of the output:
* Low temperature produces highly probable and repetitive token selections, leading to coherent but less diverse and potentially uninteresting text.
* High temperature increases randomness by making less probable tokens more likely, enhancing diversity but often at the expense of coherence, introducing errors or nonsensical content.

**Logit threshold sampling** addresses this trade-off by:
* Filtering out low-coherence tokens: It removes tokens with low absolute logit values (below a minimum threshold), which the model is less confident about and are likely to disrupt coherence if selected.
* Applying high temperature to high-confidence tokens: Among the remaining tokens, those with logits above a secondary temperature threshold are subjected to higher temperature scaling. This enhances creativity among the tokens the model is most confident about, without introducing incoherence. Those remaining tokens below this threshold are not eliminated, but neither do they increase their existing probabilities.

### Primary Applications
* Coherent creativity in outputs
  * Finely tune the level of creativity in generated text. By applying higher temperatures selectively, the sampler allows for diverse word choices among tokens the model deems highly coherent.
* Enhancing agent diversity
  * In systems where multiple agents use the same language model, this sampler introduces variability in their responses. Each agent can generate unique yet coherent outputs, enabling more sophisticated interactions between differing perspectives.
 
### Comparison with min-p + temperature last
**Logit threshold sampling** uses the raw logits to make decisions, preserving the model's absolute confidence in each token, and allowing for more nuanced filtering based on how confident the model is about each token.

**Min-p** operates on normalised probabilities, where the absolute confidence information has already been lost due to softmax normalization. This means it can only consider tokens based on their probability relative to others, not on the absolute confidence the model has in them. Applying temperature after min-p helps, but runs the risk of false-positives if incoherent tokens have made it past the min-p, or missing out on good, viable choices below min-p, when there are many such choices available.

### Parameters
* **min_threshold** - The lowest model logit considered for selection. If the model outputs logits which all fall below this threshold, the threshold is adjusted downward to ensure that the highest logit token is still available for selection.
  * For Gemma 2 9B, a good starting point is min_threshold=12.0
  * Tune this up or down to increase/decrease the coherence floor
* **temp_threshold** - The threshold above which logits will be subjected to temperature scaling, based on the assumption that they are highly coherent choices. If this is equal to or lower than min_threshold, it becomes equivalent to min_threshold.
  * For Gemma 2 9B, a good starting point is temp_threshold=16.0
  * Tune this up or down to increase/decrease the coherence floor
* **temperature** - The temperature applied to logits above temp_threshold. If the thresholds are set appropriately this can be increased to 10 or beyond without detrimental effects on coherence. Once low-coherence tokens are filtered out, increasing this value makes the choice of a remaining coherent token more random.
  * Set this lower for more deterministic outputs, as per standard use
  * Set this higher for more creative outputs, temperature=10.0 is a good starting point for creative outputs on Gemma 2 9B, with the above thresholds set
* **logit_threshold_stats** - This is a boolean True/False (defaulting to False), which displays logic stats during token generation. Use this to get a sense of how the logit threshold sampler is affecting token selection.

**Important** - Because this sampler works directly on provided model logits, settings need to be manually tuned for each new model used. The typical tuning process looks like this:
1. Set logit_threshold_stats to True, and min_threshold to a conservative value (around 8-10 seems to be good, but your mileage may vary)
2. Observe the output statistics in a typical generation, and adjust min_threshold upward until the number of tokens left after filtering to range between 1-200, falling mostly between 1 and 20.
3. Set temp_threshold to roughly one standard deviation (as per the output statistics) above the typical mean remaining logit.
4. Adjust temparture according to your use-case
5. Set logit_threshold_stats to False once you are comfortable that the sampler is working as desired

### Minimal example
```python
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler, ExLlamaV2DynamicJob

model_dir = "models/gemma-2-9b-it-exl2-6bpw/"

config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 4096, lazy = True)
model.load_autosplit(cache, progress = True)
tokenizer = ExLlamaV2Tokenizer(config)

generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)

prompts = []

prompt_insert = 'Once upon a time,'

# Using gemma 2 chat format
prompt = f"""
<start_of_turn>user
Write at least 500 words, beginning with and in the same style as the following:
{prompt_insert}<end_of_turn>
<start_of_turn>model
{prompt_insert}"""

prompts.append(prompt)

# Set up sampler settings - tuned for Gemma2-9B
settings = ExLlamaV2Sampler.Settings()
settings.logit_threshold_stats = False
settings.temperature = 10.0
settings.min_threshold = 12.0
settings.temp_threshold = 16.0


for idx, prompt_str in enumerate(prompts):
    job = ExLlamaV2DynamicJob(
        input_ids=tokenizer.encode(prompt_str, add_bos=True),
        max_new_tokens=500,
        stop_conditions=[tokenizer.eos_token_id],
        gen_settings=settings,
        identifier=idx,
    )
    generator.enqueue(job)

collected_outputs = [""] * len(prompts)

while generator.num_remaining_jobs():
    results = generator.iterate()
    for result in results:
        idx = result["identifier"]
        text_chunk = result.get("text", "")
        collected_outputs[idx] += text_chunk

print()
for idx, o in enumerate(collected_outputs):
    print(f'{prompt_insert}{o}')
```
