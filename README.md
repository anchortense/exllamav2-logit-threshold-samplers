## Logit samplers for coherent creativity
(with many thanks to [exllamav2](https://github.com/turboderp/exllamav2/tree/master))

Two new samplers enabling coherent diverse text generation.
*  The **logit threshold sampler (LTS)** filters low-confidence logits and enables the application of much higher temperatures to stronger candidates, generating varied outputs without losing coherence.
*  The **confidence breaker sampler (CBS)** addresses repetitive text sequences by dynamically detecting these patterns on the logit level, allowing the model to generate more diverse responses.

The key innovation is to use absolute logit values instead of softmax probabilities to retain the model’s raw confidence in each token, allowing more precise evaluation and filtering.

### Proof of concept
This repository contains a proof of concept implementation of the above samplers, forked from exllamav2 v0.2.3.

These implementations have not been optimised for efficiency (and run at around 70-80% speed), or integrated with other existing samplers beyond temperature, such as min-p, top-k, etc.

### Recommended install process
Create and activate a new environment, pip install an official exllamav2 wheel from [here](https://github.com/turboderp/exllamav2/releases/tag/v0.2.3) and then swap in the exllamav2/generator folder from this fork.

### Minimal example

```python
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler, ExLlamaV2DynamicJob

model_dir = "models/gemma-2-9b-it-exl2-6bpw/"  # Set to model path

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
settings.temperature = 10.0

settings.logit_threshold_stats = False
settings.min_threshold = 12.0
settings.temp_threshold = 16.0

settings.confidence_breaker_debug = True
settings.confidence_breaker = 8
settings.mid_threshold = 15.0
settings.high_threshold = 22.0

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

## Example outputs

Prompt: "Write at least 500 words, beginning with and in the same style as the following:\nOnce upon a time,"

Model: gemma-2-9b-it-exl2-6bpw

### Baseline output for comparison
>Once upon a time, nestled amidst rolling emerald hills and whispering willow trees, lay the quaint village of Willow Creek. Its cobblestone streets, lined with gingerbread cottages and blooming flower boxes, echoed with the laughter of children and the gentle murmur of gossip exchanged over steaming mugs of tea. The villagers lived simple lives, their days measured by the rising and setting of the sun, their hearts warmed by the close-knit community that bound them together.
>
>The heart of Willow Creek was its ancient oak, a towering sentinel that had witnessed generations come and go. Beneath its sprawling branches, stories were shared, dreams were spun, and secrets whispered on the wind. It was here, on a crisp autumn afternoon, that Elara, a young woman with eyes the color of twilight and hair like spun moonlight, sat sketching her beloved village.
>
>Elara possessed a gift, a rare talent for capturing the essence of things in her art. Her drawings weren't mere depictions; they breathed with life, pulsating with the spirit of the subject. She could capture the mischievous twinkle in a child's eye, the weariness etched beneath a farmer's brow, the quiet dignity of the aging shepherdess tending her flock.
>
>[...]

settings.temperature = 1.0

Note: If you run this 10 times you will usually get some minor variant on the above


### Baseline output for comparison (with min-p + temperature last)
>Once upon a time, nestled between rolling emerald hills and a shimmering sapphire sea, lay the kingdom of Atheria.  This wasn't any ordinary kingdom, for it thrived under the rule of not a king or queen, but a council of twelve Wise Weavers. Each Weaver possessed the unique ability to manipulate the very fabric of reality, weaving threads of light into blooming roses, stitching clouds together to form fantastical shapes, and even mending broken hearts with invisible stitches of empathy.
>
>The Weavers, each representing a different aspect of life – from the boundless energy of Dawn to the quiet wisdom of Twilight – governed Atheria with a balance so delicate that the kingdom was known throughout the land for its harmony and prosperity. But peace, even in the most idyllic havens, rarely lasts forever. A shadow began to creep across Atheria’s radiant landscape, whispering of discontent and discord.
>
>It started subtly. A rose, inexplicably thorny; a storm brewing on a cloudless day; a whispered word of doubt amongst the usually jovial villagers. The Weavers, attuned to the subtle shifts in Atheria’s very essence, grew alarmed. They gathered in their ancient council chamber, the air thick with apprehension.
>
>[...]

settings.temperature = 1.5

settings.min_p = 0.025

settings.temperature_last = True


### Logit threshold sampler
>Once upon a time, nestled deep within a whispering wood of silver oaks and jade maple, stood a tiny gingerbread house. Tiny, yes, but sturdy.  For it wasn't some whimsey spun by an overzealous oven, but a cottage meticulously built by Gidget the Gnome, her tiny hammer swinging like the blade of a hungry hornet, and her icing skillfully tinted the colour of warm sunrise. It sat atop a rise, commanding a view of sparkling streams and toadstool pastures.  Life was sweet for Gidget.
>
>Her mornings were filled with the scent of freshly baked wild mushroom crumpets, and the afternoon sun dripped gold through the leaded window panes, bathing her living room in a mosaic of shimmering light. Every evening, she'd settle by the hearth, her old clockwork spider whirring merrily beside her as she tended to her pot of luminous lichen tea, its faint blue glow casting an ethereal glow on her cheery gnome face.
>
>One particularly crisp autumn day, as Gidget stirred her dough, a frantic thump-thump-thumping against the front door shook the entire cottage. She peered through the sugary window, heart thudding like a beetle caught in a shoe.  It was Pip, a young squirrel with fur the colour of sunset and eyes bright with urgency.
>
>[...]

settings.temperature = 10.0

settings.temp_threshold = 16.0

settings.min_threshold = 12.0


### Confidence breaker
>Once upon a time, nestled between rolling emerald hills and a sparkling sapphire coast, lay the forgotten kingdom of Atheria, hidden from the world by ancient enchantments. It once flourished with life; bustling marketplaces teeming  were filled with vibrant silks from the Eastern Kingdoms, delicate glassware shimmering after sunrise and spiced cakes warming every hand. The streets echoed with laughter as artisans and merchants plied their crafts and children skipped merrily amidst cobblestone squares. But all this faded like an ancient tapestry with every passing year as Atheria fell into obscurity. The once majestic palace, now cloaked by thorny vines and whispered curses, stands sentinel amidst the silent forest, watching as time unravelled the kingdom upon itself and stole memories with its endless tide of ages
>
>Legend whispers that the Queen's Tears, mystical orbs radiating celestial beauty, once granted Atheria its enduring prosperity. Each teardrop formed as the queen grievfed her beloved husband's passing, shimmering and pulsing with a light said even angels couldn’t withstand. However, as centuries drifted into a slumber, the legend twisted. Some claimed a greedy Duke had seized the tears for himself during the Kingdom's last days, hoarding their light in secret and leaving Atheria shrouded in an enduring darkness. Others muttered of a terrible curse laid by an unknown witch, her anger festering over perceived slights. All knew, though, that Atheria wouldn't awaken from its slumber until the light of the Queen's Tear, the source of the kingdom's fortune and prosperity returned home.
>
>[...]

settings.temperature = 1.5

settings.confidence_breaker = 8

settings.mid_threshold = 15.0

settings.high_threshold = 22.0


### Logit threshold sampler + Confidence breaker
>Once upon a time, nestled amidst the swirling turquoise tendrils of a starflower mist in a pocket of iridescent light beyond our sun’s warm caress, lived a tinkling. An unseen wanderer of whispered harmonies and ethereal polkadotes of light. She was born of a sunrise sigh and dreamt in echoes, her body a luminous melody unfurled across a cosmos of dreaming stardust, forever searching, forever changing, forever alight.
>
>Her name, though even whispering it splinters into cosmic echoes and fades into starlight dust before you grasp it, translated roughly to “Wanderlust of Morningdew”. That much you are certain of because once, when she touched a curious artifact shaped like a conch, formed from a frozen note of forgotten song, a voice bloomed, singing in a thousand tongues, “Wanderlust, dawn's jewel, sought of stardust." That was the only sound she ever made, that shimmering name whispered in the silence between starbeats, yet its message resonating throughout her journey.
>
>[...]

settings.temperature = 10.0

settings.temp_threshold = 16.0

settings.min_threshold = 12.0

settings.confidence_breaker = 8

settings.mid_threshold = 15.0

settings.high_threshold = 22.0


## Explanation: Logit threshold sampler (LTS)

In language model text generation, adjusting the temperature parameter affects the randomness of the output:
* Low temperature produces highly probable and repetitive token selections, leading to coherent but less diverse and potentially uninteresting text.
* High temperature increases randomness by making less probable tokens more likely, enhancing diversity but often at the expense of coherence, introducing errors or nonsensical content.

The **logit threshold sampler** addresses this trade-off by:
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
  * Set this higher for more creative outputs, temperature=4.0 is a good starting point for creative outputs on Gemma 2 9B, with the above thresholds set, but feel free to push it to 10 or beyond.
* **logit_threshold_stats** - This is a boolean True/False (defaulting to False), which displays logic stats during token generation. Use this to get a sense of how the logit threshold sampler is affecting token selection.

**Important** - Because this sampler works directly on provided model logits, settings need to be manually tuned for each new model used. The typical tuning process looks like this:
1. Set logit_threshold_stats to True, and min_threshold to a conservative value (around 8-10 seems to be good, but your mileage may vary)
2. Observe the output statistics in a typical generation
3. Adjust min_threshold upward until the number of tokens left after filtering ranges roughly between 1 and 200, falling mostly between 1 and 20.
4. Set temp_threshold to roughly one standard deviation (as per the output statistics) above the typical mean remaining logit.
5. Adjust temparture according to your use case
6. Set logit_threshold_stats to False once you are comfortable that the sampler is working as desired

## Explanation: Confidence breaker sampler (CBS)
Current generation language models are well known for producing certain cliched phrases, which would not necessarily be problematic in a single instance, but are known to be produced repeatedly in response to varied prompts. This is the so-called ai-slop problem. In 'deterministic' use cases this is usually not an issue, as we are simply looking for the one correct answer. In scenarios where engaging, diverse language choices are valued, ai-slop represents a significant limitation.

Current approaches to resolving this issue involve user defined lists of banned strings. This can be reasonably effective, however fail to address the deeper issue, which is the tendency of language models to funnel their responses into unintentionally learned 'tram-track' token sequences, where token choices are strongly conditioned by their immediate predecessors in a manner which is not logically or grammatically implied by the prompt or by those preceding choices.

This is a far deeper and more pervasive problem than the well known handful of phrases commonly associated with the idea of ai-slop. These tram-tracks exist within trained models because the tokens within them are good predictors for text completion. Nevertheless a user passing either the same or similar prompts repeatedly, looking for diverse outputs, will quickly observe that model responses which seemed initially impressive are in fact tram-track patterns, and the apparent diversity of outputs is an illusion.

The **confidence breaker** addresses this issue by looking for logit patterns that signal we have entered a tram-track, and extending exllamav2's banned string functionality to roll-back to the token directly before we entered the tram-tracks. The tram-tracked tokens are then discarded and replaced by a novel generation, which will take us down a different, less travelled path.

Based on empirical observation of logits and the conditions for the appearance of tram-tracks, the pattern that the confidence breaker looks for to identify these tram-tracks is a sequence of mid-high valued logits, logits which have been nudged higher than a good score by over-training, but which are not yet so guaranteed as logical or grammatical necessity.

![image](https://github.com/user-attachments/assets/8eb6958c-2340-4319-8abc-19259898057a)


### Parameters
* **confidence_breaker** - If there are confidence_breaker flagged tokens in a row, then rollback is triggered.
  * For Gemma 2 9B, a good starting point is confidence_breaker=8
  * Tune this up to lengthen the required number of tokens before a confidence breaker is triggered. Tune it down to lower them. You can drop this down to 3 or even to 2 at your own risk for detection of very short sequences, or extend it out to larger numbers like 20 or 30.
* **mid_threshold** - If a logit is encountered above this threshold and below the high_threshold, it is counted as a flagged token. 
  * For Gemma 2 9B, a good starting point is mid_threshold=15.0
  * Tune this up to tighten the conditions for detecting tram-tracks, tune it down to loosen them
* **high_threshold** - If a logit is encountered above this threshold, any current detection activity is reset. This way the model is permitted to provide a prediction that it is highly confident if it is the only viable option. For example if it follows logically or grammatically from previously generated tokens.
  * For Gemma 2 9B, a good starting point is high_threshold=22.0
  * Tune this down if the model starts escaping tram-tracks by resorting to incoherence, tune it up if you get too many tram-tracks slipping through
* **confidence_breaker_debug** - This is a boolean True/False (defaulting to False), which prints out a message containing the discarded text whenever a rollback is initiated.

### Note on use

1. Use the high_threshold parameter wisely, this is how you ensure that the model can still maintain coherence if following a tram-track is the only way to do so.
2. It is entirely possible to dial the settings for this sampler too high, resulting in the model never producing an accepted output, or only after a very large number of rejected solutions. Loosen the settings a little if you find that generation is taking longer than you can bear.
3. Sometimes discarded text may be shorter than the confidence breaker setting, where there has been a rollback and then the newly generated tokens combine with still-prior generated tokens to form a new tram-track pattern. There is no recursive rollback in this case to the beginning of that new tram-track, but rather we return directly to the previously triggered rollback point and try another token from there. In this way we avoid escaping from one tram-track by jumping into another.



