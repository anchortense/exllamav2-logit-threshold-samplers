## Logit samplers for coherent creativity
(with many thanks to [exllamav2](https://github.com/turboderp/exllamav2/tree/master))

Two new samplers enabling coherent diverse text generation.
*  The **logit threshold sampler** filters low-confidence logits and enables the application of much higher temperatures to stronger candidates, generating varied outputs without losing coherence.
*  The **confidence breaker sampler** addresses repetitive text sequences by dynamically detecting them on the logit level, allowing the model to generate more diverse responses. CB builds upon the existing implementation of banned strings in exllamav2.

The key innovation is to use absolute logit values instead of softmax probabilities to retain the model’s raw confidence in each token, allowing more precise evaluation and filtering.

Scroll down for generation examples/comparisons, and parameter documentation.

### Proof of concept
This repository contains a proof of concept implementation of the above samplers, forked from exllamav2 v0.2.3.

branch master (this branch) is the original proof of concept implementation and can be used without recompiling just by dropping the relevant files into exllamav2/generator

branch full is the standardised c++ implementation intended for merging, and won't work without recompiling.

### Recommended install process
Create and activate a new environment, pip install an official exllamav2 wheel from [here](https://github.com/turboderp/exllamav2/releases/tag/v0.2.3) and then swap in the exllamav2/generator folder from this fork.

### Minimal example

```python
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler, ExLlamaV2DynamicJob

model_dir = "models/gemma-2-9b-it-exl2/"

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

# Set up sampler settings
settings = ExLlamaV2Sampler.Settings()
settings.logit_threshold_stats = False
settings.confidence_breaker_debug = False

settings.temperature = 1.0
settings.temp_threshold = 10.0
settings.min_threshold = 10.0
settings.confidence_breaker = 8
settings.mid_threshold = 18.0
settings.high_threshold = 0.999

for idx, prompt_str in enumerate(prompts):
    job = ExLlamaV2DynamicJob(
        input_ids=tokenizer.encode(prompt_str, add_bos=True),
        max_new_tokens=500,
        stop_conditions=[tokenizer.eos_token_id],
        gen_settings=settings,
        identifier=idx,
    )
    generator.enqueue(job)

# Somewhere to store the streaming results
collected_outputs = [""] * len(prompts)

while generator.num_remaining_jobs():
    results = generator.iterate()
    for result in results:
        idx = result["identifier"]
        text_chunk = result.get("text", "")
        collected_outputs[idx] += text_chunk

print()
for idx, o in enumerate(collected_outputs):
    print(f'{prompt_insert}{o}\n')
```

## Output comparison

Prompt: "Write at least 500 words, beginning with and in the same style as the following:\nOnce upon a time,"

Model: gemma-2-9b-it-exl2-6bpw

Note that the trials below are cut-off after the first paragraph block.

### Baseline output for comparison
```
settings.temperature = 1.0
```
>**Trial 1.**
>
>Once upon a time, nestled amongst rolling hills and whispering willows, lay the village of Oakhaven. It was a place where life unfolded at a gentle pace, where the scent of freshly baked bread mingled with the earthy aroma of tilled soil, and where laughter echoed through cobblestone streets as children chased fireflies at dusk.  But beneath the idyllic veneer, a shadow lurked. For generations, Oakhaven had been plagued by a mysterious enchantment, a creeping melancholy that settled over its inhabitants like a shroud. 
>
>**Trial 2.**
>
>Once upon a time, nestled amongst rolling hills and whispering willows, lay the village of Willow Creek. It was a place where laughter echoed through cobblestone streets, where the warmth of freshly baked bread filled every home, and where the nights were alive with the twinkling tales spun by elder firekeepers. But beneath this idyllic surface, a shadow lurked – a whisper of forgotten magic, a secret yearning for something more.
>
>**Trial 3.**
>
>Once upon a time, nestled amongst rolling hills draped in emerald moss and kissed by the sun's golden rays, lay the village of Whispering Willows. It was a place where time seemed to flow a little slower, where the gentle murmur of the river Willowbrook was the village's constant lullaby, and where the villagers lived simple, contented lives. 
>
>**Trial 4.**
>
>Once upon a time, nestled amidst rolling emerald hills and whispering willow trees, lay the quaint village of Sunhaven. Here, life flowed at a gentle pace, punctuated by the clang of the blacksmith's hammer, the melodic bleating of sheep, and the warm laughter of children chasing fireflies in the twilight. But beneath this idyllic surface, a secret slumbered, a tale whispered only in hushed tones around crackling hearths. A secret that spoke of an ancient prophecy, an echo of a forgotten magic, and a destiny
>
>**Trial 5.**
>
>Once upon a time, nestled amidst rolling emerald hills and whispering willow trees, lay the village of Whisperbrook.  It was a place where life moved at the gentle pace of a babbling brook, where laughter echoed through cobblestone streets, and where the scent of freshly baked bread mingled with the sweet perfume of honeysuckle. 


```
# With min_p + temperature last
settings.temperature = 1.5
settings.min_p = 0.025
settings.temperature_last = True
```
>**Trial** 1.
>
>Once upon a time, nestled amongst rolling hills that wore crowns of emerald pines, lived a young woman named Elara.  Unlike other girls her age, who dreamt of grand balls and handsome princes, Elara yearned for adventures beyond the cobblestone streets of her village. Her heart thrummed with a wild, untamed rhythm, longing for the whispered secrets of ancient forests and the call of mountains shrouded in mist.
>
>**Trial** 2.
>
>Once upon a time, nestled amongst the emerald foothills of the Whispering Mountains, lay the kingdom of Atheria. Its people, known for their vibrant artistry and unyielding optimism, lived in harmony with nature, their lives interwoven with the rhythm of the seasons. Atheria was ruled by a benevolent queen, Elara, whose wisdom flowed like the silver river that snaked through the heart of the kingdom. But even in this realm of perpetual sunshine and song, a shadow was brewing.
>
>**Trial** 3.
>
>Once upon a time, nestled between towering redwoods and whispering ferns, lived a young girl named Elara. Her hair was the color of fallen leaves, her eyes like moss-covered stones, and her laughter rang like wind chimes in the forest breeze. Elara knew every inch of the ancient woods, every secret path, every whispering brook. She spoke the language of birds, understood the sighs of the trees, and could track a deer through the densest undergrowth.
>
>**Trial** 4.
>
>Once upon a time, nestled between towering emerald hills and a sapphire sea kissed by gentle waves, lay the quaint village of Sunhaven. Its cobblestone streets, worn smooth by generations of footsteps, wound around houses painted in hues reminiscent of ripened fruit and blushing sunsets. Laughter often spilled from open windows, mingling with the scent of freshly baked bread and the melodies of birdsong. Life in Sunhaven flowed at a leisurely pace, governed by the rhythm of the sun and the whispers of the wind.  
>
>**Trial** 5.
>
>Once upon a time, nestled amongst the silver-leafed birch trees and whispering willows by the banks of the River Whisper, lived a girl named Elara. Her hair was the color of spun moonlight, her eyes like pools reflecting the twilight sky, and her spirit as untamed as the winds that rustled through the ancient trees. Elara wasn't like the other village girls, content with their embroidery hoops and quiet afternoons. She craved adventure, her heart yearning for the mysteries that lay beyond the familiar boundaries of

```
# With XTC
settings.temperature = 1.0
settings.xtc_threshold = 0.1
settings.xtc_probability = 0.5
```
>**Trial** 1.
>
>Once upon a time, nestled amongst the emerald foothills of a slumbering volcano, lay the village of Asteria.  A place where sunlight dripped like honey through the leaves of ancient willow trees, and laughter echoed from cobblestone streets lined with quaint flower stalls and bakeries overflowing with the scent of warm bread. Life in Asteria moved at the leisurely pace of a hummingbird's wings, each day a gentle brushstroke on the canvas of existence.
>
>**Trial** 2.
>
>Once upon a time, nestled amidst rolling emerald hills and whispering willow trees, lived a village known as Sunhaven.  Its name was well-earned, for the sun seemed to linger longer in that place, bathing the cobblestone streets and thatched rooftops in a glorious golden light. Life there flowed at a gentle pace, dictated by the rhythms of the seasons and the cheerful clanging of the blacksmith's hammer.  The villagers were known for their kindness, their hearty laughter echoing through the market square, and their unwavering belief in the magic that whispered within the heart of their world.
>
>**Trial** 3.
>
>Once upon a time, nestled between the whispering pines and the silver ribbon of Willow Creek, sat the village of Whisperwind. It wasn't much to look at, cobbled streets winding through rows of gingerbread cottages painted in pastel hues, smoke curling lazily from chimneys topped with crooked teacup weathervanes. But within its humble walls simmered a secret, a magic hummed beneath the everyday chatter and laughter, as palpable as the scent of honeysuckle that drifted on the evening breeze.
>
>**Trial** 4.
>
>Once upon a time, nestled amidst emerald hills and whispering willows, lay the quaint village of Sunshadow.  A ribbon of silver, the Sunstream River, snaked through its heart, reflecting the dancing sunlight and lending an air of perpetual serenity to the village. Life there unfolded at a measured pace, governed by the rising and setting of the sun, the blossoming and fading of flowers, and the gentle rhythm of the river’s flow. 
>
>**Trial** 5.
>
>Once upon a time, nestled amidst emerald hills and whispering willows, lay the village of Willow Creek. Its inhabitants, hardy folk with hearts as warm as their hearth fires, lived simple lives, their days a tapestry woven from the rhythms of nature. 
>


### Logit threshold sampler
```
# LTS with Conservative settings
settings.temperature = 1.0
settings.temp_threshold = 10.0
settings.min_threshold = 10.0
```
>**Trial** 1.
>
>Once upon a time, nestled between the whispering emerald hills of Avani, lay the tiny village of Sunhollow. Its name was apt, for sunlight seemed to linger longer there, bathing the cobblestone streets and thatched roofs in a warm, golden glow. The villagers lived in comfortable simplicity, tending their vegetable gardens, crafting handmade goods, and sharing laughter and stories around crackling hearths. But behind the peaceful facade, a secret hummed beneath the surface. 
>
>**Trial** 2.
>
>Once upon a time, deep within the Whispering Woods, lived a little firefly named Flicker. Unlike his brothers and sisters, who delighted in lighting up the night with their emerald glows, Flicker’s light was faint, a mere flicker compared to their brilliant beacons. This made him feel small and insignificant, like a single star lost in a vast, glittering galaxy.  
>
>**Trial** 3.
>
>Once upon a time, hidden deep within a whispering forest where sunlight dappled through emerald leaves and ancient oaks stretched their branches towards the heavens, lived a little firefly named Flicker. Unlike his brethren, who delighted in their nightly dances, illuminating the darkness with joyous flickers, Flicker harbored a secret fear: the dark.  He yearned to shine brightly, to be part of the wondrous luminescent spectacle, but the shadows seemed to swallow him whole, amplifying the tremor of unease in his tiny heart.
>
>**Trial** 4.
>
>Once upon a time, in a village nestled between emerald hills and a sapphire sea, lived a girl named Elara.  Her hair, the color of spun moonlight, cascaded down her back like a silver waterfall, and her eyes, as blue as the deepest ocean depths, held a wisdom far beyond her years. But Elara was no ordinary girl. She possessed a secret, a gift she kept hidden from the world: she could speak to the wind.
>
>**Trial** 5.
>
>Once upon a time, in a world painted with hues of lavender and turquoise, where the sun dipped below the horizon in a fiery blaze of apricot and rose, lived a young woman named Elara. Her skin shimmered like moonlight on water, her hair flowed like a waterfall of midnight, and her eyes held the depth of a starless night. But Elara wasn’t known for her beauty alone; she possessed a voice, a melody that could coax flowers to bloom and soothe the wildest of beasts.
>


```
# Pushing the temperature a little higher
settings.temperature = 4.0
settings.temp_threshold = 14.0
settings.min_threshold = 10.0
```
>**Trial** 1.
>
>Once upon a time, in the heartland of Whispering Glade where sunrays danced upon golden chamomile meadows and sapphire butterflies dipped their painted wings in crimson poppy blooms, resided the village of Earthen Charm.
>
>**Trial** 2.
>
>Once upon a time, in the velvet folds of dusk when the world draped itself in shadow, lived Elora, the Light Librarian of Asteria's Lunarium. Asteria itself was a whisper in the stars, a secluded isle perched precariously between dreams and wakefulness. Legend claimed it pulsed in tandem with the moon, shimmering and dissolving with its silvery tides. The Lunarium was its beating heart - an incandescent tower reaching towards the twilight sky, each of its many levels overflowing with forgotten wisdom and illuminated by the captive whispers of stardust. 
>
>**Trial** 3.
>
>Once upon a time, tucked between rolling hills sprinkled with dew-laden wild roses, lived a boy named Finnian. Finnian wasn't like the other lads who daydreamed about swords and quests, their eyes constantly on distant horizons. Finnian lived and breathed the quiet murmurings of his small forest village. He spoke with the owls before twilight bathed the leaves in emerald shadow and learned to follow the tread patterns left in soft earth after rain. His only adventure yearned, he admitted, was finding new mushrooms for Maéva, the apothecary, each cap bearing unique swirls and freckles that tickled the whimsical fancies in his mind. 
>
>**Trial** 4.
>
>Once upon a time, tucked away beyond shimmering mist-laden peaks, lay a valley steeped in forgotten tales. This was the Vale of Songweaves, its once verdant tapestry now withered under a cloak of shadow cast by the Sunthief. Whispers spoke of a woman, an archweaver who spun words into reality, cursed to vanish before sunrise. From the vibrant threads she wove, stories would spill into life, dancing across hills and tumbling from mountain peaks like lyrical streams. Then, the sun would kiss the sky crimson, and Posey, as she was known, would fade, leaving behind only remnants of her whispered enchantment, echoing sighs in the twilight.
>
>**Trial** 5.
>
>Once upon a time, there lived a baker named Elodia in a small village tucked beneath the emerald embrace of the Moonstone Peaks. Elodia was famed across the land for her magical bread: sourdough tangy with wild honeydew dreams, cinnamon swirls scented with wishes made upon the first starlight, and rye pumpernickel enriched with stories whispered from generations past. Though her talents could capture any soul with a morsel of pure delight, Elodia lived a quiet, almost solitary life within the cramped wooden frame of her bakehouse. Days melted into one another, each rising like the perfect sourdough loaf – a blend of tradition and tireless work.
>


### Confidence breaker
```
# CB with moderate settings
settings.temperature = 1.0
settings.confidence_breaker = 3
settings.mid_threshold = 18.0
settings.high_threshold = 0.999
```
>**Trial** 1.
>
>Once upon a time, tucked between emerald-hued rice fields that swayed in rhythmic patterns to a whispered breeze, nestled a village unlike any other. Here the villagers, known for their radiant smiles and laughter as infectious as a summer sunshower, possessed a unique gift – each held a piece within the ancient, shimmering tapestry of dreams. They spun stories from starlight, wove emotions into moonlight, and painted landscapes with their whispered lullaby melodies, all captured within their individual threads of luminous fabric. These dream-tapestries, vibrant and ever-evolving, hung in each home like shimmering galaxies, a constant testament to the villagers' boundless imagination and shared history.  The tapestry of Elder Esha's home, however, was strangely subdued tonight. It usually blazed with the colors and energy of a thousand suns, reflecting her spirit, as bright and boundless as a cerulean dawn sky, but tonight it hung dim and muted, a canvas of hushed twilight hues mirroring the sorrow that etched lines on her usually cheerful face
>
>**Trial** 2.
>
>Once upon a time, in a valley cloaked by emerald forests and kissed by silver moonlight streams, there existed a village named Evergleam, known for its harmonious melodies and vibrant laughter, where the air itself shimmered with a joyful aura. Its inhabitants, the Lunare, were a people of exceptional kindness, their skin as luminous as polished amber under twilight’s soft glow. Their laughter echoed through the ancient oak trees, and the sound of their weaving looms, crafting tapestries woven from moonlight and dreams, lulled the village into tranquil sleep. They lived in perfect rhythm, guided by whispers of the wind and the gentle pulse of nature.
>
>**Trial** 3.
>
>Once upon a time, there sat beneath a gnarled oak, as ancient as its surrounding forest, a young witch by the name Elora. Her cauldron, normally brimming over with fantastical concoctions, lay cold, abandoned beside a stack of unlit herbs and dried blossoms.  The air around her throbbed with an unsettling stillness, a silence thick enough to choke on. Elora wasn't grieving, not precisely. It wasn't sorrow that weighed upon her heart, a heavy stone nestled deep within its hollow. It was… discontentment, an itch she could never fully satisfy.  Years spent mastering the art of brewing potions, weaving charms and whispering spells for the good of the nearby villages had grown tedious, predictable. 
>
>**Trial** 4.
>
>Once upon a time, in a valley cradled amidst the ancient mountains that touched a perpetually stormy sky, there existed an inn called The Waking Hourglass. Built into a gnarled and knobbly oak that had witnessed centuries pass, its walls th rummed and echoed the whispering stories carried by the wind. 
>
>**Trial** 5.
>
>Once upon a time, in a valley nestled between shimmering waterfalls, there existed an ancient oak, its branches sprawling wide like a wise grandmother welcoming weary souls beneath its leafy shade. This oak was known throughout the valley and beyond as The Heartkeeper. 
>

```
# CB with a longer breaker
settings.temperature = 1.0
settings.confidence_breaker = 8
settings.mid_threshold = 17.0
settings.high_threshold = 0.999
```
>**Trial** 1.
>
>Once upon a time, deep beneath a silver-threaded sky, nestled amongst rolling hills of lavender and sapphire grass, resided a town named Whisperwind.  Its houses stood crafted from luminous moonstone, their spires reaching upwards like fingers grasping for starry blessings, their windows glowing with a thousand soft fires that banished even the deepest shadows. The townsfolk themselves were a peculiar breed – gentle whispers of wind woven from star-dust, their laughter a tinkling of distant chimes, their sorrows sighs echoing through rustling reeds.
>

>**Trial** 2.
>
>Once upon a time, in a forgotten pocket dimension veiled from our mundane reality, lived a peculiar race known as the Luminaries. Their bodies weren’t of flesh and bone, but intricate constellations of shimmering starlight, each twinkle a spark of cosmic energy. They existed on a plane where thoughts became constellations themselves, and emotions painted galaxies. Harmony was their nature, for their very essence was the cosmic ballet, a celestial orchestra where every being played a vital role in maintaining balance and wonder.
>

>**Trial** 3.
>
>Once upon a time, nestled amongst rolling hills of shimmering sunflowers, sat Oakhollow, a village seemingly stitched directly into a dreamscape. Its houses weren't merely built, they were crafted; thatched roofs curved gracefully into fantastical shapes, chimneys puffing out smoke swirls resembling ethereal dragons. Stone walls held the warmth of hearth fire within, and windows shone brightly, casting dancing patterns onto cobblestone streets. Laughter echoed from every doorway, mingling with the sweet music carried on the gentle breeze from hidden harps and wind chimes.
>

>**Trial** 4.
>
>Once upon a time, nestled between rolling green meadows where wild strawberries ripened, and whispering woods thick with ancient trees and mossy paths, there lived a peculiar creature. Not a beast or fairy tale troll or mischievous sprite, but something altogether different – an alchemist, who called himself Thistleweed. Now, most alchemists you might imagine were grand lords in their imposing towers, filled to bursting with brass retorts, glass bubbles, and the pungent aroma of bubbling elixirs. But Thistleweed was a simpler fellow.  
>

>**Trial** 5.
>
>Once upon a time, nestled amidst rolling emerald hills and whispering willow streams lay Willow Creek, a village untouched by grand conquests or sweeping revolutions. Here life flowed at the measured rhythm of the seasons, guided by the warmth of sunrises and the silvered tapestry woven across moonbeams. Each resident bore the mark of their heritage, woven into the very fabric of their being – from the stonecutter with hands roughened by granite to the cobbler whose nimble fingers sang stories into every leather stitch.
>

### Logit threshold sampler + Confidence breaker
```
# LTS + CB, conservative settings
settings.temperature = 1.0
settings.temp_threshold = 10.0
settings.min_threshold = 10.0
settings.confidence_breaker = 8
settings.mid_threshold = 17.0
settings.high_threshold = 0.999
```
>**Trial** 1.
>
>Once upon a time, in the whispering meadows where lavender bloomed as tall as a hobbit and fireflies danced with forgotten melodies beneath a velvet cloak of dusk, there resided an unusual dragon called Zinn. Most dragons, you see, were accustomed to hoarding gold, squinting maliciously from atop fiery peaks, and breathing jets of superheated air upon those unfortunate enough to cross their path.
>
>**Trial** 2.
>
>Once upon a time, in a valley whispered to be blessed by the ancient Fey Queen herself, nestled amongst emerald hills and a patchwork field of wildflowers, lived Elara, a weaver of exquisite tapestries.  Her loom sang with threads spun from moonlight and dreams, her fingers dancing over shuttle and hearth fire as they brought stories to life in vibrant hues. Yet, for all her talent and the enchantment woven into her work, a silent sorrow clung to Elara’s heart, a whisper of something missing from the vibrant tapestry of her life.
>
>**Trial** 3.
>
>Once upon a time, in the heart of a whispering forest shrouded by ancient oaks and draped in emerald moss, lived a young sapling named Fernwood. Unlike his sturdy siblings, reaching towards the sun with confident green tendrils, Fernwood remained stubbornly earth-bound. His leaves, more violet hues against the vibrant forest canopy, were soft, almost frail, and he seemed perpetually on the brink of topple.
>
>**Trial** 4.
>
>Once upon a time, in a realm woven with lavender mist and silver moonlight, nestled between ancient redwood groves and a whispering obsidian lake, there existed a village called Whisperwind.  Now, Whisperwind was no ordinary village. It hummed with a subtle, almost imperceptible magic, a harmony resonated not from the earth but from the whispers of the ancient trees themselves.  These whispers, murmured tales of forgotten times, spun dreams into existence and carried secrets carried on the wind. No map could locate Whisperwind, for it existed on the edge of perception, appearing only to those who longed for its quiet beauty, or perhaps, were summoned by it.
>
>**Trial** 5.
>
>Once upon a time, nestled in the crook of a whispering willow by an emerald pond that hummed with dragonflies, lived Pip, a peculiar boy no bigger than a hummingbird. He wasn't born small; he simply was, a wisp of a boy whose bones were starlight and whose laughter sounded like tiny windchimes. His skin shimmered with an otherworldly glow, pearlescent in sunlight but catching a twilight amethyst in moonlight. Pip, however, cared little for his difference. He was too busy collecting whispers from the willow, decoding sunbeams, and befriending the plump ladybugs who shared his lunch. 
>

```
# Pushing the temperature up
settings.temperature = 4.0
settings.temp_threshold = 14.0
settings.min_threshold = 10.0
settings.confidence_breaker = 8
settings.mid_threshold = 17.0
settings.high_threshold = 0.999
```
>**Trial** 1.
>
>Once upon a time, in a sleepy village tucked away in the valley below towering purple peaks, there lived a miller named Theodore with a smile perpetually stuck like melted marmalade upon his weather-beaten face. Though blessed with calloused hands strong from years of shaping golden barley, his wisdom often resided within the realm of airy tales. Theodore was known far and wide as "Whisperin'" because he always had an enchanting yarn spun around every crook of conversation, each word a precious grain woven into a sparkling tapestry. The villagers would perch themselves upon dusty barrels by the gurgling village fountain, enthralled by his rambling accounts of grinning moonrocks, skyfaring goldfish, and whispered bargains with grumpy river fairies.  
>
>**Trial** 2.
>
>Once upon a time, in the tangled emerald labyrinth that was the Moonwood Forest, there lived a small tribe called the Twigweavers. Dwelling within living root fortresses that spiraled into the gnarled oak trees, these peculiar people weren’t quite of earth nor truly of the fey creatures that inhabited the twilight recesses of the Moonwood. Instead, they possessed an intangible magic woven into their bloodlines – a power over plants and their boundless adaptability. Unlike other tribes that inhabited the forest, the Twigweavers remained ever mobile.  Instead of tilling the earth, their dwellings themselves followed the cycles of growth and dormancy, sprawling into hollow trunks and shedding barky layers to shelter them in accordance to the moon’s phases. Each month they journeyed with their mobile home, leaving only gentle signs of their passing -  troidered leaves blooming in dazzling constellations and twisting pathways of overgrown moss leading into hidden glens.  
>
>**Trial** 3.
>
>Once upon a time, in a realm bathed in perpetual sunlight, there lived an imp called Figgle. His horns were too pointy, his wings were an absurd shade of pickle-purple, and he harboured a secret, shameful ambition: Figgle wanted to paint. A dangerous ambition, you might say, in the whimsical realm of Ignorans. In this world of shimmering rivers and cloud-petting mountain nymphs, where shadows whispered untold stories only dreamed by dancing moonlight, a gloomster like Figgle simply wasn't supposed to exist. Or to crave something as humanly indulgent as artistic expression.  Painting, the nymphs insisted, was the domain of sunbeams, their brilliance best displayed upon fields of pollen-infused wildflowers. It wasn't fit for someone as, shall we say, “miscoloured” as Figgle.
>
>**Trial** 4.
>
>Once upon a time, high on a mountainside cloaked in ancient moss, sat a single, spindly, neglected walnut tree. Among its brethren – magnificent sycamores with boughs like weathered galleon sails and chestnut elders, sturdy sentinels draped in ivory flowers – it stood solitary, stunted and brow-beaten by neglect. Generations of woodcutters and artisans had overlooked it, valuing the hardwoods that fueled their trade and shaded the earth, leaving the walnut, its wood prized but scant and brittle, to toil unseen and unheard. 
>
>**Trial** 5.
>
>Once upon a time, in the rosy mist-wreathed valleys of Dawnwind Forest, lived a brownie named Poppypetal, known to every pixie and troll throughout the Whispering Meadows. Unlike other brownies, content with tinkering in mossy hollows or pulling whimsical tricks on startled rabbits, Poppypetal dreamt of adventure, stories whispered by the rustle of oak leaves and carried on the back of migratory birds. 


```
# Pushing the temperature even further up, with a short breaker
settings.temperature = 10.0
settings.temp_threshold = 16.0
settings.min_threshold = 12.0
settings.confidence_breaker = 3
settings.mid_threshold = 17.0
settings.high_threshold = 0.999
```
>**Trial** 1.
>
>Once upon a time, the kingdom of Alderanth shimmered like an amethyst pendant strung from the sun-kissed branches of ancient, weeping willow. The air, perpetually laced with honeysweet blossoms, danced with laughter and the murmur of joyful chatter. Alderanth belonged to Princess Aethel, whose hair cascaded down her emerald shawl in waves as black and lustrous as night and eyes sparkled like aquamarine on stormy seas. Known for her gentle soul and wit sharper than any dagger, she reigned with kindness, ensuring even the smallest village boy felt her watchful embrace and booming laugh resonated across their merry folk music festivals. But, hidden behind the iridescent curtains drawn over her silken-cushion balcony, the truth was a different story altogether. Princess Aethel yearned. 
>
>**Trial** 2.
>
>Once upon a time, the sky wasn’t a dependable blue dome woven from morning light and summer cloud candy. Instead, it pulsed and morphed like an immense, ethereal heart. Scarlet rivulets would seep in, spreading like molten fireflies across the midday expanse, before yielding to bands of shimmering jade that dripped like luminous honey towards sunset's edge. Then, the heavens would cloak the land in twilight’s ebony embrace, sprinkled with a galaxy born from whispered wishes and sleeping dragons’ fireflies, so bright that stars felt like celestial marshmallows just begging to be grasped.
>
>**Trial** 3.
>
>Once upon a time, tucked away between rolling green fields of clover and wildflowers and under a sky washed bright by perpetual dawn, lay the city of Aetheris. Now Aetherians, unlike folk in most lands where people woke, bathed, and breakfasted on mundane earth-bound schedules, lived a rhythm far more magical, fueled by the pulsations of the great luminaire at Aether's heart, the Singing Skypearl. The Pearl thrummed, its light growing steadily brighter each morning until reaching zenith in the late afternoon. At that point, bathed in the warm, silvery luminescence of the pearl's glow, life itself pulsed to its zenith. But just before midnight when the pearl began to fade, Aether’s inhabitants retired for slumber, cocooning themselves in woven blankets, humming soft, lyrical tunes, lulled to dreams by the dying reverberation of the Singing Skypearl's lonesome call. 
>
>**Trial** 4.
>
>Once upon a time, the sky poured with silver.  No rain it was, though it resembled a soft, shimmering shower of countless sequins scattered by some colossal and benevolent prankster. Instead of pooling in rivers, this silvery fluid drizzled into every creak of the floorboard, seeped between cobblestone stones, and collected within the dusty hollow of each fallen acorn. On days of silver showers the air held the sharp, crisp bite of coming starlight and whispered promises of unseen adventures.  One particularly silver-heavy morn, in an age before trains or telegraph, when carriages still clopped across mud-rutted lanes and tales travelled faster on windblown whispers than ink stained pages, young Anya dreamt she could breathe constellations and fly with fireflies. The silver rain kissed her skin as she sat upon a window sill, gazing at the vibrant tapestry woven by the shimmering droplets on her worn wool cloak. A glinting pathway snaking towards the forest's emerald fringe, formed by the spilled silver, called to her adventurous
>
>**Trial** 5.
>
>Once upon a time, in a world draped in hues unseen to any but the kind-hearted and those brave enough to peer beneath the shimmer, lived the Willow Weaver, Elwyn. Not by profession, for weaving wasn’t something you chose in a land of perpetual starlight where every blossom sang, but by essence. Elwyn was spun from the same light as the starfire moss and the Moon-Dancer butterflies. His whispers carried the scent of blooming Duskweed and the rustle of wings unseen in the emerald forests.


## Explanation: Logit threshold sampler

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
  * For Gemma 2 9B, a good starting point is min_threshold=10.0
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

## Explanation: Confidence breaker sampler
Current generation language models are well known for producing certain cliched phrases, which would not necessarily be problematic in a single instance, but are known to be produced repeatedly in response to varied prompts. This is the so-called ai-slop problem. In 'deterministic' use cases this is usually not an issue, as we are simply looking for the one correct answer. In scenarios where engaging, diverse language choices are valued, ai-slop represents a significant limitation.

Current approaches to resolving this issue involve user defined lists of banned strings. This can be reasonably effective, however fail to address the deeper issue, which is the tendency of language models to funnel their responses into unintentionally learned 'tram-track' token sequences, where token choices are strongly conditioned by their immediate predecessors in a manner which is not logically or grammatically implied by the prompt or by those preceding choices.

This is a far deeper and more pervasive problem than the well known handful of phrases commonly associated with the idea of ai-slop. These tram-tracks exist within trained models because the tokens within them are good predictors for text completion. Nevertheless a user passing either the same or similar prompts repeatedly, looking for diverse outputs, will quickly observe that model responses which seemed initially impressive are in fact tram-track patterns, and the apparent diversity of outputs is an illusion.

### Using the confidence breaker to jump tracks

The **confidence breaker sampler** addresses the issue by looking for logit patterns that signal we have entered a tram-track, and extending exllamav2's banned string functionality to roll-back to the token directly before we entered the tram-tracks. The tram-tracked tokens are then discarded and replaced by a novel generation, which will take us down a different, less travelled path.

Based on empirical observation of logits and the conditions for the appearance of tram-tracks, the pattern that the confidence breaker looks for to identify these tram-tracks is a sequence of mid-high valued logits, logits which have been nudged higher than a good score by over-training, but which are not yet so guaranteed as logical or grammatical necessity.

Empirical observation also validates the decision to return back and alter the token *before* the tram-track, rather than the first tram-track. This is a token which had a reasonable range of viable alternatives, but once the model settled on the decision it made, the tram-track became nearly inevitable. So, this is the error we have to correct.


### Parameters
* **confidence_breaker** - An integer setting the length of a sequence match. If there are confidence_breaker flagged tokens in a row, then rollback is triggered.
  * For Gemma 2 9B, a good starting point is confidence_breaker=8
  * Tune this up to lengthen the required number of tokens before a confidence breaker is triggered. Tune it down to lower them. You can drop this down to 3 or even to 2 at your own risk for detection of very short sequences, or extend it out to larger numbers like 20 or 30.
* **mid_threshold** - If a logit is encountered above this threshold and below the high_threshold, it is counted as a flagged token.
  * If this value is <= 1, it is treated as a probability value, if it is > 1 it is treated as a logit threshold
  * For Gemma 2 9B, a good starting point is mid_threshold=17.0 - it is recommended to use this as a logit threshold, as these reflect absolute token assessments
  * Tune this up to tighten the conditions for detecting tram-tracks, tune it down to loosen them
* **high_threshold** - If a logit is encountered above this threshold, it is assigned a None flag, and neither adds to the sequence nor resets it. This way the model is permitted to provide a prediction that it is highly confident if it is the only viable option. For example if it follows logically or grammatically from previously generated tokens.
  * If this value is <= 1, it is treated as a probability value, if it is > 1 it is treated as a logit threshold
  * For Gemma 2 9B, a good starting point is high_threshold=0.999 - it is recommended to use this as a probability value, as this reflects a relative confidence wrt other options
  * Tune this down if the model starts escaping tram-tracks by resorting to incoherence, tune it up if you get too many tram-tracks slipping through
* **confidence_breaker_debug** - This is a boolean True/False (defaulting to False), which prints out a message containing the discarded text whenever a rollback is initiated.

### Note on use

1. Confidence breaker is a workaround for model overfitting. Its use may be detrimental to a model that has no overfitting (however all major models do, for now).
2. Use the high_threshold parameter wisely, this is how you ensure that the model can still maintain coherence if following a tram-track is the only way to do so.
3. As you lower the temperature, consider raising the mid_threshold - if the confidence breaker triggers on too many sequences in a row the model may reduce coherency to produce accepted outputs.
4. It is entirely possible to dial the settings for this sampler too high, resulting in the model never producing an accepted output, or only after a very large number of rejected solutions. Loosen the settings a little if you find that generation is taking longer than you can bear.
5. Sometimes discarded text may be shorter than the confidence breaker setting, where there has been a rollback and then the newly generated tokens combine with still-prior generated tokens to form a new tram-track pattern. There is no recursive rollback in this case to the beginning of that new tram-track, but rather we return directly to the previously triggered rollback point and try another token from there. In this way we avoid escaping from one tram-track by jumping into another.



