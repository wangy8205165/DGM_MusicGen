"""
Generate 50 culturally diverse English prompts with MusicGen and save as
musicgen_prompt_1.wav ... musicgen_prompt_50.wav

Usage:
  python generate_multicultural_prompts.py           # all 50
  python generate_multicultural_prompts.py --limit 3  # first 3 only (smoke test)
"""
import argparse
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

MAX_TOKENS = 1024
MODEL_ID = "facebook/musicgen-large"

# 50 prompts: East Asia (China, Japan, Korea), Middle East, India, and other traditions
PROMPTS = [
    # China & broader Chinese cultural sphere
    "Traditional Chinese guzheng and erhu, pentatonic melody, calm meditation atmosphere",
    "Beijing opera style: percussion, jinghu strings, dramatic theatrical pacing",
    "Chinese bamboo flute xiao solo over soft silk-string backing, riverside dusk",
    "Cantonese narrative feel: yangqin and pipa, teahouse storytelling mood",
    "Tibetan singing bowls and long drones, high plateau wind ambience",
    # Japan
    "Japanese koto and shamisen, cherry blossom spring, gentle melancholy",
    "Taiko drums festival groove, energetic matsuri street energy",
    "Shakuhachi solo, Zen bamboo forest, sparse breathy phrases",
    "Bright J-pop anime opening: catchy synth hooks, driving drums, electric guitars",
    "Gagaku court music: slow sho clusters, biwa, ceremonial grandeur",
    # Korea
    "Korean gayageum plucked strings, slow han river evening, reflective",
    "Samul nori: janggu and buk drums, tight interlocking street rhythm",
    "K-pop dance track: punchy bass, crisp snares, glossy synth leads",
    "Pansori-inspired narrative: janggu pulse with expressive vocal-like melodic line",
    "Korean minyo folk: gentle 3/4 sway, nostalgic mountain air",
    # Middle East & North Africa
    "Arabic oud and darbuka, maqam improvisation, intimate coffee-house night",
    "Persian setar and tombak, dastgah mood, poetic and introspective",
    "Turkish ney and saz, mystical Sufi atmosphere, slow ornamented lines",
    "Lebanese dabke dance rhythm: mijwiz and hand drums, celebratory circle",
    "Moroccan gnawa: guembri bass and qraqeb metal castanets, trance pulse",
    # India & South Asia
    "North Indian classical: sitar and tabla, evening raga, alap to jhalla arc",
    "Carnatic South Indian: violin and mridangam, temple courtyard resonance",
    "Bollywood dance number: dhol groove, strings stabs, festive brass hits",
    "Bhajan devotional: harmonium drone, tabla theka, call-and-response feel",
    "Rajasthani folk: sarangi slides, peppy dholak, desert caravan energy",
    # Southeast & East Asia (more)
    "Indonesian gamelan: layered bronze metallophones, shimmering interlocking",
    "Vietnamese đàn tranh zither, water-puppet theater mood, flowing pentatonic",
    "Thai piphat classical ensemble, elegant slow procession",
    "Mongolian morin khuur fiddle with throat-singing drones, open steppe wind",
    "Philippine kulintang gongs and agung, Mindanao ceremonial layering",
    # Africa, Americas, Europe, Oceania (breadth)
    "West African griot: kora harp and calabash, storytelling pentatonic flow",
    "Ethiopian jazz: krar plucks, minor pentatonic horn lines, swinging groove",
    "Brazilian bossa nova: nylon guitar, soft brushed drums, Copacabana sunset",
    "Andean panpipes and charango, high-altitude wind, mountain trail",
    "Cuban son: tres guitar, claves, warm brass punches, dance hall",
    "Flamenco: nylon guitar, palmas, cajón, intense Phrygian flair",
    "Irish jig: fiddle and bodhrán, lively pub session",
    "Scottish Highlands: bagpipes and snare, misty march cadence",
    "Greek rebetiko: bouzouki and baglama, smoky taverna minor key",
    "Russian balalaika ensemble, brisk folk dance tempo",
    "Ukrainian bandura: plucked drones, open-field hopeful melody",
    "Armenian duduk solo with soft dhol, mountain valley lament",
    "Azerbaijani mugham: tar and kamancha, microtonal emotional arcs",
    "Egyptian electro-shaabi: synth stabs, tabla, street-party bounce",
    "Afrobeat: talking drums, tight hi-hats, horn section stabs",
    "Reggae dub: deep bass, rim shots, spring reverb, Kingston night",
    "Jazz fusion: electric piano and saxophone, 1970s urban groove",
    "Hawaiian slack-key guitar and ukulele, gentle surf sunset",
    "Australian didgeridoo with clapsticks, desert dawn drone",
    "Maori-inspired deep chants and wooden percussion, ceremonial stomp",
    # Cross-cultural fusion
    "Fusion: Chinese erhu meets Middle Eastern oud, shared minor modes, ambient hall",
]


def main() -> None:
    assert len(PROMPTS) == 50, f"Expected 50 prompts, got {len(PROMPTS)}"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N prompts (1..N) for testing.",
    )
    args = parser.parse_args()
    n = len(PROMPTS) if args.limit is None else min(max(args.limit, 0), len(PROMPTS))
    if n == 0:
        print("Nothing to generate (limit 0).")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    sampling_rate = model.config.audio_encoder.sampling_rate

    subset = PROMPTS[:n]
    for i, prompt in enumerate(subset, start=1):
        out_path = f"musicgen_prompt_{i}.wav"
        print(f"[{i}/{n}] {out_path}")
        print(f"    {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3,
                max_new_tokens=MAX_TOKENS,
            )
        wav = audio_values[0, 0].detach().cpu().numpy()
        scipy.io.wavfile.write(out_path, rate=sampling_rate, data=wav)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"Done. Saved musicgen_prompt_1.wav through musicgen_prompt_{n}.wav")


if __name__ == "__main__":
    main()
