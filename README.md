# LLM Randomness Test

A simple experiment testing whether large language models can generate truly random binary sequences.

## Summary of Findings

The experiment tested several LLM models' ability to generate random sequences of 0s and 1s. Results suggest most models show bias toward generating more 1s than would be expected in a truly random sequence.

![Proportion of 1s Generated by Different LLM Models](images/prop_1s.png)

*Figure 1: Proportion of 1s generated by different LLM models. The horizontal blue line represents the expected 0.5 probability for a truly random sequence. Green bars indicate models that passed the randomness test (p > 0.05), while red bars indicate models that failed (p < 0.05).*

Key findings:
- Only GPT-4o passed the randomness test (p = 0.281)
- All other tested models showed statistically significant bias (p < 0.05)
- Most models generate more 1s than 0s (proportions ranging from 0.52 to 0.55)
- Interestingly, Gemini-2.0-flash-lite showed the opposite bias (generating fewer 1s, proportion = 0.47)
- Sample sizes ranged from n=500 (Gemini-2.0-flash) to n=2998 (GPT-4o)

## How to Run

The script `random_llm_test.py` contains the testing logic. You'll need to set the following environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

Run the script with:
```
python random_llm_test.py
``` 
