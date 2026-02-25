import random


# ─────────────────────────────────────────────────────────────────────────────
# Game rules (English)
# ─────────────────────────────────────────────────────────────────────────────

GAME_RULES = (
    "Bulls and Cows is a code-breaking game. A secret number of {num_digits} digits "
    "has been chosen. Every digit in the secret number is unique (no repeated digits), "
    "and digits are taken from 0-9. The number MAY start with 0.\n\n"
    "You are given a series of guesses along with their scores:\n"
    "- A \"bull\" means a digit in the guess is correct AND in the correct position.\n"
    "- A \"cow\" means a digit in the guess is correct BUT in the wrong position.\n"
    "- Each digit in the secret and each digit in the guess is used at most once when "
    "counting bulls and cows. Bulls are counted first; only the remaining (non-bull) "
    "digits can produce cows.\n\n"
    "Your task: determine the secret number that is consistent with ALL of the given "
    "clues. There is exactly one solution."
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATES = [
    (
        "{rules}\n\n"
        "Here are the clues:\n{clues}\n\n"
        "What is the secret {num_digits}-digit number?\n\n"
        "{output_format}"
    ),
    (
        "{rules}\n\n"
        "The following guesses and their scores are provided:\n{clues}\n\n"
        "Determine the secret {num_digits}-digit number.\n\n"
        "{output_format}"
    ),
    (
        "{rules}\n\n"
        "Below are previous guesses with their bull/cow scores:\n{clues}\n\n"
        "Find the secret {num_digits}-digit number.\n\n"
        "{output_format}"
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Output format instruction
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FORMAT = (
    "Please provide your final answer as the {num_digits}-digit string inside "
    "\\boxed{{}}. For example, if the secret number is 0123, write \\boxed{{0123}}."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: format clues
# ─────────────────────────────────────────────────────────────────────────────

def _format_clues(clues: list[dict]) -> str:
    """
    Format a list of clue dicts into readable text.

    Each clue dict has keys: 'guess', 'bulls', 'cows'.
    """
    lines = []
    for i, clue in enumerate(clues, start=1):
        guess_str = clue["guess"]
        bulls = clue["bulls"]
        cows = clue["cows"]
        lines.append(f"Guess {i}: {guess_str} → {bulls} bull(s), {cows} cow(s)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def prompt_bulls_and_cows(num_digits: int, clues: list[dict]) -> str:
    """
    Build the full prompt for a Bulls-and-Cows puzzle.

    @param num_digits: length of the secret number
    @param clues: list of dicts with keys 'guess', 'bulls', 'cows'
    @return: complete prompt string
    """
    rules = GAME_RULES.format(num_digits=num_digits)
    formatted_clues = _format_clues(clues)
    output_format = OUTPUT_FORMAT.format(num_digits=num_digits)

    template = random.choice(PROMPT_TEMPLATES)
    prompt = template.format(
        rules=rules,
        clues=formatted_clues,
        num_digits=num_digits,
        output_format=output_format,
    )
    return prompt
