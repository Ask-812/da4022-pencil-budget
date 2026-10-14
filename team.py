"""Team list for the slide and proposal builders.
Roll numbers are read from team_rolls.json (git-ignored) so they never reach the public repo."""
import json, os

NAMES = ["Amit Kumar", "Ashmit Satpute", "Arnav Kshirsagar", "Pranav Shalya", "Atharv Wankhede"]
_F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_rolls.json")
ROLLS = json.load(open(_F)) if os.path.exists(_F) and not os.environ.get("PENCIL_PUBLIC") else {}


def team_line(order=NAMES, sep=" \u00b7 "):
    return sep.join(f"{n} ({ROLLS[n]})" if n in ROLLS else n for n in order)
