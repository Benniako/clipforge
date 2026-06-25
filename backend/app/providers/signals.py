"""Transcript signal extraction shared by detection and scoring.

These are deliberately transparent, lexicon-driven features — the PRD's hardest
requirement for the score is that it be *explainable*, not a black box. Each
feature returns a value in [0, 1] and a short human-readable reason, so the same
computation drives both candidate ranking and the user-facing score.

The lexicons are language-aware: pass the transcript's detected language and the
matching keyword sets are used, so moment detection and scoring work on German
source videos as well as English. Unknown languages fall back to English.
The reason strings stay in English (the UI is English).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Word


@dataclass(frozen=True)
class Lexicon:
    hook: frozenset[str]
    emotion: frozenset[str]
    payoff: frozenset[str]
    dangling: frozenset[str]       # weak sentence openers
    second_person: frozenset[str]  # "you/your" equivalents
    quote_extra: frozenset[str]    # extra punchy/quotable words
    enumeration: frozenset[str]    # list/steps markers

    @property
    def quote(self) -> frozenset[str]:
        return self.hook | self.emotion | self.second_person | self.quote_extra


_EN = Lexicon(
    hook=frozenset({
        "how", "why", "what", "secret", "nobody", "everyone", "never", "always",
        "stop", "mistake", "truth", "reason", "actually", "surprising", "imagine",
        "warning", "honestly", "crazy", "wild", "ever", "biggest", "worst", "best",
    }),
    emotion=frozenset({
        "love", "hate", "fear", "amazing", "incredible", "terrible", "shocking",
        "insane", "beautiful", "painful", "hilarious", "scary", "exciting",
        "heartbreaking", "powerful", "frustrating", "grateful", "angry", "happy",
        "sad", "proud", "afraid", "excited", "wow", "unbelievable",
    }),
    payoff=frozenset({
        "because", "so", "therefore", "result", "realized", "learned", "lesson",
        "point", "means", "answer", "finally", "turns", "discovered", "secret",
        "key", "bottom", "ultimately", "conclusion",
    }),
    dangling=frozenset({
        "and", "but", "so", "or", "because", "which", "that", "it", "they", "he",
        "she", "this", "those", "these", "then", "also", "however",
    }),
    second_person=frozenset({"you", "your", "yourself"}),
    quote_extra=frozenset({
        "everything", "anything", "nothing", "life", "world", "matter", "moment",
        "change", "now",
    }),
    enumeration=frozenset({
        "first", "second", "third", "three", "two", "steps", "ways", "reasons", "tips",
    }),
)

_DE = Lexicon(
    hook=frozenset({
        "wie", "warum", "wieso", "weshalb", "was", "geheimnis", "niemand", "jeder",
        "nie", "niemals", "immer", "stopp", "fehler", "wahrheit", "grund",
        "eigentlich", "überraschend", "achtung", "ehrlich", "verrückt", "wahnsinn",
        "je", "größte", "schlimmste", "beste", "krass", "unglaublich", "nimm",
    }),
    emotion=frozenset({
        "liebe", "hasse", "angst", "erstaunlich", "unglaublich", "schrecklich",
        "schockierend", "wahnsinnig", "schön", "schmerzhaft", "lustig",
        "beängstigend", "aufregend", "herzzerreißend", "kraftvoll", "frustrierend",
        "dankbar", "wütend", "glücklich", "stolz", "begeistert", "wow", "krass",
    }),
    payoff=frozenset({
        "weil", "also", "deshalb", "deswegen", "ergebnis", "erkannt", "gelernt",
        "lektion", "punkt", "bedeutet", "antwort", "endlich", "entdeckt",
        "schlüssel", "letztendlich", "fazit", "darum",
    }),
    dangling=frozenset({
        "und", "aber", "oder", "weil", "der", "die", "das", "dass", "es", "sie",
        "er", "dies", "diese", "dann", "auch", "jedoch", "denn", "dieser",
    }),
    second_person=frozenset({"du", "dich", "dir", "dein", "deine", "deinen", "ihr", "euch", "euer"}),
    quote_extra=frozenset({
        "alles", "nichts", "etwas", "leben", "welt", "moment", "ändern", "jetzt", "heute",
    }),
    enumeration=frozenset({
        "erstens", "zweitens", "drittens", "drei", "zwei", "schritte", "wege",
        "gründe", "tipps", "punkte",
    }),
)

# --------------------------------------------------------------------------- #
# French lexicon
# --------------------------------------------------------------------------- #
_FR = Lexicon(
    hook=frozenset({
        "comment", "pourquoi", "quoi", "secret", "personne", "tout le monde",
        "jamais", "toujours", "arrête", "erreur", "vérité", "raison",
        "en fait", "surprenant", "imagine", "attention", "honnêtement",
        "fou", "dingue", "incroyable", "jamais", "plus grand", "pire", "meilleur",
    }),
    emotion=frozenset({
        "amour", "haine", "peur", "incroyable", "terrible", "choquant",
        "insensé", "magnifique", "douloureux", "hilarant", "effrayant",
        "excitant", "déchirant", "puissant", "frustrant", "reconnaissant",
        "fâché", "content", "triste", "fier", "effrayé", "excité", "wow",
        "incroyable", "génial",
    }),
    payoff=frozenset({
        "parce que", "donc", "par conséquent", "résultat", "réalisé", "appris",
        "leçon", "point", "signifie", "réponse", "enfin", "découvert",
        "secret", "clé", "fondamental", "finalement", "conclusion",
    }),
    dangling=frozenset({
        "et", "mais", "donc", "ou", "parce que", "qui", "que", "cela", "ils",
        "elles", "il", "elle", "ce", "ces", "cette", "alors", "aussi", "cependant",
    }),
    second_person=frozenset({"tu", "toi", "te", "ton", "ta", "tes", "vous", "votre"}),
    quote_extra=frozenset({
        "tout", "rien", "quelque chose", "vie", "monde", "importe", "moment",
        "changer", "maintenant", "aujourd'hui",
    }),
    enumeration=frozenset({
        "premièrement", "deuxièmement", "troisièmement", "trois", "deux",
        "étapes", "façons", "raisons", "conseils", "points",
    }),
)

# --------------------------------------------------------------------------- #
# Spanish lexicon
# --------------------------------------------------------------------------- #
_ES = Lexicon(
    hook=frozenset({
        "cómo", "por qué", "qué", "secreto", "nadie", "todo el mundo",
        "nunca", "siempre", "para", "error", "verdad", "razón",
        "en realidad", "sorprendente", "imagina", "cuidado", "honestamente",
        "loco", "salvaje", "increíble", "nunca", "más grande", "peor", "mejor",
    }),
    emotion=frozenset({
        "amor", "odio", "miedo", "increíble", "terrible", "impactante",
        "insano", "hermoso", "doloroso", "divertido", "aterrador",
        "emocionante", "desgarrador", "poderoso", "frustrante", "agradecido",
        "enojado", "feliz", "triste", "orgulloso", "asustado", "emocionado",
        "guau", "increíble",
    }),
    payoff=frozenset({
        "porque", "entonces", "por lo tanto", "resultado", "se dio cuenta",
        "aprendí", "lección", "punto", "significa", "respuesta", "finalmente",
        "descubrí", "secreto", "clave", "fondo", "últimamente", "conclusión",
    }),
    dangling=frozenset({
        "y", "pero", "entonces", "o", "porque", "que", "ello", "ellos",
        "él", "ella", "esto", "estos", "esas", "esos", "entonces", "también",
        "sin embargo",
    }),
    second_person=frozenset({"tú", "ti", "te", "tu", "tus", "vosotros", "vuestro",
                              "usted", "su", "sus"}),
    quote_extra=frozenset({
        "todo", "nada", "algo", "vida", "mundo", "importa", "momento",
        "cambiar", "ahora", "hoy",
    }),
    enumeration=frozenset({
        "primero", "segundo", "tercero", "tres", "dos",
        "pasos", "formas", "razones", "consejos", "puntos",
    }),
)

# --------------------------------------------------------------------------- #
# Portuguese lexicon
# --------------------------------------------------------------------------- #
_PT = Lexicon(
    hook=frozenset({
        "como", "por que", "o que", "segredo", "ninguém", "todo mundo",
        "nunca", "sempre", "pare", "erro", "verdade", "razão",
        "na verdade", "surpreendente", "imagine", "cuidado", "honestamente",
        "louco", "selvagem", "incrível", "nunca", "maior", "pior", "melhor",
    }),
    emotion=frozenset({
        "amor", "ódio", "medo", "incrível", "terrível", "chocante",
        "insano", "lindo", "doloroso", "hilário", "assustador",
        "emocionante", "devastador", "poderoso", "frustrante", "grato",
        "bravo", "feliz", "triste", "orgulhoso", "com medo", "animado",
        "uau", "inacreditável",
    }),
    payoff=frozenset({
        "porque", "então", "portanto", "resultado", "percebeu", "aprendi",
        "lição", "ponto", "significa", "resposta", "finalmente", "descobri",
        "segredo", "chave", "fundamental", "ultimamente", "conclusão",
    }),
    dangling=frozenset({
        "e", "mas", "então", "ou", "porque", "que", "isso", "eles",
        "ele", "ela", "este", "estes", "essas", "esses", "aí", "também",
        "contudo",
    }),
    second_person=frozenset({"tu", "ti", "te", "teu", "tua", "teus", "você",
                              "seu", "sua", "seus", "vocês"}),
    quote_extra=frozenset({
        "tudo", "nada", "algo", "vida", "mundo", "importa", "momento",
        "mudar", "agora", "hoje",
    }),
    enumeration=frozenset({
        "primeiro", "segundo", "terceiro", "três", "dois",
        "passos", "maneiras", "razões", "dicas", "pontos",
    }),
)

# --------------------------------------------------------------------------- #
# Italian lexicon
# --------------------------------------------------------------------------- #
_IT = Lexicon(
    hook=frozenset({
        "come", "perché", "cosa", "segreto", "nessuno", "tutti",
        "mai", "sempre", "fermati", "errore", "verità", "ragione",
        "in realtà", "sorprendente", "immagina", "attenzione", "onestamente",
        "pazzo", "selvaggio", "incredibile", "mai", "più grande", "peggiore", "migliore",
    }),
    emotion=frozenset({
        "amore", "odio", "paura", "incredibile", "terribile", "scioccante",
        "folle", "bellissimo", "doloroso", "divertente", "spaventoso",
        "emozionante", "straziante", "potente", "frustrante", "grato",
        "arrabbiato", "felice", "triste", "orgoglioso", "spaventato", "eccitato",
        "wow", "incredibile",
    }),
    payoff=frozenset({
        "perché", "quindi", "pertanto", "risultato", "realizzato", "imparato",
        "lezione", "punto", "significa", "risposta", "finalmente", "scoperto",
        "segreto", "chiave", "fondamentale", "ultimamente", "conclusione",
    }),
    dangling=frozenset({
        "e", "ma", "quindi", "o", "perché", "che", "questo", "loro",
        "lui", "lei", "questo", "questi", "quelle", "quelli", "poi", "anche",
        "tuttavia",
    }),
    second_person=frozenset({"tu", "te", "ti", "tuo", "tua", "tuoi", "tue",
                              "voi", "vostro", "vostra"}),
    quote_extra=frozenset({
        "tutto", "niente", "qualcosa", "vita", "mondo", "importa", "momento",
        "cambiare", "adesso", "oggi",
    }),
    enumeration=frozenset({
        "primo", "secondo", "terzo", "tre", "due",
        "passi", "modi", "ragioni", "consigli", "punti",
    }),
)

_LEXICONS: dict[str, Lexicon] = {"en": _EN, "de": _DE, "fr": _FR, "es": _ES, "pt": _PT, "it": _IT}


def get_lexicon(lang: str | None) -> Lexicon:
    return _LEXICONS.get((lang or "en").lower()[:2], _EN)


_WORD_RE = re.compile(r"[a-zà-ÿ']+", re.IGNORECASE)


def _tokens(words: list[Word]) -> list[str]:
    out: list[str] = []
    for w in words:
        out.extend(m.lower() for m in _WORD_RE.findall(w.text))
    return out


def _has_number(words: list[Word]) -> bool:
    return any(re.search(r"\d", w.text) for w in words)


def _ends_complete(words: list[Word]) -> bool:
    return bool(words) and words[-1].text.strip().endswith((".", "!", "?", '"'))


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
# Features. Each returns (value 0..1, reason). `lex` defaults to English.
# --------------------------------------------------------------------------- #
def hook_strength(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    """Curiosity in the first ~3 seconds — does it stop the scroll?

    The feed commits in the first ~1.5s, so a hook that lands *immediately*
    (a question, number, curiosity word or direct address in the opening beat)
    is rewarded above one that only arrives by second three.
    """
    if not words:
        return 0.0, ""
    start = words[0].t
    head = [w for w in words if w.t - start < 3.0] or words[:8]
    head_early = [w for w in words if w.t - start < 1.5] or words[:4]
    toks = _tokens(head)
    if not toks:
        return 0.0, ""
    toks_early = _tokens(head_early)
    hooks = sum(t in lex.hook for t in toks)
    second_person = sum(t in lex.second_person for t in toks)
    question = any(w.text.strip().endswith("?") for w in head)
    number = _has_number(head)
    # Front-loaded: a hook word / question / number / "you" inside the opening ~1.5s.
    early = (any(t in lex.hook for t in toks_early)
             or any(w.text.strip().endswith("?") for w in head_early)
             or _has_number(head_early)
             or any(t in lex.second_person for t in toks_early))
    raw = 0.30 * min(hooks, 2) / 2 + 0.18 * min(second_person, 2) / 2
    raw += 0.24 * question + 0.12 * number + 0.16 * early
    reason = ("Hooks in the first 1.5s" if early and (hooks or question or number) else
              "Opens with a question" if question else (
                  "Curiosity hook up front" if hooks else (
                      "Speaks directly to the viewer" if second_person else "Opens on a concrete detail")))
    return _clamp(raw), reason


_FILLER_OPENERS = frozenset({
    "um", "uh", "okay", "ok", "so", "well", "like", "yeah", "also", "äh",
    "ehm", "ja", "naja", "okay",
})


def instant_hook(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    """Scroll-stop strength inside the first two seconds."""
    if not words:
        return 0.0, ""
    start = words[0].t
    head = [w for w in words if w.t - start < 2.0] or words[:5]
    toks = _tokens(head)
    if not toks:
        return 0.0, ""
    first = toks[0]
    hook = any(t in lex.hook for t in toks)
    direct = any(t in lex.second_person for t in toks)
    question = any(w.text.strip().endswith("?") for w in head)
    number = _has_number(head)
    weak = first in (lex.dangling | _FILLER_OPENERS)
    raw = 0.30 * hook + 0.22 * direct + 0.22 * question + 0.18 * number
    raw += 0.10 if not weak else -0.20
    reason = ("First 2s hook lands" if (hook or direct or question or number) else
              "Clean opening beat" if not weak else "Soft opening beat")
    return _clamp(raw), reason


def swipe_resistance(words: list[Word], duration: float,
                     lex: Lexicon = _EN) -> tuple[float, str]:
    """Positive proxy for low swipe-away risk."""
    if not words or duration <= 0:
        return 0.0, ""
    start = words[0].t
    first_two = [w for w in words if w.t - start < 2.0] or words[:5]
    toks = _tokens(first_two)
    first = toks[0] if toks else ""
    weak_open = first in (lex.dangling | _FILLER_OPENERS)
    hook_val, _ = instant_hook(words, lex)
    early_wps = len(first_two) / max(min(duration, 2.0), 0.5)
    pace_ok = 1.0 if 1.4 <= early_wps <= 4.8 else 0.35
    complete_open = 1.0 if first_two and len(first_two) >= 3 else 0.45
    raw = 0.55 * hook_val + 0.25 * pace_ok + 0.20 * complete_open
    if weak_open:
        raw -= 0.25
    reason = "Low swipe-away risk" if raw >= 0.55 else "Opening may lose viewers"
    return _clamp(raw), reason


def emotional_payoff(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    toks = _tokens(words)
    if not toks:
        return 0.0, ""
    hits = sum(t in lex.emotion for t in toks)
    density = hits / max(len(toks), 1)
    return _clamp(density * 9.0), "Clear emotional charge"


def standalone_clarity(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    """Does it stand on its own — clean open, complete close?"""
    if not words:
        return 0.0, ""
    first = _WORD_RE.findall(words[0].text.lower())
    dangling = bool(first) and first[0] in lex.dangling
    complete = _ends_complete(words)
    raw = (0.0 if dangling else 0.55) + (0.45 if complete else 0.0)
    reason = "Clean standalone story" if not dangling and complete else (
        "Resolves cleanly" if complete else "Self-contained")
    return _clamp(raw), reason


def pace_energy(words: list[Word], duration: float) -> tuple[float, str]:
    if duration <= 0:
        return 0.0, ""
    wps = len(words) / duration
    if wps < 1.4 or wps > 4.5:
        raw = 0.2
    else:
        raw = 1.0 - abs(wps - 2.8) / 2.0
    return _clamp(raw), "Energetic, well-paced delivery"


def quotability(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    """A short, punchy line lands well as a standalone quote."""
    if not words:
        return 0.0, ""
    toks = _tokens(words)
    strong = sum(t in lex.quote for t in toks)
    short_punchy = len(words) <= 28
    raw = _clamp(0.5 * min(strong, 4) / 4 + (0.5 if short_punchy else 0.2))
    return raw, "Quotable, punchy line"


def length_fit(duration: float, min_len: float, max_len: float) -> tuple[float, str]:
    ideal = (min_len + max_len) / 2.0
    span = max((max_len - min_len) / 2.0, 1.0)
    raw = _clamp(1.0 - abs(duration - ideal) / (span * 1.6))
    return raw, "Ideal length for the format"


def replay_value(words: list[Word], duration: float, lex: Lexicon = _EN) -> tuple[float, str]:
    """Rewatch/loopability — does it end on a clean 'button' that invites a loop?

    Short clips that finish on a complete, punchy line read as a tight loop, and
    the feeds reward rewatches. We reward: a complete close (terminal punctuation),
    a strong/quotable final beat, and a concise length; we penalise trailing off
    on a weak connective ("and…", "but…", "so…").
    """
    if not words:
        return 0.0, ""
    complete = _ends_complete(words)
    tail = _tokens(words[-4:])
    strong_close = any(t in (lex.hook | lex.emotion | lex.quote_extra) for t in tail)
    last = (_WORD_RE.findall(words[-1].text.lower()) or [""])[-1]
    dangling_close = last in lex.dangling
    concise = duration <= 35.0
    raw = (0.5 if complete else 0.0) + (0.25 if strong_close else 0.0) \
        + (0.25 if concise else 0.0) - (0.35 if dangling_close else 0.0)
    return _clamp(raw), "Loops cleanly — ends on a tight button"


def list_payoff(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    has_num = _has_number(words)
    toks = _tokens(words)
    enumeration = any(t in lex.enumeration for t in toks)
    raw = _clamp(0.6 * has_num + 0.6 * enumeration)
    return raw, "Concrete, list-style payoff"
