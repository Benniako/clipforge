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
    controversy: frozenset[str] = frozenset()  # polarising/argument-triggering words

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
    controversy=frozenset({
        "controversial", "unpopular", "opinion", "actually", "wrong", "hate",
        "nobody tells you", "truth", "real reason", "why everyone", "change my mind",
        "fight me", "prove me wrong", "hot take", "debate", "argue",
        "overrated", "underrated", "worst", "terrible", "awful",
        "cancel", "toxic", "problematic", "divisive",
        "conspiracy", "unpopular opinion",
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
    controversy=frozenset({
        "kontrovers", "unpopular", "meinung", "eigentlich", "falsch", "hassen",
        "sagt niemand", "wahrheit", "echter grund", "warum alle", "beweis mir",
        "heißer take", "diskutieren", "streiten", "überbewertet",
        "unterbewertet", "schlimmste", "furchtbar",
        "toxisch", "problematisch", "spaltend",
        "verschwörung", "unbeliebte meinung",
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



def get_lexicon(lang: str | None) -> Lexicon:
    return _LEXICONS.get((lang or "en").lower()[:2], _EN)


_WORD_RE = re.compile(r"[a-zà-ÿ']+", re.IGNORECASE)


def _tokens(words: list[Word]) -> list[str]:
    """Extract tokens plus bigrams from words, so multi-word lexicon entries
    (e.g. French 'en fait', Spanish 'por qué') actually match.

    Returns both single words and adjacent pairs: ["en", "fait", "en fait"].
    This inflates the token count slightly but the matching functions use
    ``t in frozenset`` lookups, so false positives from partial matches are
    prevented by the set membership check — a bigram "en fait" won't match
    a unigram-only lexicon entry.
    """
    singles: list[str] = []
    for w in words:
        singles.extend(m.lower() for m in _WORD_RE.findall(w.text))
    if len(singles) < 2:
        return singles
    bigrams = [f"{singles[i]} {singles[i+1]}" for i in range(len(singles) - 1)]
    return singles + bigrams


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


def hook_potential_index(words: list[Word], duration: float,
                         lex: Lexicon = _EN, *,
                         clip_start: float | None = None) -> tuple[float, str]:
    """Algorithmic Hook Potential Index (HPI) over the opening 3-5 seconds.

    Combines linguistic curiosity (S_ling), speech pacing (S_pace),
    Voice Onset Time penalty (Phi_VOT for silence before speech), and filler deductions.
    Strictly bounded in [0.0, 1.0].
    """
    if not words or duration <= 0:
        return 0.0, ""

    start = words[0].t
    head_3s = [w for w in words if w.t - start < 3.5] or words[:8]
    toks = _tokens(head_3s)
    if not toks:
        return 0.0, ""

    # 1. Linguistic Curiosity (S_ling)
    hooks = sum(t in lex.hook for t in toks)
    second_person = sum(t in lex.second_person for t in toks)
    question = any(w.text.strip().endswith("?") for w in head_3s)
    has_num = _has_number(head_3s)
    first_tok = toks[0]
    is_filler = first_tok in (lex.dangling | _FILLER_OPENERS)

    s_ling = 0.35 * min(hooks, 2) / 2.0 + 0.25 * min(second_person, 2) / 2.0
    s_ling += 0.25 * float(question) + 0.15 * float(has_num)
    if is_filler:
        s_ling = max(0.0, s_ling - 0.25)
    s_ling = min(1.0, s_ling)

    # 2. Speech Pacing (S_pace): Target 165 - 205 WPM (2.75 - 3.4 words/s)
    window_dur = max(head_3s[-1].end - start, 0.5)
    wps = len(head_3s) / window_dur
    if 2.4 <= wps <= 3.8:
        s_pace = 1.0
    elif 1.8 <= wps < 2.4 or 3.8 < wps <= 4.5:
        s_pace = 0.75
    elif wps < 1.4:
        s_pace = 0.35
    else:
        s_pace = 0.50

    # 3. Voice Onset Time penalty (Phi_VOT) - strict 0.15s dead-air elimination
    vot_mult = 1.0
    if clip_start is not None and clip_start >= 0:
        vot = max(0.0, start - clip_start)
        if vot > 0.15:
            import math
            vot_mult = max(0.1, min(1.0, math.exp(-(vot - 0.15) / 0.40)))

    # Composite HPI
    hpi = (0.55 * s_ling + 0.45 * s_pace) * vot_mult
    if is_filler:
        hpi = max(0.0, hpi - 0.15)
    hpi = max(0.0, min(1.0, hpi))

    reason = ("High-energy curiosity hook" if hpi >= 0.75 and (hooks or question) else
              "Immediate question opener" if question else
              "Direct address hook" if second_person else
              "Fast, engaging opening pace" if s_pace >= 0.9 else
              "Delayed speech onset" if vot_mult < 0.7 else
              "Clean opening beat" if not is_filler else "Soft opener with filler")

    return round(hpi, 4), reason


def emotional_payoff(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    toks = _tokens(words)
    if not toks:
        return 0.0, ""
    hits = sum(t in lex.emotion for t in toks)
    density = hits / max(len(toks), 1)
    return _clamp(density * 9.0), "Clear emotional charge"


def climax_narrative_pacing(words: list[Word], duration: float,
                            lex: Lexicon = _EN) -> tuple[float, float, str]:
    """Score adherence to the Golden Climax Interval (72%-88% of clip duration).

    In short-form video retention dynamics (TikTok/Shorts), viral engagement is
    maximized when the emotional or comedic punchline lands at 75%-85% into the
    clip, leaving 2-4 seconds for reaction and clean resolution.

    Returns:
        (pacing_score 0..1, climax_timestamp, reason)
    """
    if not words or duration <= 0:
        return 0.5, 0.0, ""

    start_t = words[0].t

    # Score each word or local window for climax energy
    best_energy = -1.0
    best_t = start_t + 0.75 * duration  # default fallback if completely flat

    for w in words:
        txt = w.text.strip().lower()
        energy = 0.5
        # Exclamations and punctuation punchlines
        if w.text.strip().endswith("!"):
            energy += 2.5
        elif w.text.strip().endswith("?"):
            energy += 1.2
        # Lexical signals: emotion, payoff, controversy
        clean_toks = _WORD_RE.findall(txt)
        for t in clean_toks:
            if t in lex.emotion:
                energy += 2.0
            if t in lex.payoff:
                energy += 1.5
            if t in lex.controversy:
                energy += 1.2
            if t in lex.quote:
                energy += 0.8
        if len(txt) > 7:
            energy += 0.4

        if energy > best_energy:
            best_energy = energy
            best_t = w.t + (w.end - w.t) * 0.5

    # Normalized position tau*
    tau = max(0.0, min(1.0, (best_t - start_t) / max(duration, 0.1)))

    # Golden interval: [0.72, 0.88]
    if 0.72 <= tau <= 0.88:
        score = 1.0
        reason = f"Climax lands at optimal {int(tau * 100)}% narrative arc"
    elif 0.60 <= tau < 0.72:
        score = 0.75 + (tau - 0.60) / 0.12 * 0.25
        reason = f"Good build-up to climax at {int(tau * 100)}%"
    elif 0.88 < tau <= 0.94:
        score = 0.75 + (0.94 - tau) / 0.06 * 0.25
        reason = f"Late punchline at {int(tau * 100)}%"
    elif tau < 0.60:
        score = max(0.15, 0.75 - (0.60 - tau) * 1.5)
        reason = f"Climax occurs early at {int(tau * 100)}% (potential viewer drop-off)"
    else:  # tau > 0.94
        score = max(0.15, 0.75 - (tau - 0.94) * 8.0)
        reason = f"Climax cut off too abruptly at {int(tau * 100)}%"

    return round(_clamp(score), 4), round(best_t, 3), reason


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


def controversy(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    """Polarising/argument-triggering language that drives comments.

    Controversial statements are the strongest engagement driver on short-form
    platforms — people comment to argue, correct, or agree. We detect trigger
    phrases (hot takes, unpopular opinions, direct challenges) and rate the
    clip's controversy density.
    """
    if not words:
        return 0.0, ""
    toks = _tokens(words)
    if not toks:
        return 0.0, ""
    hits = sum(1 for t in toks if t in lex.controversy)
    # Multi-word phrases need sliding window matching
    text = " ".join(toks).lower()
    phrase_hits = sum(1 for p in lex.controversy
                      if " " in p and p in text)
    total = hits + phrase_hits * 2  # phrases are stronger signals
    density = total / max(len(toks), 1)
    raw = _clamp(density * 15.0)
    return raw, "Controversial — drives comments"


def qa_pattern(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    """Does the clip start with a question and answer within its span?

    Question→answer clips have high retention (the viewer stays for the answer)
    and high shareability. We check if any sentence near the start ends with
    '?', and if the following text contains an answer marker.
    """
    if not words:
        return 0.0, ""
    # Find question marks in the first ~40% of words
    split_at = max(len(words) // 5, 3)  # first 20% or 3 words
    head = words[:split_at]
    question = any(w.text.rstrip().endswith("?") for w in head)
    if not question:
        return 0.0, ""
    # Look for answer markers after the question
    answer_markers = {"because", "so", "reason", "explain", "means", "actually",
                      "turns out", "the truth", "answer", "weil", "deshalb",
                      "deswegen", "erklärung", "bedeutet", "eigentlich"}
    rest = " ".join(w.text.lower() for w in words[split_at:])
    has_answer = any(m in rest for m in answer_markers)
    raw = _clamp(0.8 if has_answer else 0.4)
    return raw, "Question triggers then answers — drives retention"
# --------------------------------------------------------------------------- #
# Turkish lexicon
# --------------------------------------------------------------------------- #
_TR = Lexicon(
    hook=frozenset({
        "nasil", "neden", "ne", "sir", "kimse", "herkes",
        "asla", "her zaman", "dur", "hata", "gercek", "sebep",
        "aslinda", "sasirtici", "hayal et", "uyari", "dürüst",
        "cilgin", "vahsi", "hic", "en büyük", "en kötü", "en iyi",
    }),
    emotion=frozenset({
        "ask", "nefret", "korku", "harika", "inanilmaz", "korkunç",
        "sok edici", "deli", "güzel", "aci verici", "komik",
        "korkutucu", "heyecan verici", "yürek parçalayici", "güçlü",
        "sinir bozucu", "minnettar", "kizgin", "mutlu", "üzgün",
        "gururlu", "korkmus", "heyecanli", "vay", "inanilmaz",
    }),
    payoff=frozenset({
        "çünkü", "bu yüzden", "bu nedenle", "sonuç", "fark ettim",
        "ögrendim", "ders", "nokta", "anlamina gelir", "cevap",
        "sonunda", "kesfettim", "sir", "anahtar", "dip",
        "nihayetinde", "sonuç",
    }),
    dangling=frozenset({
        "ve", "ama", "bu yüzden", "veya", "çünkü", "hangi", "bu",
        "onlar", "o", "su", "bunlar", "sonra", "ayrica", "ancak",
    }),
    second_person=frozenset({"sen", "sana", "seni", "senin", "sizin", "kendin"}),
    quote_extra=frozenset({
        "her sey", "hicbir sey", "bir sey", "hayat", "dünya",
        "önemli", "an", "degisim", "simdi",
    }),
    enumeration=frozenset({
        "birinci", "ikinci", "üçüncü", "üç", "iki",
        "adimlar", "yollar", "nedenler", "ipuçlari",
    }),
)

# --------------------------------------------------------------------------- #
# Japanese lexicon (romaji)
# --------------------------------------------------------------------------- #
_JP = Lexicon(
    hook=frozenset({
        "naze", "doushite", "nani", "himitsu", "daremo", "minna",
        "kesshite", "itsumo", "yamete", "ayamari", "shinjitsu", "riyuu",
        "jitsuwa", "odoroki", "souzou", "keikoku", "shoujiki",
        "kurutte", "yabai", "ichiban", "saitei", "saikou",
    }),
    emotion=frozenset({
        "ai", "nikushimi", "kyoufu", "subarashii", "fushigi", "hidoi",
        "shougeki", "okashii", "utsukushii", "kurushii", "omoshiroi",
        "kowai", "wakuwaku", "setsunai", "tsuyoi", "iraira",
        "kansha", "okotte", "shiawase", "kanashii", "hokori",
        "obiete", "waku", "sugoi", "shinjirarenai",
    }),
    payoff=frozenset({
        "dakara", "nazenara", "kekka", "kidzuita", "mananda",
        "lesson", "pointo", "imi", "kotae", "tsui ni",
        "hakken", "himitsu", "kagi", "kekkyoku", "ketsuron",
    }),
    dangling=frozenset({
        "soshite", "demo", "dakara", "mata", "keredo", "sore", "kore",
        "karera", "kare", "kanojo", "kono", "sono", "ano",
        "soshite", "mata", "datte",
    }),
    second_person=frozenset({"anata", "kimi", "anta", "anata no", "kimi no", "jibun"}),
    quote_extra=frozenset({
        "subete", "nani mo", "nanika", "jinsei", "sekai",
        "mondai", "shunkan", "henka", "ima", "kyou",
    }),
    enumeration=frozenset({
        "ichiban", "niban", "sanban", "mittsu", "futatsu",
        "suteppu", "houhou", "riyuu", "hinto",
    }),
)

# --------------------------------------------------------------------------- #
# Dutch lexicon
# --------------------------------------------------------------------------- #
_NL = Lexicon(
    hook=frozenset({
        "hoe", "waarom", "wat", "geheim", "niemand", "iedereen",
        "nooit", "altijd", "stop", "fout", "waarheid", "reden",
        "eigenlijk", "verrassend", "stel je voor", "waarschuwing",
        "eerlijk", "gek", "wild", "ooit", "grootste", "slechtste", "beste",
    }),
    emotion=frozenset({
        "liefde", "haat", "angst", "geweldig", "ongelooflijk",
        "verschrikkelijk", "schokkend", "krankzinnig", "mooi",
        "pijnlijk", "hilarisch", "eng", "opwindend",
        "hartverscheurend", "krachtig", "frustrerend", "dankbaar",
        "boos", "blij", "verdrietig", "trots", "bang", "opgewonden",
        "wauw", "ongelooflijk",
    }),
    payoff=frozenset({
        "omdat", "dus", "daarom", "resultaat", "besefte", "geleerd",
        "les", "punt", "betekent", "antwoord", "eindelijk", "ontdekt",
        "geheim", "sleutel", "bodem", "uiteindelijk", "conclusie",
    }),
    dangling=frozenset({
        "en", "maar", "dus", "of", "omdat", "welke", "dat", "het",
        "zij", "hij", "deze", "die", "dan", "ook", "echter",
    }),
    second_person=frozenset({"jij", "jou", "jouw", "je", "u", "uzelf", "jezelf"}),
    quote_extra=frozenset({
        "alles", "niets", "iets", "leven", "wereld", "zaak",
        "moment", "veranderen", "nu", "vandaag",
    }),
    enumeration=frozenset({
        "eerste", "tweede", "derde", "drie", "twee",
        "stappen", "manieren", "redenen", "tips", "punten",
    }),
)

# --------------------------------------------------------------------------- #
# Polish lexicon
# --------------------------------------------------------------------------- #
_PL = Lexicon(
    hook=frozenset({
        "jak", "dlaczego", "co", "sekret", "nikt", "wszyscy",
        "nigdy", "zawsze", "przestan", "blad", "prawda", "powod",
        "wlasciwie", "zaskakujacy", "wyobraz sobie", "ostrzezenie",
        "szczerze", "szalony", "dziki", "kiedykolwiek",
        "najwiekszy", "najgorszy", "najlepszy",
    }),
    emotion=frozenset({
        "milosc", "nienawisc", "strach", "niesamowity",
        "niewiarygodny", "straszny", "szokujacy", "oblokany",
        "piekny", "bolesny", "zabawny", "przerazajacy", "ekscytujacy",
        "rozrywajacy serce", "potrzny", "frustrujacy", "wdzieczny",
        "zly", "szczesliwy", "smutny", "dumny", "przestraszony",
        "podekscytowany", "wal", "niewiarygodne",
    }),
    payoff=frozenset({
        "poniewaz", "wiec", "dlatego", "wynik", "zdalem sobie sprawe",
        "nauczylem sie", "lekcja", "punkt", "znaczy", "odpowiedz",
        "w koncu", "odkrylem", "sekret", "klucz", "dno",
        "ostatecznie", "wniosek",
    }),
    dangling=frozenset({
        "i", "ale", "wiec", "lub", "poniewaz", "ktory", "ze", "to",
        "oni", "on", "ona", "ten", "ci", "te", "wtedy", "rowniez",
        "jednak",
    }),
    second_person=frozenset({"ty", "ciebie", "cie", "twoj", "twoja", "twoje",
                              "panski", "wasz"}),
    quote_extra=frozenset({
        "wszystko", "nic", "cos", "zycie", "swiat", "sprawa",
        "moment", "zmienic", "teraz", "dzis",
    }),
    enumeration=frozenset({
        "pierwszy", "drugi", "trzeci", "trzy", "dwa",
        "kroki", "sposoby", "powody", "wskazowki", "punkty",
    }),
)

# --------------------------------------------------------------------------- #
# Swedish lexicon
# --------------------------------------------------------------------------- #
_SV = Lexicon(
    hook=frozenset({
        "hur", "varfor", "vad", "hemlighet", "ingen", "alla",
        "aldrig", "alltid", "sluta", "misstag", "sanning", "anledning",
        "egentligen", "overraskande", "forestall dig", "varning",
        "arligt", "galen", "vild", "nagonsin", "storsta", "samsta", "basta",
    }),
    emotion=frozenset({
        "kärlek", "hat", "radsla", "fantastisk", "otrolig",
        "fruktansvard", "chockerande", "vansinnig", "vacker",
        "smartsam", "rolig", "skrammande", "spännande",
        "hjartskarande", "kraftfull", "frustrerande", "tacksam",
        "arg", "glad", "ledsen", "stolt", "radd", "upprymd",
        "wow", "otroligt",
    }),
    payoff=frozenset({
        "for att", "sa", "darfor", "resultat", "insag", "larde",
        "laxa", "poang", "betyder", "svar", "antligen", "upptackte",
        "hemlighet", "nyckel", "botten", "slutligen", "slutsats",
    }),
    dangling=frozenset({
        "och", "men", "sa", "eller", "for att", "vilken", "som", "det",
        "de", "han", "hon", "denna", "dessa", "de dar", "sedan", "ocksa",
        "dock",
    }),
    second_person=frozenset({"du", "dig", "din", "ditt", "dina", "er", "ert", "era"}),
    quote_extra=frozenset({
        "allt", "inget", "nagot", "liv", "varld", "sak",
        "ogonblick", "forandra", "nu", "idag",
    }),
    enumeration=frozenset({
        "forsta", "andra", "tredje", "tre", "tva",
        "steg", "satt", "anledningar", "tips", "punkter",
    }),
)

_LEXICONS: dict[str, Lexicon] = {"en": _EN, "de": _DE, "fr": _FR, "es": _ES, "pt": _PT, "it": _IT,
	                                  "tr": _TR, "jp": _JP, "nl": _NL, "pl": _PL, "sv": _SV}


def continuous_emotion_curve(words: list[Word], duration: float, hop: float = 0.1,
                             lex: Lexicon = _EN) -> tuple[list[float], list[float]]:
    """Compute a continuous 10 Hz emotional energy curve over the clip duration."""
    if duration <= 0:
        return [], []
    import numpy as np
    n = max(1, int(round(duration / hop)))
    times = [round(i * hop, 3) for i in range(n)]
    curve = np.zeros(n, dtype=np.float32)

    for w in words:
        txt = w.text.strip().lower()
        charge = 0.20  # spoken speech floor
        if w.text.strip().endswith("!"):
            charge += 0.50
        elif w.text.strip().endswith("?"):
            charge += 0.25
        clean_toks = _WORD_RE.findall(txt)
        for t in clean_toks:
            if t in lex.emotion:
                charge += 0.45
            if t in lex.payoff:
                charge += 0.35
            if t in lex.controversy:
                charge += 0.25
        i0 = max(0, int(round(w.t / hop)))
        i1 = min(n, int(round((w.t + w.d) / hop)))
        if i1 > i0:
            curve[i0:i1] = np.maximum(curve[i0:i1], min(1.0, charge))

    return times, [round(float(v), 4) for v in curve]
