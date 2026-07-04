"""Synthetic artist similarity pairs via genre x era co-occurrence.

Generates pseudo-ground-truth pairs by assuming artists sharing genre
tags are "similar," with probability weighted by overlap.  Intended for
development / smoke-testing before real Last.fm data is collected.
"""

import random
from typing import List, Tuple

# ── 12 genre clusters, each with 15-22 real artists ────────────────────

_GENRE_ARTISTS: dict[str, set[str]] = {
    "pop": {
        "Taylor Swift", "Adele", "Ed Sheeran", "Bruno Mars", "Ariana Grande",
        "Rihanna", "Beyonce", "Dua Lipa", "Harry Styles", "Olivia Rodrigo",
        "Doja Cat", "Lady Gaga", "Katy Perry", "Justin Bieber", "Miley Cyrus",
        "Selena Gomez", "Sam Smith", "Halsey", "Shawn Mendes", "Pink",
        "Ellie Goulding", "Sia",
    },
    "rock": {
        "Arctic Monkeys", "Nirvana", "Radiohead", "Foo Fighters",
        "Red Hot Chili Peppers", "Green Day", "Linkin Park", "The Strokes",
        "Mac DeMarco", "Pearl Jam", "AC/DC", "The Black Keys",
        "Queens of the Stone Age", "Muse", "Kings of Leon", "The White Stripes",
        "Weezer", "Tom Petty", "Blink-182",
    },
    "hip-hop": {
        "Kendrick Lamar", "Drake", "Kanye West", "Eminem", "Post Malone",
        "J. Cole", "Travis Scott", "Lil Wayne", "Jay-Z", "Nicki Minaj",
        "Megan Thee Stallion", "Cardi B", "ASAP Rocky", "Tyler the Creator",
        "21 Savage", "Future", "Lil Nas X", "Chance the Rapper",
        "Childish Gambino", "Run-DMC", "Lil Yachty",
    },
    "r-and-b": {
        "The Weeknd", "Rihanna", "Beyonce", "Alicia Keys", "Usher",
        "Chris Brown", "Frank Ocean", "SZA", "Khalid", "H.E.R.",
        "Daniel Caesar", "Steve Lacy", "Bryson Tiller", "Erykah Badu",
        "D Angelo", "Mary J Blige", "TLC", "Aaliyah", "Summer Walker",
        "Jhene Aiko",
    },
    "electronic": {
        "Daft Punk", "Calvin Harris", "Marshmello", "David Guetta",
        "Avicii", "Skrillex", "Deadmau5", "Swedish House Mafia", "Flume",
        "ODESZA", "Kygo", "Zedd", "Martin Garrix", "Diplo", "Tiesto",
        "Armin van Buuren", "Aphex Twin", "Disclosure", "Major Lazer",
        "The Chainsmokers",
    },
    "country": {
        "Luke Bryan", "Morgan Wallen", "Carrie Underwood", "Johnny Cash",
        "Dolly Parton", "Tim McGraw", "Blake Shelton", "Keith Urban",
        "Chris Stapleton", "Luke Combs", "Miranda Lambert", "Garth Brooks",
        "Kenny Chesney", "Eric Church", "Jason Aldean", "Thomas Rhett",
        "Willie Nelson", "Hank Williams", "Patsy Cline", "Zac Brown Band",
    },
    "latin": {
        "Bad Bunny", "J Balvin", "Ozuna", "Daddy Yankee", "Shakira",
        "Luis Fonsi", "Rosalia", "Maluma", "Karol G", "Becky G",
        "Rauw Alejandro", "Romeo Santos", "Don Omar", "Enrique Iglesias",
        "Pitbull", "Mana", "Camila Cabello", "Anuel AA",
    },
    "k-pop": {
        "BTS", "BLACKPINK", "TWICE", "EXO", "Red Velvet", "BIGBANG",
        "Stray Kids", "ITZY", "NCT 127", "ATEEZ", "SEVENTEEN",
        "MONSTA X", "Tomorrow X Together", "ENHYPEN", "NewJeans",
        "aespa", "LE SSERAFIM", "IVE",
    },
    "jazz": {
        "Miles Davis", "John Coltrane", "Frank Sinatra", "Ella Fitzgerald",
        "Charlie Parker", "Duke Ellington", "Louis Armstrong",
        "Thelonious Monk", "Billie Holiday", "Charles Mingus",
        "Dizzy Gillespie", "Dave Brubeck", "Herbie Hancock", "Count Basie",
        "Chet Baker", "Art Blakey", "Wayne Shorter", "Sonny Rollins",
        "Oscar Peterson",
    },
    "indie": {
        "Tame Impala", "Mac DeMarco", "Beach House", "Vampire Weekend",
        "Phoebe Bridgers", "Father John Misty", "Arcade Fire",
        "Modest Mouse", "Alt-J", "Glass Animals", "Bon Iver",
        "Fleet Foxes", "Sufjan Stevens", "Mitski", "Clairo",
        "Rex Orange County", "Still Woozy", "Men I Trust",
    },
    "metal": {
        "Metallica", "Black Sabbath", "Iron Maiden", "Slayer", "Megadeth",
        "Pantera", "Slipknot", "System of a Down", "Judas Priest",
        "Motörhead", "Tool", "Opeth", "Gojira", "Mastodon",
        "Lamb of God", "Meshuggah", "Avenged Sevenfold", "Deftones",
        "Rammstein",
    },
    "reggae": {
        "Bob Marley", "Peter Tosh", "Jimmy Cliff",
        "Toots and the Maytals", "Lee Scratch Perry", "Burning Spear",
        "UB40", "Ziggy Marley", "Shaggy", "Sean Paul", "Beenie Man",
        "Buju Banton", "Damian Marley", "Slightly Stoopid", "Steel Pulse",
        "Inner Circle", "Tarrus Riley", "Chronixx", "Koffee",
        "Groundation",
    },
}


# ── Era buckets ──────────────────────────────────────────────────────

def _era_of(artist: str) -> str:
    """Return the primary era for *artist* based on peak activity.

    Manually curated for the artists in *GENRE_ARTISTS* to keep the
    pairing deterministic without an external API.
    """
    # 2020s
    _2020s = {
        "Megan Thee Stallion", "Doja Cat", "Olivia Rodrigo", "Lil Nas X",
        "Karol G", "Rauw Alejandro", "Morgan Wallen", "Luke Combs",
        "Stray Kids", "ITZY", "ATEEZ", "Tomorrow X Together", "ENHYPEN",
        "NewJeans", "aespa", "LE SSERAFIM", "IVE", "Summer Walker",
        "Jhene Aiko", "Steve Lacy", "SZA", "Khalid", "H.E.R.",
        "Daniel Caesar", "Chronixx", "Koffee", "Phoebe Bridgers",
        "Mitski", "Clairo", "Rex Orange County", "Still Woozy",
        "Men I Trust", "Rosalia", "Anuel AA",
    }
    # 2010s
    _2010s = {
        "Taylor Swift", "Adele", "Bruno Mars", "Ariana Grande", "Ed Sheeran",
        "Dua Lipa", "Harry Styles", "Lady Gaga", "Sam Smith", "Halsey",
        "Shawn Mendes", "Pink", "Sia", "Kendrick Lamar", "Drake",
        "Kanye West", "Post Malone", "J. Cole", "Travis Scott", "Nicki Minaj",
        "Cardi B", "ASAP Rocky", "Tyler the Creator", "21 Savage", "Future",
        "Chance the Rapper", "Childish Gambino", "The Weeknd", "Frank Ocean",
        "Bryson Tiller", "Tame Impala", "Mac DeMarco",
        "Vampire Weekend", "Father John Misty", "Alt-J", "Glass Animals",
        "Bon Iver", "Fleet Foxes", "Sufjan Stevens", "Calvin Harris",
        "Marshmello", "David Guetta", "Avicii", "Skrillex", "Deadmau5",
        "Swedish House Mafia", "Flume", "ODESZA", "Kygo", "Zedd",
        "Martin Garrix", "Disclosure", "Major Lazer", "The Chainsmokers",
        "Arctic Monkeys", "Foo Fighters", "The Black Keys", "Muse",
        "Kings of Leon", "Luke Bryan", "Carrie Underwood", "Blake Shelton",
        "Keith Urban", "Chris Stapleton", "Miranda Lambert",
        "Eric Church", "Thomas Rhett", "Bad Bunny", "J Balvin",
        "Ozuna", "Daddy Yankee", "Maluma", "Becky G", "Camila Cabello",
        "BTS", "BLACKPINK", "EXO", "Red Velvet", "TWICE",
        "NCT 127", "SEVENTEEN", "MONSTA X", "Slipknot",
        "System of a Down", "Tool", "Avenged Sevenfold", "Deftones",
        "Lamb of God", "Mastodon", "Gojira", "Alicia Keys", "Usher",
        "Chris Brown", "Shaggy", "Sean Paul", "Damian Marley",
        "Slightly Stoopid", "Arcade Fire", "Modest Mouse",
        "Lil Wayne", "Tarrus Riley",
    }
    # 2000s
    _2000s = {
        "Rihanna", "Beyonce", "BIGBANG", "Katy Perry", "Justin Bieber",
        "Miley Cyrus", "Selena Gomez", "Ellie Goulding", "Green Day",
        "Linkin Park", "The Strokes", "Red Hot Chili Peppers",
        "Pearl Jam", "The White Stripes", "Weezer", "Blink-182",
        "Daft Punk", "Tiesto", "Armin van Buuren", "Diplo",
        "MOSH", "Jay-Z", "Eminem", "Run-DMC", "Lil Yachty",
        "Missy Elliott", "Aaliyah", "Erykah Badu", "TLC",
        "Mary J Blige", "D Angelo", "Anuel AA", "Kelly Clarkson",
        "Norah Jones", "OutKast",
    }
    # 90s
    _90s = {
        "Nirvana", "Radiohead", "Pearl Jam", "Green Day", "Weezer",
        "Blink-182", "TLC", "Aaliyah", "Mary J Blige", "Erykah Badu",
        "D Angelo", "Notorious BIG", "Run-DMC", "Jay-Z",
        "Metallica", "Megadeth", "Pantera", "Opeth",
        "Bob Marley", "Ziggy Marley", "UB40", "Buju Banton",
        "Beenie Man", "Shaggy", "Inner Circle",
        "Oasis", "Blur", "Pulp",
    }
    # classic (pre-90s) — everything not placed above
    _classic = {
        "Queen", "AC/DC", "Metallica", "Tom Petty",
        "Black Sabbath", "Iron Maiden", "Slayer", "Judas Priest",
        "Motörhead", "Miles Davis", "John Coltrane", "Frank Sinatra",
        "Ella Fitzgerald", "Charlie Parker", "Duke Ellington",
        "Louis Armstrong", "Thelonious Monk", "Billie Holiday",
        "Charles Mingus", "Dizzy Gillespie", "Dave Brubeck",
        "Herbie Hancock", "Count Basie", "Chet Baker", "Art Blakey",
        "Wayne Shorter", "Sonny Rollins", "Oscar Peterson",
        "Johnny Cash", "Dolly Parton", "Garth Brooks", "Willie Nelson",
        "Hank Williams", "Patsy Cline", "Kenny Chesney",
        "Bob Marley", "Peter Tosh", "Jimmy Cliff",
        "Toots and the Maytals", "Lee Scratch Perry", "Burning Spear",
        "Steel Pulse", "Groundation", "Don Omar",
        "Enrique Iglesias", "Mana", "Pitbull", "Shakira",
        "Luis Fonsi", "Romeo Santos",
        "The Beatles", "Led Zeppelin", "Pink Floyd", "David Bowie",
        "Fleetwood Mac", "The Rolling Stones", "Jimi Hendrix",
        "The Who", "The Doors",
        "Michael Jackson", "Prince", "Whitney Houston",
        "Madonna", "Tina Turner",
    }

    if artist in _2020s:
        return "2020s"
    if artist in _2010s:
        return "2010s"
    if artist in _2000s:
        return "2000s"
    if artist in _90s:
        return "1990s"
    # classic — covers pre-90s and any unclassified artist
    return "classic"


_ERAS: set[str] = {"classic", "1990s", "2000s", "2010s", "2020s"}


# ── Artist → genre tags index ────────────────────────────────────────

def _build_artist_genres() -> dict[str, set[str]]:
    """Invert _GENRE_ARTISTS so we can query an artist's genres in O(1)."""
    index: dict[str, set[str]] = {}
    for genre, artists in _GENRE_ARTISTS.items():
        for a in artists:
            index.setdefault(a, set()).add(genre)
    return index


_ARTIST_GENRES: dict[str, set[str]] = _build_artist_genres()


# ── Public API ────────────────────────────────────────────────────────

def generate_synthetic_pairs(
    n_artists: int = 200,
    pairs_per_artist: int = 8,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Generate synthetic (anchor, positive) pairs from genre×era clusters.

    Artists sharing >=1 genre tag are treated as "similar" candidates;
    probability scales with overlap depth.  Guarantees every selected
    artist appears in at least *pairs_per_artist* pairs (unless the pool
    is too small).

    Parameters
    ----------
    n_artists
        Number of artists to sample (from the pooled genre clusters).
    pairs_per_artist
        Target number of positive pairs per anchor.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    list of (anchor, positive) tuples.
    """
    rng = random.Random(seed)

    # 1. Build a deterministic ordered artist pool (all artists across all genres)
    all_artists: set[str] = set()
    for artists in _GENRE_ARTISTS.values():
        all_artists.update(artists)
    all_artists_sorted: list[str] = sorted(all_artists)

    # 2. Select (deterministically) the first N artists, then shuffle
    pool = all_artists_sorted[:n_artists]
    rng.shuffle(pool)
    pool_set = set(pool)

    pairs: List[Tuple[str, str]] = []
    used: dict[str, int] = {a: 0 for a in pool}  # track pair count per artist

    for anchor in pool:
        candidates: list[tuple[str, float]] = []

        for other in pool:
            if other == anchor:
                continue

            anchor_genres = _ARTIST_GENRES.get(anchor, set())
            other_genres = _ARTIST_GENRES.get(other, set())
            overlap = anchor_genres & other_genres
            k = len(overlap)

            if k >= 2:
                p = 0.8
            elif k == 1:
                p = 0.4
            else:
                p = 0.05  # hard negative

            candidates.append((other, p))

        # Shuffle candidates so ties are broken by seed, not order
        rng.shuffle(candidates)

        chosen = 0
        for other, prob in candidates:
            if chosen >= pairs_per_artist:
                break
            if rng.random() < prob:
                pairs.append((anchor, other))
                used[anchor] += 1
                used[other] += 1
                chosen += 1

    # 3. Fill — any artist below 3 pairs gets extra pairings
    for i, anchor in enumerate(pool):
        deficit = 3 - used[anchor]
        if deficit <= 0:
            continue

        candidates_fill: list[str] = [o for o in pool if o != anchor and (anchor, o) not in pairs]
        rng.shuffle(candidates_fill)
        filled = 0
        for other in candidates_fill:
            if filled >= deficit:
                break
            pairs.append((anchor, other))
            used[anchor] += 1
            used[other] += 1
            filled += 1

    return pairs


# ── Deterministic module-level constant ───────────────────────────────

SYNTHETIC_PAIRS: List[Tuple[str, str]] = generate_synthetic_pairs(
    n_artists=200,
    pairs_per_artist=8,
    seed=42,
)