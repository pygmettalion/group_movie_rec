"""Version 2: attach movie metadata (title, release year, director, critic score).

Inputs (data_extra/ folder)
  - ml-1m/movies.dat                   : title + release year (all MovieLens 1M movies)
  - ml-latest-small/links.csv          : MovieLens id -> IMDb id
  - title_crew_tsv.gz                  : IMDb id -> director id (official IMDb data)
  - Rotten Tomatoes Movies.csv         : critic score (tomatometer), director name, release date  (matched by title + year)

Time principle (no future information)
  The model recommends as of the year 2000. Only information available at that time is used.
  - release year, director: fixed when the movie came out -> used
  - critic score: mostly reviews written around release -> used (some later reviews are mixed in: noted as a limitation in the report)
  - audience score (audience_rating): rated by general audiences in 2000~2020 = information after the test period -> not used
"""
import gzip
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

EXTRA = Path(__file__).resolve().parents[1] / "data_extra"


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


ARTICLES = r"(The|A|An|Les|La|Le|L'|Il|Das|Der|Die|El|Los|Las|Lo|I)"


def _title_variants(title):
    """'Matrix, The (1999)' -> ['the matrix']; titles in other languages (in parentheses) are also candidates."""
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", title)
    base, year = m.group(1), int(m.group(2))
    names = [re.sub(r"\s*\([^)]*\)", "", base).strip()] + re.findall(r"\(([^)]*)\)", base)
    out = []
    for x in names:
        x = re.sub(r"^a\.k\.a\.\s*", "", x.strip(), flags=re.I)
        mm = re.match(rf"^(.*), {ARTICLES}$", x)
        if mm:
            x = f"{mm.group(2)} {mm.group(1)}"
        out.append(_norm(x))
    return out, year


def load_movies():
    mv = pd.read_csv(EXTRA / "ml-1m" / "movies.dat", sep="::", engine="python",
                     names=["movie_id", "title", "genres"], encoding="latin1")
    mv["year"] = mv.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(int)
    return mv


def match_rotten_tomatoes(mv):
    """Match by normalized title + release year (+-1 year). If several match, take the smallest year difference."""
    rt = pd.read_csv(EXTRA / "Rotten Tomatoes Movies.csv")
    rt["y"] = pd.to_datetime(rt.in_theaters_date, errors="coerce").dt.year
    rt["k"] = rt.movie_title.map(_norm)
    index, by_year = {}, {}
    for i, (k, y) in enumerate(zip(rt.k, rt.y)):
        index.setdefault(k, []).append((y, i))
        if pd.notna(y):
            by_year.setdefault(int(y), []).append(i)
    rt_words = [set(k.split()) - {"the", "a", "an", "of"} for k in rt.k]
    rows = []
    for mid, title in zip(mv.movie_id, mv.title):
        names, year = _title_variants(title)
        best = None
        for n in names:
            for y, i in index.get(n, []):
                if pd.notna(y) and abs(y - year) <= 1 and (best is None or abs(y - year) < best[0]):
                    best = (abs(y - year), i)
            if best:
                break
        if best is None:
            # step 2 (loose match): among movies of the same year (+-1 year caused wrong matches like 'The Fear(1995)'<->'Fear(1996)', so same year only), the one with the most word overlap. e.g. 'Star Trek: The Wrath of Khan'
            #   <-> 'Star Trek II: The Wrath of Khan'. Word-overlap ratio >= 0.75 and only the single highest.
            words = [set(n.split()) - {"the", "a", "an", "of"} for n in names]
            cands = []
            for i in by_year.get(year, []):
                w = rt_words[i]
                for ws in words:
                    if ws and w:
                        jac = len(ws & w) / len(ws | w)
                        if jac >= 0.75:
                            cands.append((jac, -abs(rt.y.iat[i] - year), i))
            if cands:
                cands.sort(reverse=True)
                best = (None, cands[0][2])
        if best:
            row = rt.iloc[best[1]]
            rows.append(dict(movie_id=mid, rt_title=row.movie_title, rt_year=row.y,
                             critic=row.tomatometer_rating / 100.0, critic_count=row.tomatometer_count,
                             rt_directors=row.directors))
    return pd.DataFrame(rows)


def load_directors(mv):
    """links.csv (IMDb id) -> directors from IMDb title.crew. Only the needed movies are read (the file is 425MB)."""
    links = pd.read_csv(EXTRA / "ml-latest-small" / "links.csv", dtype={"imdbId": str})
    links = links[links.movieId.isin(mv.movie_id)]
    want = {"tt" + i.zfill(7): m for m, i in zip(links.movieId, links.imdbId)}
    want.update({"tt" + i.zfill(8): m for m, i in zip(links.movieId, links.imdbId)})
    out = {}
    with gzip.open(EXTRA / "title_crew_tsv.gz", "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            t, d, _ = line.rstrip("\n").split("\t")
            if t in want and d != "\\N":
                out[want[t]] = d.split(",")
    return out


def build_metadata():
    mv = load_movies()
    rt = match_rotten_tomatoes(mv)
    imdb_dir = load_directors(mv)
    meta = mv.merge(rt, on="movie_id", how="left")
    meta["directors"] = meta.movie_id.map(lambda m: imdb_dir.get(m))
    # if missing in IMDb, use the RT director name (the name string itself is used like an ID)
    fill = meta.directors.isna() & meta.rt_directors.notna()
    meta.loc[fill, "directors"] = meta.loc[fill, "rt_directors"].map(lambda s: ["rt:" + x.strip() for x in str(s).split(",")])
    meta["director_source"] = np.where(meta.movie_id.isin(imdb_dir), "imdb", np.where(fill, "rt", "none"))
    return meta
