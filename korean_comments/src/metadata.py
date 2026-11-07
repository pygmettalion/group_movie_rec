"""2차 버전: 영화 메타데이터 붙이기 (제목, 개봉 연도, 감독, 평론가 점수).

입력 (data_extra/ 폴더)
  - ml-1m/movies.dat                   : 제목 + 개봉 연도 (MovieLens 1M 영화 전부)
  - ml-latest-small/links.csv          : MovieLens 번호 -> IMDb 번호
  - title_crew_tsv.gz                  : IMDb 번호 -> 감독 번호 (IMDb 공식 데이터)
  - Rotten Tomatoes Movies.csv         : 평론가 점수(tomatometer), 감독 이름, 개봉일  (제목+연도로 짝 맞춤)

시간 원칙 (미래 정보 금지)
  모델은 2000년 시점에서 추천하는 상황이다. 그 시점에 알 수 있던 정보만 쓴다.
  - 개봉 연도, 감독: 영화가 나올 때 정해진 정보 -> 사용
  - 평론가 점수: 대부분 개봉 무렵 쓰인 리뷰 -> 사용 (단, 나중에 추가된 리뷰도 일부 섞임: 보고서에 한계로 기록)
  - 관객 점수(audience_rating): 2000~2020년 일반 관객이 매긴 점수 = 시험 기간 이후 정보 -> 사용하지 않음
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
    """'Matrix, The (1999)' -> ['the matrix'], 다른 언어 제목 (괄호 안)도 후보로."""
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
    """제목(정규화) + 개봉 연도(±1년)로 짝 맞춤. 여러 개면 연도 차이가 가장 작은 것."""
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
            # 2단계(느슨한 짝): 같은 해 영화 중 (±1년이면 'The Fear(1995)'↔'Fear(1996)' 같은 오답이 생겨서 같은 해만) 단어가 많이 겹치는 것. 예: 'Star Trek: The Wrath of Khan'
            #   <-> 'Star Trek II: The Wrath of Khan'. 단어 겹침 비율 0.75 이상 + 가장 높은 것 하나만.
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
    """links.csv(IMDb 번호) -> IMDb title.crew 의 감독. 필요한 영화만 읽는다(파일이 425MB)."""
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
    # IMDb 에 없으면 RT 감독 이름을 쓴다 (이름 문자열 그대로 ID 처럼 사용)
    fill = meta.directors.isna() & meta.rt_directors.notna()
    meta.loc[fill, "directors"] = meta.loc[fill, "rt_directors"].map(lambda s: ["rt:" + x.strip() for x in str(s).split(",")])
    meta["director_source"] = np.where(meta.movie_id.isin(imdb_dir), "imdb", np.where(fill, "rt", "none"))
    return meta
