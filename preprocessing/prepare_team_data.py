"""Build the project CSVs from three source dataset ZIPs and a GroupLens links ZIP.

Requires Python 3.10+ and NumPy. Run python prepare_team_data.py --help
for the input and output options. Files are written to the folder given by --output.

Example (run from the directory containing the script and input files):
    python prepare_team_data.py --movielens ml-1m.zip --movietweetings MovieTweetings-master.zip --beliefs ml_belief_2024_data_release_2.zip --links ml-latest-small.zip --output prepared_data

Standard library imports
- argparse turns command-line options into named attributes.
- collections.Counter() counts entries and returns zero for a missing key.
  defaultdict(factory) calls int, list, or set to create a missing value.
- csv reads and writes delimited text and handles quoting and escaping.
- datetime.datetime represents a calendar date and time.
- io.TextIOWrapper reads or writes text through a binary ZIP member stream.
- itertools.combinations(iterable, r) yields each unordered group of r items once.
- pathlib.Path(path) wraps a path string for platform-aware file operations.
- random.Random(seed) creates a local generator with a repeatable sequence.
- tempfile.TemporaryDirectory(dir=parent) creates a working directory under
  parent and removes it when the block exits.
- zipfile.ZipFile(file, mode='r', compression=ZIP_STORED) reads ZIPs by default.
  For temporary archives we pass mode='w' and compression=ZIP_DEFLATED.

NumPy (np) stores rating rows and genre flags as arrays and sorts timestamps and row indices.

Python built-ins and methods used below
- int(value), float(value), and str(value) convert values. len(container)
  counts items, min(...) and max(...) select extremes, and sum(values) adds them.
- sorted(iterable, key=...) returns a new list. list.sort(key=...) sorts in place.
  In both cases key computes each item's sort value, and ties keep their input order.
- enumerate(iterable, start=0) pairs items with counters. Group IDs use start=1.
  range(start, stop) counts up to stop - 1. range(stop) starts at zero.
- map(function, iterable) applies a function to each item. zip(a, b) pairs items
  by position, and next(iterator) consumes the next item.
- tuple(iterable) and list(iterable) collect items. set(iterable) keeps distinct
  items. dict(pairs) builds a mapping and dict(existing_mapping) copies one.
- dict.items(), dict.keys(), and dict.values() expose pairs, keys, and values.
  & finds common keys between two key views. dict.get(key, default) returns
  default for a missing key (or None if omitted) without inserting an entry.
- dict.pop(key) returns and removes an entry. dict.clear() removes all entries.
  set.add(value) and list.append(value) add one item. list.extend(iterable) adds many.
- split(sep, maxsplit) separates fields. rsplit(sep, maxsplit) starts on the right.
  strip() trims surrounding whitespace. rstrip(chars) trims trailing characters
  in chars. separator.join(strings) places the separator between items.
- print(value, flush=True) displays progress immediately. ValueError(message)
  signals an invalid input or a failed data requirement.

Command-line interface
ArgumentParser(description=...) displays a summary in --help. We use the first
line of the module docstring, obtained with str.splitlines().
add_argument(name, type=Path, required=True, help=...) defines a required
option, converts its value to a Path, and provides help text.
parse_args() returns a Namespace with the values passed to prepare(args).
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
from io import TextIOWrapper
from itertools import combinations
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
)

RATING_TICKS = 20

def genre_matrix(genres):
    """Convert genres (movie ID to genre labels) into a binary movie-by-genre array.

    np.zeros(shape) fills a (row count, column count) array with floating-point zeros.
    max(genres) + 1 covers all movie IDs up to the largest ID, including unused IDs.
    """
    columns = {name: index for index, name in enumerate(GENRES)}
    features = np.zeros((max(genres) + 1, len(columns)))
    for movie, labels in genres.items():
        for label in labels:
            if label in columns:
                features[movie, columns[label]] = 1
    return features

def split_boundaries(ratings):
    """Find the validation and test cutoffs from the timestamp column of ratings.

    np.sort(array) returns an ascending sorted copy. The 70% and 85% positions define
    the cutoffs, and all ratings with the same timestamp are assigned to the same split.
    """
    timestamps = np.sort(ratings[:, 3])
    return timestamps[len(ratings) * 70 // 100], timestamps[len(ratings) * 85 // 100]

def load_ratings(path):
    """Read the MovieLens 1M ZIP at path into user, movie, rating-tick, and timestamp columns.

    ZipFile.open(name) streams the member as bytes. split(b"::") uses a byte delimiter.
    np.array(rows) converts the integer tuples to a 2D array and infers the integer dtype.
    """
    with ZipFile(path) as archive, archive.open("ml-1m/ratings.dat") as source:
        ratings = np.array([tuple(map(int, line.split(b"::"))) for line in source])
    ratings[:, 2] = (ratings[:, 2] - 1) * (RATING_TICKS // 4)
    return ratings

def load_data(path):
    """Read the ZIP at path and return train ratings, validation ratings, genre flags, and cutoffs.

    bytes.decode(encoding) returns text. MovieLens 1M metadata uses latin-1.
    """
    ratings = load_ratings(path)
    with ZipFile(path) as archive, archive.open("ml-1m/movies.dat") as source:
        genres = {}
        for line in source:
            movie, _, labels = line.decode("latin-1").strip().split("::")
            genres[int(movie)] = labels.split("|")
    movie_features = genre_matrix(genres)

    validation_start, test_start = split_boundaries(ratings)
    train = ratings[ratings[:, 3] < validation_start]
    validation = ratings[(ratings[:, 3] >= validation_start) & (ratings[:, 3] < test_start)]
    return train, validation, movie_features, (validation_start, test_start)

MINIMUM_HISTORY = 10

MINIMUM_COMMON = 5

SPLITS = ("validation", "test")

def load_movie_masks(archive_path, test_only=False):
    """Build per-user masks of eligible movies from the MovieLens 1M ZIP at archive_path.

    test_only=True fills only the test masks. Movie ID n uses bit 1 << n.
    Bitwise OR (|) adds movies, and bitwise AND (&) finds shared movies.
    """
    with ZipFile(archive_path) as archive, archive.open("ml-1m/ratings.dat") as source:
        rows = [tuple(map(int, line.split(b"::"))) for line in source]
    timestamps = sorted(row[3] for row in rows)
    validation_start = timestamps[len(rows) * 70 // 100]
    test_start = timestamps[len(rows) * 85 // 100]
    history = Counter()
    known_movies = set()
    for user, movie, _, timestamp in rows:
        if timestamp < validation_start:
            history[user] += 1
            known_movies.add(movie)

    masks = {split: defaultdict(int) for split in SPLITS}
    split_counts = Counter()
    for user, movie, _, timestamp in rows:
        if timestamp < validation_start:
            split_counts["train"] += 1
            continue
        split = "validation" if timestamp < test_start else "test"
        split_counts[split] += 1
        if (not test_only or split == "test") and history[user] >= MINIMUM_HISTORY and movie in known_movies:
            masks[split][user] |= 1 << movie
    return masks, split_counts, (validation_start, test_start)

def find_groups(masks, test_only=False, minimum_common=MINIMUM_COMMON):
    """Yield four-person groups with at least minimum_common shared movies in each split.

    masks maps split names to user movie masks. test_only=True checks only the test split.
    int.bit_count() counts set bits. combinations(candidates, 2) chooses two distinct
    candidates to complete each group after the first two members have been chosen.

    Each remaining member must preserve the pair's movie overlap on its own.
    """
    test = masks["test"]
    validation = test if test_only else masks["validation"]
    users = sorted(
        user for user in validation.keys() & test.keys()
        if min(validation[user].bit_count(), test[user].bit_count()) >= minimum_common
    )
    print(f"Eligible users: {len(users)}")
    for first_index, first in enumerate(users):
        first_validation, first_test = validation[first], test[first]
        for second_index in range(first_index + 1, len(users)):
            second = users[second_index]
            pair_validation = first_validation & validation[second]
            pair_test = first_test & test[second]
            if min(pair_validation.bit_count(), pair_test.bit_count()) < minimum_common:
                continue

            candidates = []
            for user in users[second_index + 1:]:
                shared_validation = pair_validation & validation[user]
                shared_test = pair_test & test[user]
                if min(shared_validation.bit_count(), shared_test.bit_count()) >= minimum_common:
                    candidates.append((user, shared_validation, shared_test))

            for (third, v3, t3), (fourth, v4, t4) in combinations(candidates, 2):
                group_validation, group_test = v3 & v4, t3 & t4
                if min(group_validation.bit_count(), group_test.bit_count()) >= minimum_common:
                    yield (first, second, third, fourth), group_validation, group_test

def movie_ids(mask):
    """Convert the integer movie mask to a pipe-delimited list of movie IDs.

    int.bit_length() counts the bits needed to store the value. For a single set bit,
    subtract one to get its zero-based position, which is the MovieLens ID here.
    """
    result = []
    while mask:
        bit = mask & -mask
        result.append(str(bit.bit_length() - 1))
        mask ^= bit
    return "|".join(result)

def source_crosswalk(path):
    """Read links.csv from the GroupLens ZIP at path and return one-to-one IMDb-to-MovieLens links.

    ZipFile.namelist() lists member paths. str.endswith(suffix) selects links.csv.
    """
    movielens_to_imdb, imdb_to_movielens = defaultdict(set), defaultdict(set)
    with ZipFile(path) as archive:
        name, = (name for name in archive.namelist() if name.endswith("/links.csv"))
        with TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if not row["imdbId"]:
                    continue
                movielens = int(row["movieId"])
                imdb = int(row["imdbId"])
                movielens_to_imdb[movielens].add(imdb)
                imdb_to_movielens[imdb].add(movielens)
    return {imdb: movie for imdb, candidates in imdb_to_movielens.items() if len(candidates) == 1
            for movie in candidates if len(movielens_to_imdb[movie]) == 1}

def read_source(path):
    """Read per-user ratings and movie metadata from the MovieTweetings ZIP at path.

    rsplit('/', 1) splits at the final slash to locate movies.dat in the same directory.
    """
    histories, metadata = defaultdict(list), {}
    with ZipFile(path) as archive:
        rating_path, = (name for name in archive.namelist() if name.endswith("/latest/ratings.dat"))
        with archive.open(rating_path) as source:
            for line in source:
                user, movie, rating, timestamp = map(int, line.split(b"::"))
                histories[user].append((timestamp, movie, rating))
        with archive.open(rating_path.rsplit("/", 1)[0] + "/movies.dat") as source:
            for line in source:
                movie, rest = line.rstrip(b"\r\n").split(b"::", 1)
                title, genres = rest.rsplit(b"::", 1)
                metadata[int(movie)] = title.decode("utf-8"), genres.decode("ascii")
    return histories, metadata

TEST_FILMS_PER_GROUP = 5

MAX_TEST_FILMS_PER_GROUP = 10

MAX_GROUPS_PER_FILM = 2

MINIMUM_MAPPED_HISTORY = 10

RATINGS_COLUMNS = ("dataset", "user_id", "movie_id", "rating_raw", "rating_norm", "timestamp", "genres", "split")

GROUP_COLUMNS = ("group_id", *(f"user_id_{index}" for index in range(1, 5)), "validation_common_movies",
                 "test_common_movies", "validation_movie_ids", "test_movie_ids")

def set_bits(mask):
    """Yield each set bit in mask. mask & -mask isolates the lowest set bit, and XOR removes it."""
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit

def find_partners(base, remaining):
    """Find three users in remaining whose movie masks overlap enough with base."""
    overlaps = [(user, base & films) for user, films in remaining.items()]
    partners = [(user, films) for user, films in overlaps if films.bit_count() >= TEST_FILMS_PER_GROUP]
    partners.sort(key=lambda partner: -partner[1].bit_count())
    for index, (second, pair) in enumerate(partners):
        thirds = [(user, pair & films) for user, films in partners[index + 1:]]
        thirds = [(user, films) for user, films in thirds if films.bit_count() >= TEST_FILMS_PER_GROUP]
        for (third, left), (fourth, right) in combinations(thirds, 2):
            shared = left & right
            if shared.bit_count() >= TEST_FILMS_PER_GROUP:
                return (second, third, fourth), shared
    return None

def choose_groups(histories, mapping):
    """Select external groups from per-user histories and an IMDb-to-MovieLens mapping.

    Random(42) fixes the seed so ties in the following sort use a reproducible order.
    Random.shuffle(sequence) changes the candidate list in place.
    """
    movie_ids = sorted(mapping)
    movie_bits = {movie: 1 << index for index, movie in enumerate(movie_ids)}
    eligible, popularity = {}, defaultdict(int)
    for user, history in histories.items():
        mapped = sorted((time, movie) for time, movie, _ in history if movie in mapping)
        if len(mapped) < MINIMUM_MAPPED_HISTORY + TEST_FILMS_PER_GROUP:
            continue
        cutoff = mapped[MINIMUM_MAPPED_HISTORY - 1][0]
        movies = {movie for time, movie in mapped if time > cutoff}
        if len(movies) >= TEST_FILMS_PER_GROUP:
            eligible[user] = sum(movie_bits[movie] for movie in movies)
            for movie in movies:
                popularity[movie_bits[movie]] += 1

    pivots = list(eligible)
    Random(42).shuffle(pivots)
    pivots.sort(key=lambda user: eligible[user].bit_count())
    remaining = dict(eligible)
    full = 0
    usage = defaultdict(int)
    groups = []

    def select_films(mask, limit):
        """Take up to limit movies from mask, favoring lower group usage and then lower popularity."""
        nonlocal full
        films = sorted(set_bits(mask & ~full), key=lambda bit: (usage[bit], popularity[bit]))[:limit]
        for bit in films:
            usage[bit] += 1
            if usage[bit] == MAX_GROUPS_PER_FILM:
                full |= bit
        return films

    for first in pivots:
        if first not in remaining:
            continue
        base = remaining.pop(first) & ~full
        if base.bit_count() < TEST_FILMS_PER_GROUP:
            continue
        result = find_partners(base, remaining)
        if result is None:
            continue
        partners, shared = result
        for user in partners:
            del remaining[user]
        groups.append(((first, *partners), select_films(shared, TEST_FILMS_PER_GROUP)))

    if not groups:
        raise ValueError("No groups met the external coverage requirement.")
    for users, selected in groups:
        common = eligible[users[0]] & eligible[users[1]] & eligible[users[2]] & eligible[users[3]]
        selected.extend(select_films(common & ~sum(selected), MAX_TEST_FILMS_PER_GROUP - len(selected)))
    return [(tuple(sorted(users)), tuple(sorted(movie_ids[bit.bit_length() - 1] for bit in films)))
            for users, films in groups]

def write_csv(archive, name, columns, rows):
    """Write CSV member name to archive, using columns as the header and rows as records.

    ZipFile.open(name, 'w') opens a writable member stream. TextIOWrapper writes UTF-8
    with a BOM via encoding='utf-8-sig' and lets the CSV writer handle line endings
    via newline=''. csv.writer(dest) uses commas by default.
    writerow(columns) writes the header. writerows(rows) consumes the records one at a time.
    """
    with TextIOWrapper(archive.open(name, "w"), encoding="utf-8-sig", newline="") as dest:
        writer = csv.writer(dest)
        writer.writerow(columns)
        writer.writerows(rows)

def imdb(movie):
    """Format integer movie as an IMDb ID. 07d zero-pads to at least seven digits."""
    return f"tt{movie:07d}"

def build_bundle(path, raw_archive, crosswalk_archive, train, movielens_genres):
    """Write the MovieTweetings test groups and past ratings to the intermediate ZIP at path.

    raw_archive is the MovieTweetings ZIP and crosswalk_archive is the GroupLens links ZIP.
    train contains MovieLens training ratings. movielens_genres contains movie-by-genre flags.

    Path.mkdir(parents=True, exist_ok=True) creates missing parent directories and accepts
    an existing directory. Path.with_suffix(suffix) returns a path with the suffix replaced.

    ZipFile.testzip() checks member headers and CRCs. None means no errors.

    Path.replace(target) moves the completed ZIP and replaces an existing file.
    """
    histories, metadata = read_source(raw_archive)
    known = set(train[:, 1])
    mapping = {movie: internal for movie, internal in source_crosswalk(crosswalk_archive).items()
               if internal in known and movie in metadata}
    groups = choose_groups(histories, mapping)
    test_rows, history_rows, group_rows, used_links = [], [], [], set()

    def rating_row(user, movie, score, time, split):
        """Build a CSV row from IDs user and movie, raw score, timestamp time, and split."""
        if movie in mapping:
            used_links.add(movie)
            labels = [genre for genre, flag in zip(GENRES, movielens_genres[mapping[movie]]) if flag]
        else:
            raw_genres = metadata[movie][1].split("|")
            labels = [genre for genre in GENRES if genre in raw_genres]
        return ("movietweetings", user, imdb(movie), score, f"{score / 10:.2f}", time, "|".join(labels), split)

    for group_id, (users, movies) in enumerate(groups, 1):
        group_rows.append((group_id, *users, "", len(movies), "", "|".join(map(imdb, movies))))
        for user in users:
            by_movie = {movie: (time, score) for time, movie, score in histories[user]}
            cutoff = min(by_movie[movie][0] for movie in movies)
            earlier = [(time, movie, score) for time, movie, score in sorted(histories[user]) if time < cutoff]
            assert sum(movie in mapping for _, movie, _ in earlier) >= MINIMUM_MAPPED_HISTORY
            history_rows.extend(rating_row(user, movie, score, time, "history") for time, movie, score in earlier)
            test_rows.extend(rating_row(user, movie, by_movie[movie][1], by_movie[movie][0], "test") for movie in movies)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".zip.tmp")
    with ZipFile(temp, "w", ZIP_DEFLATED) as archive:
        write_csv(archive, "movietweetings_test.csv", RATINGS_COLUMNS, test_rows)
        write_csv(archive, "movietweetings_test_groups.csv", GROUP_COLUMNS, group_rows)
        write_csv(archive, "movietweetings_past_history.csv", RATINGS_COLUMNS, history_rows)
        write_csv(archive, "movie_id_crosswalk.csv", ("movie_id", "movielens_movie_id"),
                  ((imdb(movie), mapping[movie]) for movie in sorted(used_links)))
    with ZipFile(temp) as archive:
        assert archive.testzip() is None
    temp.replace(path)
    unique_films = len({movie for _, movies in groups for movie in movies})
    print(f"External CSV: {len(groups)} groups, {len(groups) * 4} users, {unique_films} films, {len(test_rows)} test ratings, "
          f"{len(history_rows)} prior ratings.", flush=True)

EPOCH = datetime(1970, 1, 1)

BELIEF_TICKS = 9

def timestamp(value):
    """Convert Beliefs ISO timestamp value to integer seconds since 1970-01-01.

    datetime(1970, 1, 1) represents the epoch at midnight. datetime.fromisoformat(value)
    parses the timestamp, and timedelta.total_seconds() returns the full elapsed seconds.
    """
    return int((datetime.fromisoformat(value) - EPOCH).total_seconds())

def rating_ticks(value):
    """Convert Beliefs rating value to 0–9 half-star steps. Return None for invalid ratings.

    float.is_integer() checks for a whole number to exclude values off the half-star scale.
    """
    if value in ("", "NA", "-1"):
        return None
    steps = float(value) * 2 - 1
    return int(steps) if steps.is_integer() and 0 <= steps <= BELIEF_TICKS else None

def read_csv(archive, name):
    """Yield dictionaries from Beliefs CSV name (without its extension) in archive.

    csv.DictReader(text_stream) uses the first row as field names and returns a dictionary
    per row. When reading, encoding='utf-8-sig' removes an initial BOM if present.
    """
    with archive.open(f"data_release/{name}.csv") as source:
        yield from csv.DictReader(TextIOWrapper(source, encoding="utf-8-sig", newline=""))


def archive_rows(archive, name):
    """Yield data rows from CSV member name in archive, skipping the header.

    csv.reader(source) returns lists of strings. next(reader) skips the header.
    """
    with TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        next(reader)
        yield from reader


def write_movielens(archive, raw):
    """Write ratings and validation/test groups from the raw MovieLens ZIP into archive.

    np.lexsort(keys) sorts row positions with the last key as the primary key.
    Here the sort order is user, then timestamp, then movie ID.
    """
    ratings = load_ratings(raw)
    train, _, features, (validation_start, test_start) = load_data(raw)
    genres = ["|".join(genre for genre, flag in zip(GENRES, flags) if flag) for flags in features]
    order = np.lexsort((ratings[:, 1], ratings[:, 3], ratings[:, 0]))

    def rating_rows():
        """Yield MovieLens 1M ratings in sorted order, with each split assigned by its timestamp."""
        for user, movie, ticks, time in ratings[order]:
            split = "train" if time < validation_start else "validation" if time < test_start else "test"
            yield "movielens_1m", user, movie, 1 + ticks // (RATING_TICKS // 4), \
                f"{ticks / RATING_TICKS:.2f}", time, genres[movie], split

    masks, _, _ = load_movie_masks(raw)
    groups = ((group, *users, valid.bit_count(), test.bit_count(), movie_ids(valid), movie_ids(test))
              for group, (users, valid, test) in enumerate(find_groups(masks), 1))
    write_csv(archive, "01_MovieLens_1M/ratings.csv", RATINGS_COLUMNS, rating_rows())
    write_csv(archive, "01_MovieLens_1M/eval_groups.csv", GROUP_COLUMNS, groups)
    return train, features


def write_movietweetings(archive, raw, links, train, features, intermediate):
    """Build and write MovieTweetings CSVs with mapped movie IDs in the output archive.

    raw is the source ZIP, links is the GroupLens mapping ZIP, and intermediate is the
    temporary ZIP path. train and features are the MovieLens training and movie genre arrays.

    ZipFile.read(name) returns a member's bytes. ZipFile.writestr(name, data)
    writes those bytes to the named output member.
    """
    build_bundle(intermediate, raw, links, train, features)
    folder = "02_MovieTweetings/"
    with ZipFile(intermediate) as external:
        mapping = dict(archive_rows(external, "movie_id_crosswalk.csv"))

        def rating_rows():
            """Yield history rows followed by test rows, replacing IMDb IDs when a mapping exists."""
            for name in ("movietweetings_past_history.csv", "movietweetings_test.csv"):
                for row in archive_rows(external, name):
                    row[2] = mapping.get(row[2], row[2])
                    yield row

        def group_rows():
            """Apply the same movie ID mapping to the test movie list in each group row."""
            for row in archive_rows(external, "movietweetings_test_groups.csv"):
                row[-1] = "|".join(mapping[movie] for movie in row[-1].split("|"))
                yield row

        write_csv(archive, folder + "movietweetings_ratings.csv", RATINGS_COLUMNS, rating_rows())
        write_csv(archive, folder + "movietweetings_test_groups.csv", GROUP_COLUMNS, group_rows())
        archive.writestr(folder + "movie_id_crosswalk.csv", external.read("movie_id_crosswalk.csv"))
    print(f"MovieTweetings applied ID links: {len(mapping):,}", flush=True)


def write_beliefs(archive, raw):
    """Read Beliefs ZIP raw and write expected ratings and actual history to separate CSVs in archive.

    Collect eligible actual ratings before checking the final per-user cutoff.

    Write past_history.csv first so first contains each user's earliest retained rating time.
    """
    folder = "03_Beliefs/"
    dataset = "movielens_beliefs_2024"
    responses = {}
    counts = Counter()
    with ZipFile(raw) as source:
        for row in read_csv(source, "belief_data"):
            if row["isSeen"] != "0":
                continue
            ticks = rating_ticks(row["userPredictRating"])
            if ticks is None:
                continue
            counts["valid_unseen"] += 1
            key = int(row["userId"]), int(row["movieId"])
            time = timestamp(row["tstamp"])
            if key not in responses or time < responses[key][0]:
                responses[key] = time, ticks

        cutoff = min(time for time, _ in responses.values())
        latest = {}
        for (user, _), (time, _) in responses.items():
            latest[user] = max(time, latest.get(user, time))

        conflicts = set()
        history = []
        for row in read_csv(source, "user_rating_history"):
            ticks = rating_ticks(row["rating"])
            if ticks is None:
                continue
            user, movie = int(row["userId"]), int(row["movieId"])
            if user not in latest:
                continue
            time = timestamp(row["tstamp"])
            target = responses.get((user, movie))
            if target is not None and time <= target[0]:
                conflicts.add((user, movie))
            if time < latest[user]:
                history.append((user, movie, time, ticks))

        genres = {}
        for row in read_csv(source, "movies"):
            labels = {"Children's" if label == "Children" else label for label in row["genres"].split("|")}
            genres[int(row["movieId"])] = "|".join(genre for genre in GENRES if genre in labels)
        license_text = source.read("data_release/README.txt")

    counts["repeated_unseen_responses"] = counts["valid_unseen"] - len(responses)
    counts["unseen_with_actual_rating"] = len(conflicts)
    targets = sorted((time, user, movie, ticks) for (user, movie), (time, ticks) in responses.items()
                     if (user, movie) not in conflicts)
    latest.clear()
    for time, user, _, _ in targets:
        latest[user] = time

    first = {}
    users_without_prior_history = set()

    def history_rows():
        """Yield unique actual rating rows strictly before the user's last retained response.

        round(value, 8) preserves the eight-decimal format in past_history.csv.
        unseen_test.csv keeps the full float representation.

        Deduplicate the entire (user, movie, time, ticks) tuple.
        Ratings at different times or with different scores remain separate.
        """
        seen = set()
        for early in (True, False):
            for row in history:
                user, movie, time, ticks = row
                if (time < cutoff) == early and user in latest and time < latest[user]:
                    if row in seen:
                        counts["duplicate_history_rows_removed"] += 1
                        continue
                    seen.add(row)
                    first[user] = min(first.get(user, time), time)
                    counts["history_rows"] += 1
                    yield (dataset, user, movie, (ticks + 1) / 2, round(ticks / BELIEF_TICKS, 8),
                           time, genres.get(movie, ""), "history")

    def test_rows():
        """Yield unrounded expected scores for responses with strictly earlier rating history.

        CSV uses Python's full round-trip float representation, without a decimal-place limit.
        """
        for time, user, movie, ticks in targets:
            if user not in first or first[user] >= time:
                counts["no_prior_history_rows_removed"] += 1
                users_without_prior_history.add(user)
                continue
            counts["test_rows"] += 1
            yield dataset, user, movie, (ticks + 1) / 2, ticks / BELIEF_TICKS, time, genres.get(movie, ""), "test"

    write_csv(archive, folder + "past_history.csv", RATINGS_COLUMNS, history_rows())
    write_csv(archive, folder + "unseen_test.csv", RATINGS_COLUMNS, test_rows())
    archive.writestr(folder + "SOURCE_LICENSE_README.txt", license_text)
    counts["users_with_removed_responses"] = len(users_without_prior_history)
    print(f"Beliefs audit: {dict(counts)}", flush=True)


def prepare(args):
    """Run all preprocessing steps and write the result files to args.output.

    Path.resolve() returns an absolute path. Path.exists() and Path.is_dir()
    check that an existing destination is a directory. NotADirectoryError(message)
    reports a conflicting file at that path.

    TemporaryDirectory(dir=output.parent) creates a working directory next to the
    output folder and removes it when processing finishes.

    ZipFile.extractall(path=output) writes the generated members to the output folder,
    creating their subdirectories as needed. The temporary ZIP is then cleaned up.
    """
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output must be a directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as work:
        work = Path(work)
        result = work / "prepared.zip"
        with ZipFile(result, "w", ZIP_DEFLATED) as archive:
            train, features = write_movielens(archive, args.movielens)
            write_movietweetings(archive, args.movietweetings, args.links, train, features, work / "external.zip")
            write_beliefs(archive, args.beliefs)
        with ZipFile(result) as archive:
            archive.extractall(path=output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--movielens", type=Path, required=True, help="ml-1m.zip")
    parser.add_argument("--movietweetings", type=Path, required=True, help="MovieTweetings-master.zip")
    parser.add_argument("--beliefs", type=Path, required=True, help="ml_belief_2024_data_release_2.zip")
    parser.add_argument("--links", type=Path, required=True, help="GroupLens ml-latest-small.zip containing links.csv")
    parser.add_argument("--output", type=Path, required=True, help="Destination folder for the generated CSV files")
    prepare(parser.parse_args())
