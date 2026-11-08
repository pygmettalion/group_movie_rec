// Original 12x12 pixel-art illustrations, one per genre (drawn for this project; no real movie posters are used).
// Each sprite: rows of characters, '.' = empty, other letters = colours in `pal`.
window.SPRITES = {
  Action: { pal: { a: "#e8edf5", b: "#f2c14e", c: "#8a5a2b" }, rows: [
    "..........aa", ".........aaa", "........aaa.", ".......aaa..", "......aaa...", ".b...aaa....",
    ".bb.aaa.....", "..bbaa......", "..cbb.......", ".cc.bb......", "cc...b......", "c..........."] },
  Adventure: { pal: { a: "#5b7f6e", b: "#f2c14e", c: "#3a4f3f", d: "#eef2f0" }, rows: [
    ".........bb.", "........bbbb", "........bbbb", ".........bb.", "....d.......", "...ada......",
    "..aaaaa..d..", "..aaaaa.aaa.", ".aaaaaaaaaaa", "aaaaaaaaaaaa", "cccccccccccc", "cccccccccccc"] },
  Animation: { pal: { a: "#ffd166", b: "#3b2a1a" }, rows: [
    ".....aa.....", ".....aa.....", "....aaaa....", "aaaaaaaaaaaa", ".aaaaaaaaaa.", "..aabaabaa..",
    "...aaaaaa...", "...aabbaa...", "..aaa..aaa..", "..aa....aa..", ".aa......aa.", "............"] },
  "Children's": { pal: { a: "#ff6b6b", b: "#ffd6d6", c: "#e8edf5" }, rows: [
    "...aaaa.....", "..aaaaaa....", ".aabaaaaa...", ".aabaaaaa...", ".aaaaaaaa...", ".aaaaaaaa...",
    "..aaaaaa....", "...aaaa.....", "....cc......", ".....c......", "....c.......", ".....c......"] },
  Comedy: { pal: { a: "#ffd166", b: "#3b2a1a" }, rows: [
    "...aaaaaa...", ".aaaaaaaaaa.", ".aaaaaaaaaa.", "aabbaaaabbaa", "aabbaaaabbaa", "aaaaaaaaaaaa",
    "aaaaaaaaaaaa", "abaaaaaaaaba", "aabaaaaaabaa", ".aabbbbbbaa.", "..aaaaaaaa..", "....aaaa...."] },
  Drama: { pal: { a: "#c9b8ef", b: "#2a1f3d", c: "#7fd1ff" }, rows: [
    "...aaaaaa...", ".aaaaaaaaaa.", ".aaaaaaaaaa.", "aabbaaaabbaa", "aabbaaaabbaa", "aacaaaaaaaaa",
    "aacaaaaaaaaa", "aaaaaaaaaaaa", "aaaabbbbaaaa", ".aabaaaabaa.", "..aaaaaaaa..", "....aaaa...."] },
  "Sci-Fi": { pal: { a: "#e8edf5", b: "#4cc9f0", c: "#ff6b6b", d: "#ffb020" }, rows: [
    ".....aa.....", "....aaaa....", "....abba....", "....abba....", "....aaaa....", "....aaaa....",
    "...caaaac...", "..cc.aa.cc..", "..c..dd..c..", ".....dd.....", "....d..d....", "............"] },
  Horror: { pal: { a: "#eef1f6", b: "#1b1b22" }, rows: [
    "....aaaa....", "..aaaaaaaa..", ".aaaaaaaaaa.", ".aabbaabbaa.", ".aabbaabbaa.", ".aaaaaaaaaa.",
    ".aaaabbaaaa.", ".aaaabbaaaa.", ".aaaaaaaaaa.", ".aaaaaaaaaa.", ".aa.aa.aa.aa", ".a..a..a..a."] },
  // easter egg: a couple on a date (18x18) - he rests a finger on his chin, she smiles next to him
  Romance: { pal: { a: "#f1c7a0", h: "#1b1b22", w: "#f4f4f6", n: "#23315e", k: "#1e1a1a", c: "#7a4524", e: "#ff7aa2", g: "#2a2d36", f: "#ff3b5c", m: "#c0604a" }, rows: [
    ".......ff.ff......", ".......fffff......", "........fff.......", "..wwwww....ccccc..", ".wwwwwww..ccccccc.",
    ".kkkkkkk..ccccccc.", ".kaaaaak..caaaaac.", ".ahaaaha..cahaahc.", ".aaaaaaa..caaaaacc", ".amaaama..cmaaamcc",
    ".aammmaaa.caammacc", "..aaaaaaa.ccaaaacc", "..wnnnwaw..eeeeee.", ".wwwwwwww.eeeeeeee", "awwwwwwww.eeeeeeea",
    "anwwwwwn..aeeeeeea", "..gg..gg...gg.gg..", "..kk..kk...kk.kk.."] },
  Musical: { pal: { a: "#f2f4f8" }, rows: [
    "....aaaaaaaa", "....aaaaaaaa", "....a......a", "....a......a", "....a......a", "....a......a",
    "....a......a", "..aaa....aaa", ".aaaa...aaaa", ".aaaa...aaaa", "..aa.....aa.", "............"] },
  Mystery: { pal: { a: "#c7ccd6", b: "#7fb3ff", c: "#8a5a2b", d: "#e8f2ff" }, rows: [
    "..aaaa......", ".abbbba.....", "abbddbba....", "abdbbbba....", "abbbbbba....", "abbbbbba....",
    ".abbbba.....", "..aaaaaa....", ".......cc...", "........cc..", ".........cc.", "..........cc"] },
  Fantasy: { pal: { a: "#d9d2f2", b: "#3a2a5c", c: "#5b4a86" }, rows: [
    "a.a..a..a.a.", "aaa.aaa.aaa.", "aaa.aaa.aaa.", ".aaaaaaaaa..", ".aaaaaaaaa..", ".aabaaabaa..",
    ".aaaaaaaaa..", ".aaaabaaaa..", ".aaabbbaaa..", ".aaabbbaaa..", "cccccccccccc", "cccccccccccc"] },
  Western: { pal: { a: "#4f9d5d", b: "#ffb020", c: "#c98a4b" }, rows: [
    "........bb..", ".......bbbb.", ".....a..bb..", ".a...a......", ".a..aaa.....", ".aa.aaa..a..",
    "..aaaaa..a..", "....aaa.aa..", "....aaaaa...", "....aaa.....", "cccccccccccc", "cccccccccccc"] },
  War: { pal: { a: "#8a8f5a", b: "#4a4d30" }, rows: [
    "............", "....aaaa....", "..aaaaaaaa..", ".aaaaaaaaaa.", ".aaaaaaaaaa.", "aaaaaaaaaaaa",
    "bbbbbbbbbbbb", ".b........b.", ".b........b.", "..b......b..", "...bbbbbb...", "............"] },
  Thriller: { pal: { a: "#e8edf5", b: "#ff5a5f", c: "#111318" }, rows: [
    "............", "............", "....aaaa....", "..aa....aa..", ".a..bbbb..a.", "a..bbccbb..a",
    "a..bbccbb..a", ".a..bbbb..a.", "..aa....aa..", "....aaaa....", "............", "............"] },
  "Film-Noir": { pal: { a: "#2a2d36", b: "#ffd166", c: "#e8edf5" }, rows: [
    ".........cc.", "........c...", "........c...", ".........cc.", ".aa.........", ".aa..aaa....",
    ".aa..aba.aa.", "aaaa.aaa.aa.", "abaa.aba.ab.", "aaaaaaaaaaaa", "abaaabaaabaa", "aaaaaaaaaaaa"] },
  Crime: { pal: { a: "#2a2d36", b: "#c7ccd6", c: "#ff5a5f" }, rows: [
    "............", "...aaaaaa...", "...aaaaaa...", "...aaaaaa...", "..accccccaa.", ".aaaaaaaaaaa",
    "aaaaaaaaaaaa", "............", "...bbbbbb...", "..bb.bb.bb..", "...bbbbbb...", "....b..b...."] },
  Documentary: { pal: { a: "#c7ccd6", b: "#2a2d36", c: "#ff5a5f" }, rows: [
    "............", "..aa.aa.....", ".abba.bba...", ".abba.bba...", "..aa..aa....", "aaaaaaaaa...",
    "aaaaaaaaaab.", "aacaaaaaabb.", "aaaaaaaaaab.", "aaaaaaaaa...", ".a.....a....", "a.......a..."] },
};

// sprite (12x12 or 16x16) -> inline SVG (crisp pixels)
window.spriteSVG = function (genre, cls) {
  const s = window.SPRITES[genre] || window.SPRITES.Drama;
  let r = "";
  s.rows.forEach((row, y) => [...row].forEach((ch, x) => {
    if (ch !== ".") r += `<rect x="${x}" y="${y}" width="1.02" height="1.02" fill="${s.pal[ch]}"/>`;
  }));
  const n = s.rows[0].length;
  return `<svg class="${cls || ""}" viewBox="0 0 ${n} ${n}" shape-rendering="crispEdges" aria-hidden="true">${r}</svg>`;
};
