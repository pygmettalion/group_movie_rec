// Pixel-art poster scenes, generated per movie (original artwork for this project; no real posters are used).
// Each genre has a recipe (sky, landscape, props); a seeded random generator varies every movie.
(function () {
  const W = 40, H = 60;

  function rng(seed) {                       // small deterministic PRNG (mulberry32)
    let t = seed >>> 0;
    return () => { t += 0x6D2B79F5; let r = Math.imul(t ^ t >>> 15, 1 | t); r ^= r + Math.imul(r ^ r >>> 7, 61 | r); return ((r ^ r >>> 14) >>> 0) / 4294967296; };
  }

  function painter(ctx, R) {
    const p = {
      px(x, y, c) { if (x >= 0 && y >= 0 && x < W && y < H) { ctx.fillStyle = c; ctx.fillRect(x | 0, y | 0, 1, 1); } },
      rect(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(x | 0, y | 0, w | 0, h | 0); },
      sky(cols, y0 = 0, y1 = H) {             // stepped vertical gradient
        const n = cols.length, step = (y1 - y0) / n;
        cols.forEach((c, i) => p.rect(0, y0 + Math.floor(i * step), W, Math.ceil(step) + 1, c));
        for (let i = 1; i < n; i++) {         // dithered seam between bands
          const y = y0 + Math.floor(i * step);
          for (let x = (i % 2); x < W; x += 2) p.px(x, y - 1, cols[i]);
        }
      },
      disc(cx, cy, r, c) { for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (x * x + y * y <= r * r + r * .6) p.px(cx + x, cy + y, c); },
      stars(n, yMax, c = "#ffffff") { for (let i = 0; i < n; i++) p.px(R() * W, R() * yMax, R() < .3 ? c : "rgba(255,255,255,.55)"); },
      ridge(base, amp, rough, c, smooth = false) {   // mountain / hill silhouette from a random walk
        let h = base + (R() - .5) * amp;
        for (let x = 0; x < W; x++) {
          h += smooth ? Math.sin(x / 6 + R()) * .6 : (R() - .5) * rough;
          h = Math.max(base - amp, Math.min(base + amp, h));
          p.rect(x, h, 1, H - h, c);
        }
      },
      city(base, c, win, density = .25) {
        let x = 0;
        while (x < W) {
          const w = 3 + (R() * 5 | 0), h = 6 + (R() * 16 | 0);
          p.rect(x, base - h, w, h + H, c);
          for (let yy = base - h + 2; yy < base - 1; yy += 2) for (let xx = x + 1; xx < x + w - 1; xx += 2) if (R() < density) p.px(xx, yy, win);
          x += w + (R() < .3 ? 1 : 0);
        }
      },
      clouds(n, yMax, c) {
        for (let i = 0; i < n; i++) {
          const cx = R() * W, cy = 4 + R() * yMax, w = 6 + R() * 8;
          p.rect(cx - w / 2, cy, w, 2, c); p.rect(cx - w / 4, cy - 1, w / 2, 1, c);
        }
      },
      rain(n, c) { for (let i = 0; i < n; i++) { const x = R() * W, y = R() * H; p.px(x, y, c); p.px(x - 1, y + 1, c); } },
      tree(x, base, c) { p.rect(x, base - 6, 1, 6, c); p.rect(x - 2, base - 6, 5, 1, c); p.rect(x - 1, base - 8, 3, 2, c); p.rect(x, base - 10, 1, 2, c); },
      deadTree(x, base, c) {
        p.rect(x, base - 14, 1, 14, c);
        [[-1, -10], [-2, -11], [-3, -12], [1, -8], [2, -9], [3, -10], [4, -10], [-1, -5], [-2, -6], [1, -13], [2, -14]].forEach(([dx, dy]) => p.px(x + dx, base + dy, c));
      },
      person(x, base, c, hat = false) { p.rect(x, base - 5, 2, 4, c); p.px(x, base - 1, c); p.px(x + 1, base - 1, c); p.rect(x - 1, base - 5, 1, 3, c); p.rect(x + 2, base - 5, 1, 3, c); p.rect(x, base - 8, 2, 2, c); if (hat) { p.rect(x - 1, base - 9, 4, 1, c); p.rect(x, base - 10, 2, 1, c); } },
      umbrella(x, base, c, u) { p.person(x, base, c); p.rect(x - 2, base - 10, 6, 1, u); p.rect(x - 1, base - 11, 4, 1, u); p.rect(x + 1, base - 9, 1, 2, c); },
      lamp(x, base, c, light) {
        p.rect(x, base - 16, 1, 16, c); p.rect(x - 1, base - 17, 3, 1, c);
        for (let y = 0; y < 16; y++) p.rect(x - Math.floor(y / 2.5), base - 15 + y, 2 * Math.floor(y / 2.5) + 1, 1, light);
        p.px(x, base - 16, "#ffe9a8");
      },
      sprite(name, x, y, scale = 1) {
        const s = window.SPRITES && window.SPRITES[name]; if (!s) return;
        s.rows.forEach((row, yy) => [...row].forEach((ch, xx) => { if (ch !== ".") p.rect(x + xx * scale, y + yy * scale, scale, scale, s.pal[ch]); }));
      },
    };
    return p;
  }

  const RECIPES = {
    Action(p, R) {
      p.sky(["#1b0f2e", "#4a1d4f", "#a3354a", "#ef7a45", "#f7c05a"]);
      p.disc(10 + R() * 20 | 0, 34, 7, "#ffd27a");
      for (let i = 0; i < 26; i++) { const a = R() * 6.28, d = 3 + R() * 9; p.px(28 + Math.cos(a) * d, 30 + Math.sin(a) * d, R() < .5 ? "#ffcf5c" : "#ff6a3d"); }
      p.city(44, "#170d1c", "#ffcf5c", .18);
      p.rect(0, 52, W, 8, "#0d0710");
      p.person(18, 52, "#0d0710"); p.person(21, 52, "#0d0710");
    },
    Adventure(p, R) {
      p.sky(["#3a6ea5", "#6fa3d2", "#a9cfe8", "#f3d9a4"]);
      p.disc(30, 14, 4, "#fff1c2");
      p.clouds(3, 18, "#eef6fb");
      p.ridge(30, 6, 3, "#6d86a8"); p.ridge(38, 6, 2.5, "#4a6a7c"); p.ridge(46, 4, 1.5, "#2f4f3f");
      p.tree(8, 52, "#1f3a2a"); p.tree(31, 54, "#1f3a2a");
      p.person(19, 55, "#1a1a1a");
    },
    Animation(p, R) {
      p.sky(["#5aa9f0", "#86c5f5", "#bfe2fb"]);
      const cols = ["#ff5f5f", "#ffa94d", "#ffe066", "#69db7c", "#4dabf7", "#9775fa"];
      cols.forEach((c, i) => { const r = 22 - i; for (let a = 0; a <= 3.15; a += .02) p.px(20 + Math.cos(a) * r, 44 - Math.sin(a) * r, c); });
      p.clouds(4, 30, "#ffffff");
      p.ridge(48, 2, 0, "#7bd389", true); p.rect(0, 54, W, 6, "#5bb36b");
      p.sprite("Animation", 14, 32, 1);
    },
    "Children's"(p, R) {
      p.sky(["#7cc6fe", "#a8dbff", "#d6eeff"]);
      p.clouds(3, 20, "#ffffff");
      const cols = ["#ff6b6b", "#ffd43b", "#4dabf7", "#69db7c", "#da77f2"];
      for (let i = 0; i < 6; i++) { const x = 6 + R() * 28 | 0, y = 8 + R() * 22 | 0, c = cols[i % cols.length]; p.disc(x, y, 2, c); p.px(x - 1, y - 1, "#ffffff"); for (let k = 3; k < 9; k++) p.px(x + (k % 2), y + k, "#555"); }
      p.ridge(46, 3, 0, "#8ce99a", true); p.ridge(52, 2, 0, "#51cf66", true);
    },
    Comedy(p, R) {
      p.sky(["#ffb84d", "#ffd56b", "#ffe9a8", "#fff6d8"]);
      p.clouds(4, 30, "#ffffff");
      p.sprite("Comedy", 14, 20, 1);
      p.ridge(44, 3, 0, "#9bd36a", true); p.rect(0, 50, W, 10, "#7dbb52");
      for (let i = 0; i < 12; i++) p.px(R() * W, 51 + R() * 8, R() < .5 ? "#ff6b8a" : "#ffffff");
    },
    Crime(p, R) {
      p.sky(["#070a14", "#0f1626", "#1a2238"]);
      p.city(46, "#05070d", "#ffd56b", .12);
      p.rect(0, 46, W, 14, "#0b0e16");
      for (let x = 0; x < W; x += 4) p.rect(x, 52, 2, 1, "#3a4152");
      p.rect(24, 49, 6, 3, "#1b1f2b"); p.px(25, 48, "#ff3b3b"); p.px(28, 48, "#3b8bff");
      p.rain(40, "rgba(160,180,220,.45)");
      p.person(10, 54, "#000");
    },
    Documentary(p, R) {
      p.sky(["#203a4c", "#3f6f80", "#9cc5c1", "#f3e3b5"]);
      p.disc(28, 30, 3, "#fff3d6");
      p.rect(0, 36, W, 24, "#2f5d6b"); for (let i = 0; i < 30; i++) p.rect(R() * W, 37 + R() * 20, 3, 1, "#3f7484");
      p.ridge(36, 2, 1, "#1f3a3f"); p.rect(0, 36, W, 24, "rgba(47,93,107,.0)");
      p.rect(0, 50, W, 10, "#1b2f27"); p.tree(30, 50, "#0f1d17");
      for (let i = 0; i < 4; i++) { const x = 6 + i * 4, y = 12 + (i % 2) * 2; p.px(x, y, "#1b2f27"); p.px(x + 1, y - 1, "#1b2f27"); p.px(x - 1, y - 1, "#1b2f27"); }
    },
    Drama(p, R, m) {
      p.sky(["#1c2430", "#28333f", "#36475a", "#4b5f74"]);
      p.rain(90, "rgba(190,210,235,.5)");
      if (m && m.id === 2357) {                 // easter egg: the couple sharing an umbrella in the rain
        p.rect(0, 40, W, 20, "#141a22"); for (let x = 0; x < W; x += 3) p.px(x, 41, "#3e5163");
        p.lamp(33, 40, "#0b0f15", "rgba(255,214,120,.12)");
        p.rect(7, 20, 24, 1, "#c2536a"); p.rect(9, 19, 20, 1, "#c2536a"); p.rect(12, 18, 14, 1, "#d8667e"); p.rect(16, 17, 6, 1, "#d8667e");
        p.rect(19, 21, 1, 3, "#2a2d36");
        p.sprite("Romance", 10, 22, 1);
        for (let x = 10; x < 28; x += 2) p.px(x, 40, "rgba(255,143,171,.35)");   // reflection in the puddle
        return;
      }
      p.rect(0, 50, W, 10, "#141a22"); for (let x = 0; x < W; x += 3) p.px(x, 51, "#3e5163");
      p.lamp(30, 50, "#0b0f15", "rgba(255,214,120,.12)");
      p.umbrella(14, 50, "#07090d", "#c2536a");
    },
    Fantasy(p, R) {
      p.sky(["#120a2b", "#2a1650", "#4a2a7a", "#8a5ab0"]);
      p.stars(30, 30);
      p.disc(29, 12, 5, "#f6efc9"); p.disc(31, 11, 4, "#2a1650");
      p.ridge(46, 3, 0, "#1a1033", true);
      const cx = 14; [[0, 10, 4], [5, 14, 3], [9, 10, 4], [3, 8, 6]].forEach(([dx, h, w]) => p.rect(cx + dx, 46 - h - 6, w, h + 6, "#140c26"));
      for (let k = 0; k < 4; k++) p.px(cx + 1 + k * 3, 38, "#ffd56b");
      p.rect(0, 52, W, 8, "#0d0819");
    },
    "Film-Noir"(p, R) {
      p.sky(["#0b0b0d", "#18181c", "#26262c", "#38383f"]);
      p.disc(30, 10, 4, "#e9e9ea");
      p.city(44, "#0a0a0c", "#d9d9d9", .1);
      p.rect(0, 44, W, 16, "#121215");
      p.lamp(12, 54, "#000", "rgba(255,255,255,.08)");
      p.person(22, 54, "#000", true);
      for (let x = 0; x < W; x++) if (x % 2) p.px(x, 55, "#2a2a30");
    },
    Horror(p, R) {
      p.sky(["#120406", "#2a070c", "#4a0c14", "#5e1a1f"]);
      p.disc(22, 18, 9, "#f2e6c9"); p.disc(19, 16, 1, "#ddcfae"); p.disc(25, 21, 1, "#ddcfae");
      p.ridge(48, 2, 0, "#0b0203", true); p.rect(0, 52, W, 8, "#070102");
      p.deadTree(9, 50, "#070102");
      p.rect(28, 40, 8, 10, "#070102"); p.rect(27, 38, 10, 2, "#070102"); p.px(30, 43, "#ffcf5c");
      for (let i = 0; i < 3; i++) { const x = 6 + R() * 28, y = 5 + R() * 8; p.px(x, y, "#000"); p.px(x - 1, y - 1, "#000"); p.px(x + 1, y - 1, "#000"); }
    },
    Musical(p, R) {
      p.sky(["#120612", "#1d0a1d"]);
      for (let y = 0; y < 48; y++) p.rect(20 - (y / 3 | 0), y + 4, 2 * (y / 3 | 0) + 1, 1, "rgba(255,240,200,.10)");
      p.rect(13, 50, 15, 2, "rgba(255,240,200,.18)");
      for (let y = 0; y < 52; y++) { const w = 9 - Math.floor(Math.sin(y / 3) * 1.5); p.rect(0, y, w, 1, "#8e1c2e"); p.rect(W - w, y, w, 1, "#8e1c2e"); p.px(w - 1, y, "#b8304a"); p.px(W - w, y, "#b8304a"); }
      p.rect(0, 0, W, 4, "#6a1222");
      p.rect(0, 52, W, 8, "#4a2a12"); for (let x = 0; x < W; x += 5) p.rect(x, 52, 1, 8, "#3a200d");
      p.person(19, 52, "#0b0508");
      p.sprite("Musical", 24, 12, 1);
    },
    Mystery(p, R) {
      p.sky(["#0e1420", "#1a2434", "#2b3a4e"]);
      p.disc(8, 10, 3, "#d8e2f0");
      p.city(48, "#0b1018", "#9fb3cc", .06);
      p.rect(0, 48, W, 12, "#0c121b");
      for (let i = 0; i < 4; i++) p.rect(0, 38 + i * 4, W, 1, "rgba(200,215,235,.12)");
      p.lamp(28, 56, "#05080d", "rgba(255,230,160,.10)");
      p.person(16, 56, "#05080d", true);
    },
    Romance(p, R) {
      p.sky(["#3a1d4a", "#7a3060", "#d4607a", "#f4a28c", "#ffd4a3"]);
      p.disc(20, 30, 8, "#ffe3b0");
      for (let i = 0; i < 7; i++) { const x = 4 + R() * 32 | 0, y = 3 + R() * 14 | 0; p.px(x, y, "#ff8fab"); p.px(x + 2, y, "#ff8fab"); p.rect(x, y + 1, 3, 1, "#ff8fab"); p.px(x + 1, y + 2, "#ff8fab"); }
      p.ridge(40, 2, 0, "#3a1d3a", true); p.rect(0, 46, W, 14, "#2a1028");
      p.sprite("Romance", 11, 22, 1);              // the easter-egg couple
    },
    "Sci-Fi"(p, R) {
      p.sky(["#03040c", "#070b1e", "#0d1838"]);
      p.stars(45, 50);
      p.disc(10, 14, 6, "#4f7bd1"); p.disc(8, 12, 2, "#7aa3ec");
      for (let x = -10; x <= 10; x++) p.px(10 + x, 14 + Math.round(x * .25), "#c9d6ff");
      p.sprite("Sci-Fi", 24, 20, 1);
      p.ridge(50, 3, 2, "#2a1f3d"); p.rect(0, 55, W, 5, "#1b1429");
      for (let i = 0; i < 4; i++) p.disc(5 + R() * 30, 53 + R() * 4, 1, "#3a2b52");
    },
    Thriller(p, R) {
      p.sky(["#0a0a0a", "#141414", "#1f1f1f"]);
      p.disc(26, 20, 10, "#c8c3b4");
      p.rect(4, 26, 12, 34, "#050505"); p.rect(18, 34, 9, 26, "#080808");
      for (let y = 29; y < 58; y += 3) for (let x = 6; x < 15; x += 3) p.px(x, y, R() < .1 ? "#ffcf5c" : "#141414");
      p.person(9, 26, "#000");
      p.rect(0, 56, W, 4, "#000");
    },
    War(p, R) {
      p.sky(["#3b3a33", "#56544a", "#7a776a", "#a39f8c"]);
      for (let i = 0; i < 3; i++) { const x = 6 + i * 11, y = 8 + i * 3; p.rect(x, y, 5, 1, "#2a2924"); p.rect(x + 2, y - 1, 1, 3, "#2a2924"); }
      p.ridge(42, 4, 1.5, "#4a4836"); p.ridge(48, 3, 1, "#33321f");
      p.rect(10, 46, 14, 4, "#1d1c12"); p.rect(13, 43, 7, 3, "#1d1c12"); p.rect(20, 44, 8, 1, "#1d1c12");
      for (let x = 11; x < 24; x += 2) p.px(x, 50, "#000");
      for (let i = 0; i < 20; i++) p.px(R() * W, 28 + R() * 14, "rgba(60,58,50,.6)");
    },
    Western(p, R) {
      p.sky(["#e8743b", "#f39c4f", "#f7c06b", "#fbe0a0"]);
      p.disc(24, 30, 8, "#fff0c4");
      p.rect(0, 36, 12, 14, "#8a4a2a"); p.rect(2, 33, 8, 3, "#8a4a2a"); p.rect(30, 38, 10, 12, "#7a3f24"); p.rect(32, 35, 6, 3, "#7a3f24");
      p.rect(0, 48, W, 12, "#c8864b"); for (let i = 0; i < 25; i++) p.px(R() * W, 49 + R() * 10, "#b07440");
      p.sprite("Western", 16, 40, 1);
      p.person(30, 52, "#3a2010", true);
    },
  };

  window.drawScene = function (canvas, movie) {
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d"); ctx.imageSmoothingEnabled = false;
    const R = rng(movie.id * 2654435761);
    const gs = movie.genres.filter(g => RECIPES[g]);
    const g = gs.length ? gs[movie.id % gs.length] : "Drama";
    RECIPES[g](painter(ctx, R), R, movie);
  };
})();
