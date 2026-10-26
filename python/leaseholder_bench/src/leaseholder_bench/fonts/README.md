# Figure fonts

These files set the text of Leaseholder's generated figures ([`figure_style.py`](../figure_style.py)). They
are [STIX Two](https://github.com/stipub/stixfonts) 2.13 b171, from the static TTFs of its official release
(`fonts/static_ttf/` in the `v2.13b171` tag's archive), licensed under the SIL Open Font License 1.1
([OFL.txt](OFL.txt)); the license reserves the name "TM Math" only.

| File | Source |
|---|---|
| `STIXTwoText-Regular.ttf` | `STIXTwoText-Regular.ttf` |
| `STIXTwoText-Italic.ttf` | `STIXTwoText-Italic.ttf` |
| `STIXTwoMath-Regular.ttf` | `STIXTwoMath-Regular.ttf` |

Each was trimmed with fontTools' subsetter to the Unicode ranges the figures can use, its hinting dropped and its
`kern`, `liga`, `ccmp` and `locl` features kept: Basic Latin, Latin-1 and Latin Extended-A, Greek, general
punctuation, super- and subscripts, letterlike symbols and the minus sign in the text faces, and in the math face
those and arrows, mathematical operators, technical symbols and geometric shapes. Each exported SVG embeds a further
WOFF subset of only the glyphs it draws, and the demo video's frames are set from the same files.
