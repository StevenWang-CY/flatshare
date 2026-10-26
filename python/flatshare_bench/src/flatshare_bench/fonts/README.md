# Fonts

These files set the text of Flatshare's generated figures ([`figure_style.py`](../figure_style.py)) and of its logo and
social preview ([`brand.py`](../brand.py), which outlines its glyphs to paths). They are [STIX
Two](https://github.com/stipub/stixfonts) 2.13 b171, from the static TTFs of its official release (`fonts/static_ttf/`
in the `v2.13b171` tag's archive), licensed under the SIL Open Font License 1.1 ([OFL.txt](OFL.txt)); the license
reserves the name "TM Math" only.

| File | Source |
|---|---|
| `STIXTwoText-Regular.ttf` | `STIXTwoText-Regular.ttf` |
| `STIXTwoText-Italic.ttf` | `STIXTwoText-Italic.ttf` |
| `STIXTwoText-SemiBold.ttf` | `STIXTwoText-SemiBold.ttf` |
| `STIXTwoMath-Regular.ttf` | `STIXTwoMath-Regular.ttf` |

Each was trimmed with fontTools' subsetter to the Unicode ranges the figures can use, its hinting dropped and its
`kern`, `liga`, `ccmp` and `locl` features kept: Basic Latin, Latin-1 and Latin Extended-A, Greek, general
punctuation, super- and subscripts, letterlike symbols and the minus sign in the text faces, and in the math face
those and arrows, mathematical operators, technical symbols and geometric shapes. Each figure's SVG embeds a further
WOFF subset of only the glyphs it draws, and the demo video's frames are set from the same files.

The SemiBold face, which sets only the wordmark, was trimmed to the text faces' ranges in the same way, with
fontTools 4.67.0:

```sh
fonttools subset STIXTwoText-SemiBold.ttf --unicodes="U+0020-007E,U+00A0-00FF,U+0100-017F,U+0370-03FF,U+2000-206F,U+2070-209F,U+20AC,U+2100-214F,U+2212" \
  --no-hinting --layout-features='kern,liga,ccmp,locl' --name-IDs='*' --output-file=STIXTwoText-SemiBold.ttf
```
