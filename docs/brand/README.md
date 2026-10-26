# Logo and social preview

The files here are drawn by code: [`flatshare_bench.brand`](../../python/flatshare_bench/src/flatshare_bench/brand.py)
writes them (`make brand`), and the bench's tests fail when one differs from what the generator draws. Change the
generator, not the files.

| File | Use |
|---|---|
| `logo.svg`, `logo-dark.svg` | The mark and the wordmark, for light and dark pages (the README's header) |
| `mark.svg`, `mark-dark.svg` | The mark alone, for an avatar or a document's header |
| `mark-small.svg`, `mark-small-dark.svg` | The mark without its waveform, for 16 to 24 px, such as a favicon |
| `social-preview.png` | The repository's social preview, 1280 × 640 (GitHub: Settings, General, Social preview) |
| `social-preview.svg` | The preview's source, from which the PNG is rendered |

**The mark** is a KV pool of nine blocks whose middle column belongs to a voice turn: the coding agents' blocks are
gray, and the room the scheduler makes for the voice turn is green, as the [method figure](../figures/method.svg)
colors them. The wordmark is set in STIX Two Text SemiBold, the figures' typeface, with every glyph outlined so that
it renders the same everywhere.

**Colors**: the lane `#257a58`, the figures' green, on both themes; the blocks `#8c959f` on light pages and
`#68717d` on dark ones, each at least 3:1 against its page; the wordmark the figures' ink, `#1f2328` and `#e6edf3`.

**In use**, keep at least a block's width of clear space around the logo, show the logo at least 120 px wide and
the mark at least 16 px, and use the small mark below 32 px. Do not recolor, stretch, rotate or outline it, or add
shadows or gradients.
