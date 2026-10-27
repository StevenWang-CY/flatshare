"""The technical report's figures (docs/REPORT.md), docs/figures/report-*.svg: :mod:`.data` reads what each plots from
the committed records, :mod:`.charts` holds the chart parts they share on :mod:`flatshare_bench.figure_style`, and
:mod:`.figures` draws one figure a function (``python -m flatshare_bench.report.figures DIR``, ``make
report-figures``). The figures carry no title, which the report's caption under each gives with its key; the art
carries panel letters, axes, direct labels and, where labels would collide, a compact legend."""
