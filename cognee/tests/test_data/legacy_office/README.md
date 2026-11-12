# Legacy Office fixtures

These are genuine Office 97 binary files, exported with LibreOffice from synthetic
documents created using python-docx, python-pptx and openpyxl. Their OLE compound-file
signature is `d0cf11e0a1b11ae1`; they are not renamed Open XML documents.

- `legacy.doc`: a heading, Unicode paragraph (`Čuvamo tekst dokumenta. Café 東京.`)
  and a Product/Amount table containing Tangerine and 42.
- `legacy.ppt.gz`: two slides, with distinct titles and a Unicode paragraph on slide two.
- `legacy.xls`: Revenue and People worksheets, a `SUM` formula with a cached value of
  42, and Unicode cells (`Čedomir`, `東京`).

The connector tests replace only Drive listing/downloads, parse these files through
the real loader engine and merge the results into a disposable DLT destination.
LibreOffice must be installed; the dedicated CI step installs Writer, Calc and Impress.

The content is synthetic text written for this repository, so there is no third-party
material in these files; they follow the repository's Apache-2.0 licence.

`legacy.ppt` is stored gzipped (`gzip -9n`) because LibreOffice's presentation template makes the
raw file 636 KB, over the 500 KB limit of the `check-added-large-files` pre-commit hook. The
test decompresses it into a temporary directory next to the other two fixtures.
