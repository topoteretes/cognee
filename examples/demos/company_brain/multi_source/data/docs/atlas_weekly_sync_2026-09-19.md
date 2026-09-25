# Atlas weekly sync — 19 September 2026

Attendees: Marco Rossi, Dana Kim, Sam Okafor, Grace Liu

## Missing products in search (T-1041)

Dana Kim traced the missing products to the Brightline Retail catalog feed. Since
15 September the feed sends new products in a second file, and the Atlas indexer
only reads the first one.

Decision: Dana Kim will change the indexer to read every file in the feed and
re-index the catalog. Target date is 26 September. Until then, Brightline Retail
can trigger a manual re-index from the admin page.

## Autocomplete latency (T-1042)

Sam Okafor found that the autocomplete cache is evicted every time the catalog is
re-indexed. Sam will move the cache to the Beacon-monitored Redis cluster and add
a latency alert.

## Customer

Grace Liu reported that Brightline Retail's contract renewal in October depends on
Atlas search quality. Grace will send Brightline Retail an update after Dana Kim's
fix ships.

## Action items

- Dana Kim: read every catalog feed file, re-index Brightline Retail (due 26 September)
- Sam Okafor: move the autocomplete cache, add a latency alert in Beacon
- Grace Liu: send Brightline Retail a status update
