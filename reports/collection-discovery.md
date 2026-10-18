# Multi-source collection discovery

Discovery and live validation on October 18, 2026:

| Source | Discovered recipe records/pages | Collected at pilot snapshot |
|---|---:|---:|
| BrewDog via PunkAPI snapshot | 415 | 415 |
| Brewer's Friend public sitemaps | 327,867 | 27 |
| AHA recipe sitemaps | 1,528 | 10, including reused pilot records |

The first snapshot contains 452 distinct represented recipe records. No exact-content duplicates were found in this early sample; eight similar-title groups were flagged for review. Counts will change while the full queue is collected. Current local counts are in `data/collection/report.json` and the SQLite queue; the table above is a fixed pilot snapshot.

The first 30 AHA URLs yielded six accessible instructional records and 24 extraction failures. Inspecting one failure showed a membership unlock page without the ingredient/directions sections. Four additional previously collected pilot records were reused locally; they were already included in the sitemap discovery count.

Brewer's Friend returned HTTP 429 after 27 successfully collected recipes. That source was paused and the cooldown persisted. The collector now waits at least five seconds between requests, honors longer Retry-After values with a minimum one-hour cooldown on 429, and stops automatic retries after three consecutive access failures. No disallowed export route or paywall bypass is used.

The full resumable queue was started locally. AHA and Brewer's Friend each use one sequential worker. AHA has a ten-second minimum interval. The Brewer's Friend queue alone requires at least 19 days at the chosen rate; outages and throttling extend that. The actual successful count will be lower than discovered URLs where recipes are private, removed or unsupported. This is not a claim that hundreds of thousands of recipes have already been downloaded.

Twenty-five unit tests pass, including source parsing, sitemap namespaces, quantity/process variant preservation, exact-duplicate provenance, persisted state and HTTP 429 behavior. Original HTML is gzip-compressed, source data stays local, and periodic exports preserve a usable partial collection. Similar content represented in different units or wording still requires curation before any final claim of cross-source uniqueness.

See `docs/COLLECTION.md` for progress, stop and resume commands. The paused overnight automation remains paused; collection is a local background process, not a scheduled task.
