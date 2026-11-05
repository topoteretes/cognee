/**
 * The values once each, in first-seen order, comma separated. For analytics
 * properties that list a value per file: a 200-file upload of one type would
 * otherwise repeat it 200 times, and a big enough list makes the event too
 * large to send.
 */
export function joinDistinct(values: string[]): string {
  return [...new Set(values)].join(",");
}
