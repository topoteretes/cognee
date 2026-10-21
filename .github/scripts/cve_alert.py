"""Post CVEs that mention cognee to Slack, once each.

NVD (which mirrors CVE.org within an hour or two) is asked for every CVE whose
description matches the keyword and that was published since the start of this
workflow's last successful run, up to the start of this one. A skipped or failed
run therefore delays alerts instead of losing them, and back-to-back runs never
report the same CVE twice. Both ends are shifted into the past by NVD_INDEX_LAG
as a precaution: NVD does not document how soon a record becomes searchable
after the time it stamps as `published`, and a CVE stamped just before a run but
not yet visible to it would otherwise fall between two runs.

Input (env):
  GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID,
  GITHUB_WORKFLOW_REF, GITHUB_REF_NAME             read this workflow's runs.
  SLACK_BOT_TOKEN, SLACK_CHANNEL_ID                where alerts are posted.
  CVE_KEYWORD                                      NVD keyword (default cognee).
"""

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
GITHUB_API = "https://api.github.com"
HTTP_TIMEOUT = 30  # seconds
FIRST_RUN_LOOKBACK = timedelta(days=1)
NVD_INDEX_LAG = timedelta(hours=1)
# NVD rejects publish-date ranges longer than 120 days.
NVD_MAX_RANGE = timedelta(days=119)
NVD_PAGE_SIZE = 2000  # NVD's maximum
MAX_DESCRIPTION = 1500  # characters of a CVE description quoted in Slack


def main():
    until = this_run_start() - NVD_INDEX_LAG
    last = last_successful_run_start()
    since = last - NVD_INDEX_LAG if last else until - FIRST_RUN_LOOKBACK
    since = max(since, until - NVD_MAX_RANGE)
    cves = fetch_nvd_cves(os.environ.get("CVE_KEYWORD", "cognee"), since, until)
    print(f"{len(cves)} CVE(s) published {since:%Y-%m-%d %H:%M} .. {until:%Y-%m-%d %H:%M} UTC")
    for cve in cves:
        post_to_slack(format_alert(cve))


def this_run_start():
    run = github_get(f"/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    return parse_time(run["run_started_at"])


def last_successful_run_start():
    # GITHUB_WORKFLOW_REF is "<owner>/<repo>/.github/workflows/<file>@<ref>".
    workflow = os.environ["GITHUB_WORKFLOW_REF"].split("@")[0].rsplit("/", 1)[1]
    branch = urllib.parse.quote(os.environ["GITHUB_REF_NAME"])
    runs = github_get(f"/actions/workflows/{workflow}/runs?status=success&branch={branch}")
    latest = runs["workflow_runs"][:1]
    return parse_time(latest[0]["run_started_at"]) if latest else None


def github_get(path):
    url = f"{GITHUB_API}/repos/{os.environ['GITHUB_REPOSITORY']}{path}"
    headers = {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as resp:
        return json.load(resp)


def parse_time(value):
    # GitHub stamps end in "Z"; NVD stamps carry no zone but are UTC.
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def fetch_nvd_cves(keyword, since, until):
    query = urllib.parse.urlencode(
        {
            "keywordSearch": keyword,
            "pubStartDate": nvd_time(since),
            "pubEndDate": nvd_time(until),
            "resultsPerPage": NVD_PAGE_SIZE,
        }
    )
    with urllib.request.urlopen(f"{NVD_URL}?{query}", timeout=HTTP_TIMEOUT) as resp:
        data = json.load(resp)
    if data["totalResults"] > len(data["vulnerabilities"]):
        raise RuntimeError(f"NVD returned more than one page ({data['totalResults']} results)")
    return [item["cve"] for item in data["vulnerabilities"]]


def nvd_time(moment):
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def format_alert(cve):
    cve_id = cve["id"]
    published = parse_time(cve["published"])
    description = next(d["value"] for d in cve["descriptions"] if d["lang"] == "en")
    if len(description) > MAX_DESCRIPTION:
        description = description[:MAX_DESCRIPTION] + "…"
    description = slack_escape(description)
    return (
        f":rotating_light: New CVE mentioning cognee: "
        f"*<https://www.cve.org/CVERecord?id={cve_id}|{cve_id}>*\n"
        f"Severity: {severity(cve)} · published {published:%Y-%m-%d %H:%M} UTC\n"
        f"> {description}"
    )


def slack_escape(text):
    # Anyone can get a CVE filed, and Slack reads <...> in a message as a link or
    # a mention (<!channel>). "&" goes first so the other escapes are not re-escaped.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def severity(cve):
    # Newest CVSS version first; NVD's own score and the CNA's share one list.
    for version in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        for metric in cve.get("metrics", {}).get(version, []):
            data = metric["cvssData"]
            # CVSS v2 keeps the rating next to cvssData instead of inside it.
            rating = data.get("baseSeverity") or metric.get("baseSeverity", "?")
            return f"{rating} {data['baseScore']}"
    return "not scored yet"


def post_to_slack(text):
    body = {"channel": os.environ["SLACK_CHANNEL_ID"], "text": text, "unfurl_links": False}
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as resp:
        result = json.load(resp)
    # Slack answers HTTP 200 even on failure; the verdict is in "ok".
    if not result.get("ok"):
        raise RuntimeError(f"Slack chat.postMessage failed: {result.get('error')}")


if __name__ == "__main__":
    main()
