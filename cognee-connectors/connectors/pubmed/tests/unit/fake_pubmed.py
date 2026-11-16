"""In-memory stand-in for NCBI E-utilities, shaped like its real answers.

An endpoint the connector does not use is not answered: a call to it raises
``AssertionError`` and fails the test that made it. The query-term matching is a
deliberately small esearch dialect: a bare term matches as a case-insensitive substring
of title and abstract, and a ``[field]`` suffix narrows nothing further — enough to
exercise the query filter without a query parser.
"""

import re

from cognee_connector_pubmed.source import EFETCH_PATH, ESEARCH_PATH

# What the fixture leaves out of every article unless a test asks for it, so a rendering
# of something nobody set cannot pass a test by accident.
_TEMPLATE = {
    "authors": ["Smith J", "Doe A"],
    "journal": "Journal of Memory Studies",
    "year": "2026",
    "doi": "10.1000/example",
    "types": ["Journal Article"],
    "keywords": ["sleep", "memory"],
    "abstract": [("BACKGROUND", "Sleep helps memory."), ("RESULTS", "It really does.")],
    "edat": "2026/01/01",
}

_XML = """<PubmedArticle>
  <MedlineCitation>
    <PMID>{pmid}</PMID>
    <KeywordList>{keywords}</KeywordList>
    <Article>
      <Journal>
        <Title>{journal}</Title>
        <JournalIssue><PubDate><Year>{year}</Year></PubDate></JournalIssue>
      </Journal>
      <ArticleTitle>{title}</ArticleTitle>
      <AuthorList>{authors}</AuthorList>
      <Abstract>{abstract}</Abstract>
      <PublicationTypeList>{types}</PublicationTypeList>
    </Article>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList><ArticleId IdType="doi">{doi}</ArticleId>
    <ArticleId IdType="pubmed">{pmid}</ArticleId></ArticleIdList>
    <History><PubMedPubDate PubStatus="entrez">
      <Year>{edat_year}</Year><Month>{edat_month}</Month><Day>{edat_day}</Day>
    </PubMedPubDate></History>
  </PubmedData>
</PubmedArticle>"""


def article_xml(article: dict) -> str:
    authors = "".join(
        f"<Author><LastName>{name.split(' ', 1)[0]}</LastName>"
        f"<Initials>{name.split(' ', 1)[1] if ' ' in name else ''}</Initials></Author>"
        if not name.endswith("Group")
        else f"<Author><CollectiveName>{name}</CollectiveName></Author>"
        for name in article["authors"]
    )
    keywords = "".join(f"<Keyword>{keyword}</Keyword>" for keyword in article["keywords"])
    types = "".join(f"<PublicationType>{kind}</PublicationType>" for kind in article["types"])
    abstract = "".join(
        f'<AbstractText Label="{label}">{text}</AbstractText>'
        if label
        else f"<AbstractText>{text}</AbstractText>"
        for label, text in article["abstract"]
    )
    year, month, day = article["edat"].split("/")
    return _XML.format(
        pmid=article["pmid"],
        title=article["title"],
        journal=article["journal"],
        year=article["year"],
        doi=article["doi"],
        authors=authors,
        keywords=keywords,
        types=types,
        abstract=abstract,
        edat_year=year,
        edat_month=month,
        edat_day=day,
    )


class FakePubMed:
    """The two-step E-utilities flow: esearch lists ids, efetch returns article XML."""

    def __init__(self):
        self.articles: dict[str, dict] = {}
        # In the index but missing from edat listings, like a record the index skipped.
        self.hidden: set[str] = set()
        # Removed from the index: uid lookups answer 0, like a publisher-requested removal.
        self.removed: set[str] = set()
        self.total_override: int | None = None
        self._fail_after: int | None = None
        # len(self.calls) when fail_after was last armed: fail_after counts the calls
        # made after it is set, so a test arms it per run however many calls came before.
        self._fail_armed_at = 0
        self.fail_with: Exception | None = None
        self.fail_paths: dict[str, Exception] = {}
        self.calls: list[tuple[str, dict]] = []
        self.bodies: list[dict] = []

    @property
    def fail_after(self) -> int | None:
        return self._fail_after

    @fail_after.setter
    def fail_after(self, value: int | None) -> None:
        self._fail_after = value
        self._fail_armed_at = len(self.calls)

    # -- seeding -----------------------------------------------------------
    def add_article(self, pmid: str, title: str, **fields) -> dict:
        self.articles[pmid] = {**_TEMPLATE, "pmid": pmid, "title": title, **fields}
        return self.articles[pmid]

    def count(self, path: str, *, with_uid: bool | None = None) -> int:
        calls = [params for called, params in self.calls if called == path]
        if with_uid is not None:
            calls = [p for p in calls if p.get("term", "").endswith("[uid]") == with_uid]
        return len(calls)

    # -- the api -----------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        as_text: bool = False,
        check_active: None = None,
    ):
        params = params or {}
        self.calls.append((path, dict(params)))
        self.bodies.append(dict(params))
        if path in self.fail_paths:
            raise self.fail_paths[path]
        if (
            self._fail_after is not None
            and len(self.calls) - self._fail_armed_at > self._fail_after
        ):
            raise self.fail_with
        if path == ESEARCH_PATH:
            return self._esearch(params)
        if path == EFETCH_PATH:
            assert as_text, "efetch is the XML endpoint: the source must ask for text"
            return self._efetch(params)
        raise AssertionError(f"unexpected call {method} {path}")

    def _esearch(self, params: dict):
        term = params.get("term")
        if term and term.endswith("[uid]"):
            pmid = term[: -len("[uid]")]
            indexed = pmid in self.articles and pmid not in self.removed
            return {"esearchresult": {"count": "1" if indexed else "0", "idlist": []}}
        records = [
            article
            for pmid, article in sorted(self.articles.items(), key=lambda item: int(item[0]))
            if pmid not in self.hidden
            and pmid not in self.removed
            and self._matches(article, term)
            and self._in_window(article, params)
        ]
        count = self.total_override if self.total_override is not None else len(records)
        retstart = int(params.get("retstart", 0))
        retmax = int(params.get("retmax", len(records) or 1))
        page = records[retstart : retstart + retmax]
        return {
            "esearchresult": {
                "count": str(count),
                "idlist": [article["pmid"] for article in page],
            }
        }

    def _matches(self, article: dict, term: str | None) -> bool:
        if not term:
            return True
        bare = re.sub(r"\[[^\]]*\]", "", term).strip().strip('"').lower()
        haystack = f"{article['title']} {' '.join(text for _, text in article['abstract'])}".lower()
        return bare in haystack

    def _in_window(self, article: dict, params: dict) -> bool:
        if params.get("datetype") != "edat":
            return True
        mindate = params.get("mindate")
        maxdate = params.get("maxdate")
        edat = article["edat"]
        return (not mindate or edat >= mindate) and (not maxdate or edat <= maxdate)

    def _efetch(self, params: dict) -> str:
        ids = params.get("id", "").split(",")
        missing = [pmid for pmid in ids if pmid not in self.articles]
        assert not missing, f"efetch asked for unknown PMIDs: {missing}"
        body = "".join(article_xml(self.articles[pmid]) for pmid in ids)
        return f"<PubmedArticleSet>{body}</PubmedArticleSet>"


class _HTTPResponse:
    def __init__(self, status: int, payload, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def serve_over_http(fake: FakePubMed, *, keys: list[str] | None = None):
    """A stand-in for ``requests.request`` that answers from ``fake`` the way NCBI's HTTP
    API does. ``keys`` collects the ``api_key`` param of every request."""

    def request(method, url, *, params=None, timeout=None, **_):
        params = params or {}
        if keys is not None:
            keys.append(params.get("api_key"))
        path = url.removeprefix("https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
        payload = fake.request(method, path, params=params, as_text=True)
        if isinstance(payload, str):
            return _HTTPResponse(200, None, text=payload)
        return _HTTPResponse(200, payload)

    return request
