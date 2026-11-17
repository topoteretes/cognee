"""``@cognee link``: point the asking Slack member at Sign in with Slack.

One Slack OAuth install serves an entire workspace, but each member's questions and
remembered items need to know *which* cognee account they belong to. A member links
their own: logged in to cognee, they open the link this sends, which starts Sign in with Slack
(``member_link.SlackMemberIntegration``, the generic integrations routes of the
``slack_member`` provider). Slack proves which member signed in and the browser that
started it proves which cognee user, so nothing typed into Slack or forwarded to someone
else can bind the two. The former flow minted a code here that whoever opened it
confirmed for their own account, which let a member reach another user's memory (G42).

The link reuses the generic ``integration_credentials`` table: a member-link is stored
under ``provider="slack_member"`` (distinct from ``"slack"``, the workspace-level bot
credential, so the two never collide on the ``UNIQUE(provider, provider_account_id)``
index) with ``provider_account_id="{team_id}:{slack_user_id}"`` and an empty token
payload: there is no secret to keep, only an identity pointer.

The reply is ephemeral, like every reply of this integration.
"""

from cognee.modules.integrations.slack.slack_settings import require

# Distinct from PROVIDER ("slack") on purpose — see module docstring.
MEMBER_LINK_PROVIDER = "slack_member"

# Shown wherever resolve_owner_user_id comes back None. Every entry point that reads or
# writes memory reaches that state, so the wording lives here, once.
NOT_LINKED_MESSAGE = (
    "Link your Slack account first\n"
    "Send `@cognee link` to connect it to your Cognee account, then try again."
)


def member_link_account_id(team_id: str, slack_user_id: str) -> str:
    """The synthetic ``provider_account_id`` a member-link is stored/looked up under."""
    return f"{team_id}:{slack_user_id}"


def link_message() -> str:
    """Where the member signs in with Slack: the server's start route of the member
    provider, beside the callback Slack redirects to (``SLACK_MEMBER_REDIRECT_URI``), so
    no frontend page is needed. It carries no code."""
    callback = require("member_redirect_uri").rstrip("/")
    link = callback.removesuffix("/callback") + "/start"
    return (
        "Link your Slack account\n"
        f"Open {link} in the browser where you're logged in to Cognee and sign in with "
        "Slack. Your questions and saved messages then use that account's memory."
    )
