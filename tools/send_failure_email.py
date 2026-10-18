#!/usr/bin/env python3
"""Send a workflow failure notification email."""

from __future__ import annotations

import os
import smtplib
import textwrap
from email.message import EmailMessage


def main() -> None:
    subject = "dev_previous_day_commits.yml failed"
    body = textwrap.dedent(
        f"""\
        The previous-day dev commits documentation workflow failed.

        Repository: {os.environ["GITHUB_REPOSITORY"]}
        Run: {os.environ["GITHUB_RUN_NUMBER"]}
        Branch: {os.environ["GITHUB_REF_NAME"]}
        Actor: {os.environ["GITHUB_ACTOR"]}

        prepare-merged-branches: {os.environ["PREPARE_RESULT"]}
        create-docs-prs: {os.environ["CREATE_DOCS_PRS_RESULT"]}

        View the run:
        {os.environ["RUN_URL"]}
        """
    )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = os.environ["NOTIFICATION_EMAIL_SENDER"]
    message["To"] = os.environ["NOTIFICATION_EMAIL_RECEIVER"]
    message.set_content(body)

    smtp_port = int(os.environ["SMTP_PORT"])
    use_tls = os.environ["SMTP_USE_TLS"].lower() in {"1", "true", "yes"}

    with smtplib.SMTP(os.environ["SMTP_SERVER"], smtp_port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(message)


if __name__ == "__main__":
    main()
