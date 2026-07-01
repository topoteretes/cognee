# Personal memory sample data

This dataset is a small personal-memory corpus for a life-assistant or journaling
notebook. The target person is Veljko, represented as `veljko@topoteretes.com`
and tagged in Slack as `@veljko@topoteretes.com`.

Slack files follow the same lightweight format as `sample_data/slack_*.txt`:

```text
# #channel-name: short thread title
[person@example.com, 2026-06-10T09:30] message text
```

Email files are plain-text thread exports with From, To, Cc, Date, and Subject
headers.

Useful notebook questions:

- What open items does Veljko still need to respond to by EOD?
- Which messages tagged Veljko and did not receive a response from him?
- Which items did Veljko answer, but still require more work?
- Which requests can be ignored because Veljko already handled them?
