import type { ReactElement, ReactNode } from "react";
import Link from "next/link";

const CODE = "rounded bg-white/[0.08] px-1.5 py-0.5 font-mono text-[12px] text-[var(--color-cognee-fg,#EDECEA)]";

interface GuideSection {
  title: string;
  body: ReactNode;
}

// Every claim here is checked against the connector's own behaviour rather than
// written from intent: what is read, where it lands, and who can query it are
// the three things a workspace owner is deciding on, and a guide that overstates
// any of them is worse than no guide.
const SECTIONS: GuideSection[] = [
  {
    title: "What Cognee reads",
    body: (
      <>
        Messages and threads from the channels it has been switched on for, and nothing else.
        Switching a channel on also reads the history it already has, not just new messages.
        Private channels are only available once you invite Cognee into them; it cannot see a
        private channel it is not a member of.
      </>
    ),
  },
  {
    title: "Switching a channel on",
    body: (
      <>
        Type <code className={CODE}>/invite @cognee</code> in the channel. It starts reading that
        channel straight away and posts a message saying so, where everyone in the channel can see
        it. You can also tick channels on the Integrations page; the two do the same thing.
      </>
    ),
  },
  {
    title: "Where it ends up",
    body: (
      <>
        Two brains per Slack workspace. Everything read from channels goes into{" "}
        <code className={CODE}>slack-&lt;your team id&gt;</code>; anything somebody deliberately
        remembers goes into <code className={CODE}>slack-&lt;your team id&gt;-remembered</code>,
        kept apart so removing one of those can never take a channel&apos;s history with it. Both
        are in{" "}
        <Link href="/datasets" className="text-cognee-lavender underline">
          Brain
        </Link>{" "}
        alongside the brains you build yourself, and questions search all of them.
      </>
    ),
  },
  {
    title: "Who can ask it",
    body: (
      <>
        Anyone in this Cognee workspace. That is the part worth pausing on before switching a
        channel on: a message in that channel becomes answerable for every member, so treat
        switching on <code className={CODE}>#hr</code> or{" "}
        <code className={CODE}>#compensation</code> as publishing it to the whole team.
      </>
    ),
  },
  {
    title: "How to ask in Slack",
    body: (
      <>
        Type <code className={CODE}>/cognee-recall why did we choose Neon for v2?</code> in any
        Slack channel, or just mention <code className={CODE}>@cognee</code> with your question.
        The answer is ephemeral, so only you see it, and it does not post into the channel — there
        is a button if you want to share it. You can ask the same thing from Search in this app.
      </>
    ),
  },
  {
    title: "Remembering one specific thing",
    body: (
      <>
        Pick <strong className="font-semibold text-[var(--color-cognee-fg,#EDECEA)]">Remember in Cognee</strong> from a
        message&apos;s ⋯ menu to keep that message deliberately, rather than relying on it being
        swept up with the rest of the channel. It gets a ✅ so the channel can see what is being
        kept. <code className={CODE}>/cognee-remember</code> does the same for something Slack
        never saw — a decision from a call, say. <strong className="font-semibold text-[var(--color-cognee-fg,#EDECEA)]">
          Forget in Cognee
        </strong>{" "}
        removes it again, and the ✅ comes off.
      </>
    ),
  },
  {
    title: "Switching a channel off",
    body: (
      <>
        Remove Cognee from the channel, or untick it on the Integrations page. Either stops it
        reading anything new. What was already read stays searchable: entities pulled out of those
        messages may be referenced by other memory, so removing them is not a clean undo.
      </>
    ),
  },
  {
    title: 'When it says "Needs reconnect"',
    body: (
      <>
        Slack has stopped accepting the connection, usually because the app was removed from the
        workspace or its token expired. The install is still there and nobody disconnected it, so
        the fix is Reconnect rather than Connect: it is the same authorization step, and it keeps
        the channels you already selected. Nothing new is read until you do it.
      </>
    ),
  },
  {
    title: "Who is in control",
    body: (
      <>
        One Slack workspace connects to one Cognee workspace, and only the Cognee workspace owner
        can connect or disconnect it. Channels are different:{" "}
        <strong className="font-semibold text-[var(--color-cognee-fg,#EDECEA)]">
          anyone in Slack who can invite the bot into a channel can switch that channel on
        </strong>
        , which is why it announces itself in the channel when it starts. The Integrations page is
        where you see every channel that is on, and untick any of them.
      </>
    ),
  },
];

export default function SlackGuide(): ReactElement {
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full max-w-[760px] flex-col gap-6 px-8 pt-6 pb-10">
        <div>
          <Link href="/integrations" className="text-[13px] text-[var(--color-cognee-fg,#EDECEA)]/55 hover:text-[var(--color-cognee-fg,#EDECEA)]">
            ← Integrations
          </Link>
          <h1 className="mt-3 mb-1 text-[18px] font-bold tracking-[-0.01em] text-[var(--color-cognee-fg,#EDECEA)]">
            How Slack memory works
          </h1>
          <p className="m-0 text-[14px] text-[var(--color-cognee-fg,#EDECEA)]/55">
            What Cognee reads from Slack, where it goes, and how to ask it questions.
          </p>
        </div>

        <div className="flex flex-col gap-4">
          {SECTIONS.map((section) => (
            <section
              key={section.title}
              className="rounded-xl border border-white/10 bg-white/[0.06] p-5"
            >
              <h2 className="m-0 mb-1.5 text-[14px] font-semibold text-[var(--color-cognee-fg,#EDECEA)]">
                {section.title}
              </h2>
              <p className="m-0 text-[13px] leading-[1.6] text-[var(--color-cognee-fg,#EDECEA)]/55">{section.body}</p>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
