"use client";

interface SurveyThanksStepProps {
  message: string;
}

export default function SurveyThanksStep({ message }: SurveyThanksStepProps): React.ReactElement {
  return (
    <div className="flex flex-col items-center gap-2.5 px-0 py-5 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-full border-[1.5px] border-emerald-500/50 bg-emerald-500/10 text-emerald-500">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 6L9 17l-5-5" />
        </svg>
      </div>
      <h3 className="m-0 text-sm font-bold text-cognee-dark">Got it, thanks.</h3>
      <p className="m-0 max-w-[280px] text-[12.5px] text-cognee-secondary">{message}</p>
    </div>
  );
}
