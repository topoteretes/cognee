import Link from "next/link";
import { BackIcon, CheckIcon } from "@/ui/Icons";
import { CTAButton, NeutralButton } from "@/ui/elements";
import Header from "@/ui/Layout/Header";

export default function Plan() {
  return (
    <>
      <div className="absolute top-0 right-0 bottom-0 left-0 flex flex-row gap-2.5">
        <div className="flex-1/5 bg-gray-100 h-full"></div>
        <div className="flex-3/5 h-full flex flex-row gap-2.5">
          <div className="flex-1/3 bg-gray-100 h-full"></div>
          <div className="flex-1/3 bg-gray-100 h-full"></div>
          <div className="flex-1/3 bg-gray-100 h-full"></div>
        </div>
        <div className="flex-1/5 bg-gray-100 h-full"></div>
      </div>

      <Header />

      <div className="relative flex flex-row items-start justify-stretch gap-2.5">
        <div className="flex-1/5 h-full">
          <Link href="/dashboard" className="py-4 px-5 flex flex-row items-center gap-5">
            <BackIcon />
            <span>back</span>
          </Link>
        </div>

        <div className="flex-3/5">
          <div className="grid grid-cols-3 gap-x-2.5">
            <div className="pt-13 py-4 px-5 mb-2.5 rounded-tl-xl rounded-tr-xl bg-white h-full">
              <div>Basic</div>
              <div className="text-3xl mb-4 font-bold">Free</div>
            </div>

            <div className="pt-13 py-4 px-5 mb-2.5 rounded-tl-xl rounded-tr-xl bg-white h-full">
              <div>On-prem Subscription</div>
              <div className="mb-4"><span className="text-3xl font-bold">$2470</span><span className="text-gray-400"> /per month</span></div>
              <div className="mb-9"><span className="font-bold">Save 20% </span>yearly</div>
            </div>

            <div className="pt-13 py-4 px-5 mb-2.5 rounded-tl-xl rounded-tr-xl bg-white h-full">
              <div>Cloud Subscription</div>
              <div className="mb-4"><span className="text-3xl font-bold">$25</span><span className="text-gray-400"> /per month</span></div>
              <div className="mb-9 text-gray-400">(beta pricing)</div>
            </div>

            <div className="bg-white rounded-bl-xl rounded-br-xl h-full">
              <div className="mb-1 invisible">Everything in the free plan, plus...</div>
              <div className="flex flex-col gap-3 mb-28">
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />License to use Cognee open source</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Cognee tasks and pipelines</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Custom schema and ontology generation</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Integrated evaluations</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />More than 28 data sources supported</div>
              </div>
            </div>

            <div className="bg-white rounded-bl-xl rounded-br-xl h-full">
              <div className="mb-1 text-gray-400">Everything in the free plan, plus...</div>
              <div className="flex flex-col gap-3 mb-10">
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />License to use Cognee open source and Cognee Platform</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />1 day SLA</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />On-prem deployment</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Hands-on support</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Architecture review</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Roadmap prioritization</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Knowledge transfer</div>
              </div>
            </div>

            <div className="bg-white rounded-bl-xl rounded-br-xl h-full">
              <div className="mb-1 text-gray-400">Everything in the free plan, plus...</div>
              <div className="flex flex-col gap-3 mb-10">
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Fully hosted cloud platform</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Multi-tenant architecture</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Comprehensive API endpoints</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Automated scaling and parallel processing</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Ability to group memories per user and domain</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />Automatic updates and priority support</div>
                <div className="flex flex-row gap-2"><CheckIcon className="mt-1 shrink-0" />1 GB ingestion + 10,000 API calls</div>
              </div>
            </div>

            <div className="pt-4 pb-14 mb-2.5">
              <NeutralButton>Try for free</NeutralButton>
            </div>

            <div className="pt-4 pb-14 mb-2.5">
              <CTAButton>Talk to us</CTAButton>
            </div>

            <div className="pt-4 pb-14 mb-2.5">
              <NeutralButton>Sign up for Cogwit Beta</NeutralButton>
            </div>
          </div>

          <div className="grid grid-cols-4 py-4 px-5 bg-[rgba(255,255,255,0.5)] mb-12">
            <div>Feature Comparison</div>
            <div className="text-center">Basic</div>
            <div className="text-center">On-prem</div>
            <div className="text-center">Cloud</div>

            <div className="border-b-[1px] border-b-gray-100 py-3">Data Sources</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">28+</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">28+</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">28+</div>

            <div className="border-b-[1px] border-b-gray-100 py-3">Deployment</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">Self-hosted</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">On-premise</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">Cloud</div>

            <div className="border-b-[1px] border-b-gray-100 py-3">API Calls</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">Limited</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">Unlimited</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">10,000</div>

            <div className="border-b-[1px] border-b-gray-100 py-3">Support</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">Community</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">Hands-on</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">Priority</div>

            <div className="border-b-[1px] border-b-gray-100 py-3">SLA</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">—</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">1 day</div>
            <div className="text-center border-b-[1px] border-b-gray-100 py-3">Standard</div>
          </div>

          <div className="grid grid-cols-2 gap-x-2.5 gap-y-2.5 mb-12">
            <div className="bg-[rgba(255,255,255,0.5)] py-4 px-5">
              <div>Can I change my plan anytime?</div>
              <div className="text-gray-500 mt-6">Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately.</div>
            </div>
            <div className="bg-[rgba(255,255,255,0.5)] py-4 px-5">
              <div>What happens to my data if I downgrade?</div>
              <div className="text-gray-500 mt-6">Your data is preserved, but features may be limited based on your new plan constraints.</div>
            </div>
            <div className="bg-[rgba(255,255,255,0.5)] py-4 px-5">
              <div>Do you offer educational discounts?</div>
              <div className="text-gray-500 mt-6">Yes, we offer special pricing for educational institutions and students. Contact us for details.</div>
            </div>
            <div className="bg-[rgba(255,255,255,0.5)] py-4 px-5">
              <div>Is there a free trial for paid plans?</div>
              <div className="text-gray-500 mt-6">All new accounts start with a 14-day free trial of our Pro plan features.</div>
            </div>
          </div>
        </div>

        <div className="flex-1/5 h-full text-center flex flex-col self-end mb-12">
          <span className="text-sm mb-2">Need a custom solution?</span>
          <CTAButton>Contact us</CTAButton>
        </div>
      </div>
    </>
  );
}
