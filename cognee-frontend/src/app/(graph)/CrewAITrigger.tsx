import { fetch } from "@/utils";
import { CTAButton, Input } from "@/ui/elements";

interface CrewAIFormPayload extends HTMLFormElement {
  username1: HTMLInputElement;
  username2: HTMLInputElement;
}

export default function CrewAITrigger() {
  const handleRunCrewAI = (event: React.FormEvent<CrewAIFormPayload>) => {
    fetch("/v1/crew-ai/run", {
      method: "POST",
      body: new FormData(event.currentTarget),
    })
      .then(response => response.json())
      .then((data) => console.log(data));
  };

  return (
    <form className="w-full flex flex-row gap-2 items-center" onSubmit={handleRunCrewAI}>
      <Input type="text" placeholder="Github Username" required />
      <Input type="text" placeholder="Github Username" required />
      <CTAButton type="submit" className="whitespace-nowrap">Run CrewAI</CTAButton>
    </form>
  );
}
