import Link from "next/link";
import { DiscordIcon, GithubIcon } from "@/ui/Icons";

interface FooterProps {
  children?: React.ReactNode;
}

export default function Footer({ children }: FooterProps) {
  return (
    <footer className="pt-6 pb-6 flex flex-row items-center justify-between">
      <div>
        {children}
      </div>

      <div className="flex flex-row gap-4">
        <Link target="_blank" href="https://github.com/topoteretes/cognee">
          <GithubIcon color="black" />
        </Link>
        <Link target="_blank" href="https://discord.gg/m63hxKsp4p">
          <DiscordIcon color="black" />
        </Link>
      </div>
    </footer>
  );
}
