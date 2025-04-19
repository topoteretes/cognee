import Link from 'next/link';
import { Stack } from 'ohmy-ui';
import { DiscordIcon, GithubIcon } from '@/ui/Icons';
// import { TextLogo } from '@/ui/App';
import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <Stack orientation="horizontal" gap="between">
        <div className={styles.leftSide}>
          {/* <TextLogo width={92} height={24} /> */}
        </div>
        <div className={styles.rightSide}>
          <Link target="_blank" href="https://github.com/topoteretes/cognee">
            <GithubIcon color="white" />
          </Link>
          <Link target="_blank" href="https://discord.gg/m63hxKsp4p">
            <DiscordIcon color="white" />
          </Link>
        </div>
      </Stack>
    </footer>
  );
}
