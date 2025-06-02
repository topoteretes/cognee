import Link from 'next/link';

import { CTAButton, Spacer, Stack, Text } from 'ohmy-ui';
import { auth0 } from '@/modules/auth/auth0';
import { TextLogo } from '@/ui/App';
import { Divider } from '@/ui/Layout';
import Footer from '@/ui/Partials/Footer/Footer';
import AuthToken from './token/AuthToken';

import styles from './AuthPage.module.css';

export default async function AuthPage() {
  const session = await auth0.getSession();

  return (
    <main className={styles.main}>
      <Spacer inset vertical="2" horizontal="2">
        <Stack orientation="horizontal" gap="between" align="center">
          <TextLogo width={158} height={44} color="white" />
        </Stack>
      </Spacer>
      <Divider />
      <div className={styles.authContainer}>
        <Stack gap="4" style={{ width: '100%' }}>
          <h1><Text size="large">Welcome to cognee</Text></h1>
          {session ? (
            <Stack gap="4">
              <Text>Hello, {session.user.name}!</Text>
              <AuthToken />
              <Link href="/auth/logout">
                <CTAButton>
                  Log out
                </CTAButton>
              </Link>
            </Stack>
          ) : (
            <>
              <Link href="/auth/login?screen_hint=signup">
                <CTAButton>
                  Sign up
                </CTAButton>
              </Link>

              <Link href="/auth/login">
                <CTAButton>
                  Log in
                </CTAButton>
              </Link>
            </>
          )}
        </Stack>
      </div>
      <Spacer inset horizontal="3" wrap>
        <Footer />
      </Spacer>
    </main>
  )
}
