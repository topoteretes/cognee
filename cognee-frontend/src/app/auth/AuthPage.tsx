import { Spacer, Stack, Text } from 'ohmy-ui';
import { TextLogo } from '@/ui/App';
import Footer from '@/ui/Partials/Footer/Footer';

import styles from './AuthPage.module.css';
import { Divider } from '@/ui/Layout';
import SignInForm from '@/ui/Partials/SignInForm/SignInForm';

export default function AuthPage() {
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
          <h1><Text size="large">Sign in</Text></h1>
          <SignInForm />
        </Stack>
      </div>
      <Spacer inset horizontal="3" wrap>
        <Footer />
      </Spacer>
    </main>
  )
}
