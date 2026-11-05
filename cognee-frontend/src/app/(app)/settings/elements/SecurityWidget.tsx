"use client";

import { trackPageEvent } from "@/modules/analytics";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import styles from "../SettingsPage.module.css";

export default function SecurityWidget() {
  const cloud = isCloudEnvironment();

  return (
    <section className={styles.card} aria-labelledby="security-heading">
      <h2 id="security-heading">Security</h2>
      <p className={styles.subtitle}>Manage your account security and session</p>
      <div className={styles.section}>
        <h3 className={styles.label}>Password</h3>
        <p className={styles.subtitle}>
          {cloud
            ? "Your password is managed by your authentication provider."
            : "Your credentials are managed by your local Cognee instance."}
        </p>
      </div>
      <div className={styles.section}>
        <h3 className={styles.label}>Session</h3>
        <p className={styles.subtitle}>Sign out of your account on this device.</p>
        <a
          href={cloud ? "/api/signout" : "/api/local-signout"}
          className={styles.logout}
          onClick={() => trackPageEvent({ pageName: "Settings", eventName: "sign_out" })}
        >
          Log out
        </a>
      </div>
    </section>
  );
}
