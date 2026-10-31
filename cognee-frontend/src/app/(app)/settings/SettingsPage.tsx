"use client";

import ProfileWidget from "./elements/ProfileWidget";
import SecurityWidget from "./elements/SecurityWidget";
import { TrackPageView } from "@/modules/analytics";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import styles from "./SettingsPage.module.css";

export default function SettingsPage() {
  return (
    <div className={styles.page}>
      <TrackPageView page="Settings" />
      <header className={styles.header}>
        <h1>Settings</h1>
        <p>Manage your account and session.</p>
      </header>
      <div className={styles.grid}>
        {isCloudEnvironment() ? <ProfileWidget /> : (
          <section className={styles.card} aria-labelledby="account-heading">
            <h2 id="account-heading">Account</h2>
            <p className={styles.subtitle}>Your local Cognee account</p>
            <div className={styles.section}>
              <span className={styles.badge}>Local workspace</span>
              <p className={styles.subtitle}>You are signed in to your local Cognee instance. Profile editing is not available in this app.</p>
            </div>
          </section>
        )}
        <SecurityWidget />
      </div>
    </div>
  );
}
