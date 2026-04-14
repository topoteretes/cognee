import styles from "./LoadingIndicator.module.css";

export default function LoadingIndicator({ color = "" }) {
  return <div
    className={styles.loadingIndicator}
    style={{
      borderLeftColor: color,
      borderRightColor: color,
    }}
  />
}
