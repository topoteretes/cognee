import classNames from "classnames";
import styles from "./LoadingIndicator.module.css";

export default function LoadingIndicator({ color = "" }) {
  return <div className={classNames(styles.loadingIndicator, `!border-${color}`)} />
}
