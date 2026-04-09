import classNames from "classnames";
import styles from "./Title.module.css";

interface TitleProps {
  children: React.ReactNode;
  className?: string;
}

function Title({ children, className, ...rest }: TitleProps) {
  return (
    <span
      {...rest}
      className={classNames(className, styles.title)}
    >
      {children}
    </span>
  );
}

export default Title;
