import classNames from "classnames";
import { CaretIcon } from "../Icons";

export interface AccordionProps {
  isOpen: boolean;
  title: React.ReactNode;
  openAccordion: () => void;
  closeAccordion: () => void;
  tools?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  contentClassName?: string;
  switchCaretPosition?: boolean;
}

export default function Accordion({ title, tools, children, isOpen, openAccordion, closeAccordion, className, contentClassName, switchCaretPosition = false }: AccordionProps) {
  return (
    <div className={classNames("flex flex-col", className)}>
      <div className="flex flex-row justify-between items-center">
        <button className={classNames("flex flex-row items-center pr-2", switchCaretPosition ? "gap-1.5" : "gap-4")} onClick={isOpen ? closeAccordion : openAccordion}>
          {switchCaretPosition ? (
            <>
              <CaretIcon className={classNames("transition-transform", isOpen ? "rotate-360" : "rotate-270")} />
              {title}
            </>
          ) : (
            <>
              {title}
              <CaretIcon className={classNames("transition-transform", isOpen ? "rotate-0" : "rotate-180")} />
            </>
          )}
        </button>
        {tools}
      </div>

      {isOpen && (
        <div className={classNames("grid transition-[grid-template-rows] duration-300 ease-in-out [grid-template-rows:0fr]", contentClassName, {
          "[grid-template-rows:1fr]": isOpen,
        })}>
          {children}
        </div>
      )}
    </div>
  );
}
