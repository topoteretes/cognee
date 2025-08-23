import classNames from "classnames";
import { CaretIcon } from "../Icons";

interface AccordioProps {
  isOpen: boolean;
  title: React.ReactNode;
  openAccordion: () => void;
  closeAccordion: () => void;
  tools?: React.ReactNode;
  children: React.ReactNode;
}

export default function Accordion({ title, tools, children, isOpen, openAccordion, closeAccordion }: AccordioProps) {
  return (
    <div className="flex flex-col">
      <div className="flex flex-row justify-between items-center">
        <button className="flex flex-row gap-4 items-center pr-2" onClick={isOpen ? closeAccordion : openAccordion}>
          {title}
          <CaretIcon className={classNames("transition-transform", isOpen ? "rotate-0" : "rotate-180")} />
        </button>
        {tools}
      </div>

      <div className={classNames("grid transition-[grid-template-rows] duration-300 ease-in-out [grid-template-rows:0fr]", {
        "[grid-template-rows:1fr]": isOpen,
      })}>
        <div className="overflow-hidden">
          {children}
        </div>
      </div>
    </div>
  );
}
