"use client";

import { useBoolean, useOutsideClick } from "@/utils";
import { MenuIcon } from "@/ui/icons";
import { IconButton } from "@/ui/elements";
import classNames from 'classnames';

interface PopupMenuProps {
  children: React.ReactNode;
  triggerElement?: React.ReactNode;
  triggerClassName?: string;
  openToRight?: boolean;
}

export default function PopupMenu({ triggerElement, triggerClassName, children, openToRight = false }: PopupMenuProps) {
  const {
    value: isMenuOpen,
    setFalse: closeMenu,
    toggle: toggleMenu,
  } = useBoolean(false);

  const menuRootRef = useOutsideClick<HTMLDivElement>(closeMenu);

  return (
    <div className="relative inline-block" ref={menuRootRef}>
      <IconButton as="div" className={triggerClassName} onClick={toggleMenu}>
        {triggerElement || <MenuIcon />}
      </IconButton>

      {isMenuOpen && (
        <div
          className={classNames(
            "absolute top-full mt-1 flex flex-col p-1.5",
            "bg-white border border-cognee-border-light rounded-[10px]",
            "shadow-[0px_8px_30px_rgba(0,0,0,0.08)] z-10",
            {
              "left-0": openToRight,
              "right-0": !openToRight,
            },
          )}
        >
          {children}
        </div>
      )}
    </div>
  );
};
